"""Stream a bounded portion of Lichess's CC0 eval export into compact shards."""
import argparse
from collections import deque
from contextlib import contextmanager
import io
import json
import multiprocessing
import os
from pathlib import Path
import time

import chess
import numpy as np
import requests
import zstandard

from .features import COMPACT_DTYPE, VERSION, encode, position_key, target_from_white

DEFAULT_SOURCE = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


@contextmanager
def open_lines(source):
    with _source(source) as raw:
        if source.endswith(".zst"):
            with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text:
                    yield text
        else:
            with io.TextIOWrapper(raw, encoding="utf-8") as text:
                yield text


@contextmanager
def _source(source):
    if source.startswith(("https://", "http://")):
        with requests.get(source, stream=True, timeout=(30, 120), headers={"User-Agent": "scratch-chess/0.1"}) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            yield response.raw
    else:
        with open(source, "rb") as stream:
            yield stream


def parse_record(obj, min_depth=16):
    fen = obj["fen"]
    if len(fen.split()) == 4:
        fen += " 0 1"
    board = chess.Board(fen)
    if not board.is_valid() or board.is_game_over() or board.is_check():
        return None
    evaluations = [e for e in obj["evals"] if e.get("pvs") and e.get("depth", 0) >= min_depth]
    if not evaluations:
        return None
    best = max(evaluations, key=lambda e: (e["depth"], e.get("knodes", 0)))
    pv = best["pvs"][0]
    # Lichess cloud evals use WHITE perspective, including black-to-move positions.
    target = target_from_white(pv.get("cp"), pv.get("mate"), board.turn)
    ids, extras = encode(board)
    return ids, extras, target, position_key(ids, extras)


def parse_batch(task):
    """Worker: parse JSON lines into compact rows plus their split keys."""
    lines, min_depth = task
    rows = np.empty(len(lines), dtype=COMPACT_DTYPE)
    keys = np.empty(len(lines), dtype=np.uint64)
    n = 0
    for line in lines:
        try:
            record = parse_record(json.loads(line), min_depth)
        except (ValueError, KeyError, TypeError):
            record = None
        if record is not None:
            ids, extras, target, key = record
            rows[n] = ids[0], extras, target  # the opponent view is derived at training time
            keys[n] = key
            n += 1
    return rows[:n], keys[:n], len(lines)


def batched_lines(lines, size, min_depth):
    batch = []
    for line in lines:
        batch.append(line)
        if len(batch) == size:
            yield batch, min_depth
            batch = []
    if batch:
        yield batch, min_depth


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() or list(out.glob("*.npy")):
        raise ValueError(f"{out} already contains data; choose a new directory to avoid mixing datasets")
    buffers = {split: np.empty(args.shard_size, dtype=COMPACT_DTYPE) for split in ("train", "val")}
    counts = {"train": 0, "val": 0}
    shards = {"train": [], "val": []}
    sizes = {"train": 0, "val": 0}
    accepted = scanned = skipped = 0
    start = time.monotonic()
    complete = False
    reason = "position_limit"

    def flush(split):
        n = counts[split]
        if not n:
            return
        name = f"{split}-{len(shards[split]):05d}.npy"
        temp = out / (name + ".tmp")
        with temp.open("wb") as f:
            np.save(f, buffers[split][:n], allow_pickle=False)
        temp.replace(out / name)
        shards[split].append({"file": name, "count": n})
        sizes[split] += n
        counts[split] = 0

    threshold = np.uint64(round(args.val_fraction * 10000))
    next_report = 250_000
    workers = getattr(args, "workers", 1)
    pool = multiprocessing.Pool(workers) if workers > 1 else None

    def ordered_results(tasks):
        """Bounded read-ahead: Pool.imap would drain the whole 20 GB stream into RAM."""
        if pool is None:
            yield from map(parse_batch, tasks)
            return
        pending = deque()
        for task in tasks:
            pending.append(pool.apply_async(parse_batch, (task,)))
            if len(pending) >= workers * 4:
                yield pending.popleft().get()
        while pending:
            yield pending.popleft().get()

    try:
        with open_lines(args.source) as lines:
            results = ordered_results(batched_lines(lines, 2000, args.min_depth))
            for rows, keys, batch_lines in results:
                scanned += batch_lines
                skipped += batch_lines - len(rows)
                take = min(len(rows), args.positions - accepted)
                rows, keys = rows[:take], keys[:take]
                accepted += take
                # No global dedup set (it would need ~30 GB of RAM for the full export). Duplicates
                # are ~0.5% and share a key, so they always land in the same split: no leakage.
                is_val = keys % np.uint64(10000) < threshold
                for split, part in (("val", rows[is_val]), ("train", rows[~is_val])):
                    while len(part):
                        n = min(len(part), args.shard_size - counts[split])
                        buffers[split][counts[split]:counts[split]+n] = part[:n]
                        counts[split] += n
                        part = part[n:]
                        if counts[split] == args.shard_size:
                            flush(split)
                if accepted >= next_report:
                    next_report += 1_000_000
                    elapsed = time.monotonic() - start
                    print(f"accepted={accepted:,}/{args.positions:,} scanned={scanned:,} positions/s={accepted/elapsed:.0f}", flush=True)
                if accepted >= args.positions:
                    break
            else:
                reason = "source_exhausted"
            if pool:  # stop workers before the source stream closes
                pool.terminate()
                pool.join()
                pool = None
        complete = True
    except KeyboardInterrupt:
        reason = "interrupted"
        print("Saving the accepted positions before stopping.", flush=True)
    finally:
        if pool:
            pool.terminate()
            pool.join()
        for split in buffers:
            flush(split)
        manifest = {
            "feature_version": VERSION, "row_format": 2, "source": args.source,
            "source_license": "CC0 for the official Lichess source; verify custom sources",
            "score_perspective": "white in source; side-to-move after conversion",
            "sampling": "filtered source prefix, NOT a uniform sample of the entire database",
            "split": "blake2b of canonical features; duplicates and color mirrors share split",
            "rows": "side-to-move features only; opponent view derived by features.other_perspective; not deduplicated",
            "min_depth": args.min_depth, "val_fraction": args.val_fraction,
            "scanned": scanned, "skipped": skipped,
            "counts": sizes, "shards": shards, "complete": complete,
            "stop_reason": reason if complete or reason == "interrupted" else "error",
            "seconds": time.monotonic() - start,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--out", default="data/lichess-10m")
    p.add_argument("--positions", type=int, default=10_000_000)
    p.add_argument("--min-depth", type=int, default=16)
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--shard-size", type=int, default=250_000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args()
    if args.positions < 1 or args.shard_size < 1 or not 0 < args.val_fraction < 1:
        p.error("positions/shard-size must be positive and val-fraction must be between 0 and 1")
    prepare(args)


if __name__ == "__main__":
    main()
