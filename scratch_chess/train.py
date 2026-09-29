"""Time-bounded CUDA training, random initialization, atomic saves and resume."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np
import torch

from .dataset import ShardDataset, ShuffledStream
from .features import VERSION, other_perspective
from .model import Evaluator, export_model, load_checkpoint
from .native import export as export_native

# Log-spaced step snapshots show the fast early learning; hourly ones cover the long tail.
SNAPSHOT_STEPS = (100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000)


@contextmanager
def run_lock(directory):
    """OS-backed lock is released even after a crash; never unlink the lock file."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "training.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another trainer already owns {root}") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_data(root, split):
    """Whole split in RAM; used for small datasets and tests. Training streams instead."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["feature_version"] != VERSION:
        raise ValueError("Dataset feature version mismatch")
    paths = [root / entry["file"] for entry in manifest["shards"][split]]
    if not paths:
        raise ValueError(f"No {split} data. Prepare more positions first.")
    if sum(p.stat().st_size for p in paths) > 8_000_000_000:
        raise ValueError("This in-memory loader limits each split to 8GB; use a smaller dataset")
    arrays = [np.load(p, mmap_mode="r", allow_pickle=False) for p in paths]
    return np.concatenate(arrays)


def both_perspectives(ids):
    """[B, 32] side-to-move ids (compact rows) -> [B, 2, 32]; full rows pass through."""
    return ids if ids.dim() == 3 else torch.stack((ids, other_perspective(ids)), dim=1)


def batch_to_device(rows, device):
    # Structured fields are strided; make compact copies before conversion.
    pin = device.type == "cuda"
    ids = torch.from_numpy(rows["pieces"].astype(np.int64))
    extras = torch.from_numpy(rows["extras"].astype(np.float32))
    targets = torch.from_numpy(rows["target"].copy())
    if pin:
        ids, extras, targets = ids.pin_memory(), extras.pin_memory(), targets.pin_memory()
    return (both_perspectives(ids.to(device, non_blocking=pin)), extras.to(device, non_blocking=pin),
            targets.to(device, non_blocking=pin))


class Prefetcher:
    """Background thread: shard loading and shuffling never stall the GPU.

    Each batch carries the stream state AFTER it, so a checkpoint records exactly
    the consumed position, not what the thread has read ahead.
    """
    def __init__(self, stream, batch_size, depth=8):
        self.stream = stream
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=depth)
        self.error = None
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def work(self):
        try:
            while not self.closed.is_set():
                rows = self.stream.next(self.batch_size)
                state = self.stream.state()
                state["order"] = list(state["order"])
                item = (rows["pieces"].astype(np.int64), rows["extras"].astype(np.float32),
                        rows["target"].copy(), state)
                while not self.closed.is_set():
                    try:
                        self.queue.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        pass
        except BaseException as error:  # surfaced in the training thread
            self.error = error

    def next(self, device):
        while True:
            if self.error:
                raise self.error
            try:
                ids, extras, targets, state = self.queue.get(timeout=1)
                break
            except queue.Empty:
                pass
        pin = device.type == "cuda"
        tensors = [torch.from_numpy(a) for a in (ids, extras, targets)]
        if pin:
            tensors = [t.pin_memory() for t in tensors]
        ids, extras, targets = [t.to(device, non_blocking=pin) for t in tensors]
        return (both_perspectives(ids), extras, targets), state

    def close(self):
        self.closed.set()
        self.thread.join(timeout=10)


@torch.inference_mode()
def validate(model, data, batch_size, device):
    model.eval()
    squared = absolute = 0.0
    for offset in range(0, len(data), batch_size):
        ids, extras, targets = batch_to_device(data[offset:offset+batch_size], device)
        prediction = model(ids, extras).float()
        squared += (prediction - targets).square().sum().item()
        absolute += (prediction - targets).abs().sum().item()
    model.train()
    return {"val_mse": squared / len(data), "val_mae": absolute / len(data)}


def atomic_save(payload, path):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def export_both(model, path):
    """Python (.npz) and native engine (.scn) copies of the same weights."""
    export_model(model, path)
    export_native(path, Path(path).with_suffix(".scn"))


def cosine_lr(args, seconds):
    """Warmup, then cosine decay over the WHOLE time budget, so the budget is actually used."""
    budget = args.hours * 3600
    if seconds < args.warmup_seconds:
        return args.lr * max(0.05, seconds / args.warmup_seconds)
    progress = min(1.0, (seconds - args.warmup_seconds) / max(1.0, budget - args.warmup_seconds))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def train(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install CUDA-enabled PyTorch, or explicitly select --device cpu for a smoke test.")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output / "last.pt").exists():
        raise ValueError("Run already exists. Use --resume or choose another --out.")
    manifest_bytes = (Path(args.data) / "manifest.json").read_bytes()
    fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
    training = ShardDataset(args.data, "train")
    validation_set = ShardDataset(args.data, "val")
    # Fixed, seeded validation sample, drawn before any resume state is restored.
    val_indices = np.sort(rng.choice(len(validation_set), min(args.val_positions, len(validation_set)), replace=False))
    validation = validation_set[val_indices]
    del validation_set
    payload = None
    if args.resume:
        model, payload = load_checkpoint(args.resume, device)
        if payload["dataset_fingerprint"] != fingerprint:
            raise ValueError("Resume dataset differs from checkpoint; use its original dataset")
    else:
        model = Evaluator(args.width, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step = samples = 0
    previous_seconds = 0.0
    best = math.inf
    bad_checks = 0
    stream_state = None
    if payload:
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        step, samples = payload["step"], payload["samples"]
        previous_seconds, best = payload["train_seconds"], payload["best_val_mse"]
        bad_checks = payload.get("bad_checks", 0)
        stream_state = payload.get("stream_state")
        rng.bit_generator.state = payload["numpy_rng"]
        torch.set_rng_state(payload["torch_rng"].cpu())
        if device.type == "cuda" and payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(payload["cuda_rng"].cpu())
    if (output / "summary.json").exists():
        (output / "summary.json").unlink()
    config = vars(args) | {"parameters": sum(p.numel() for p in model.parameters()),
        "model_config": model.config, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "initialization": "random weights" if not payload else "resume own trained checkpoint",
        "dataset_fingerprint": fingerprint, "train_positions": len(training), "val_positions": len(validation)}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)
    if not payload:
        export_both(model, output / "initial.npz")
    stream = ShuffledStream(training, rng, stream_state)
    prefetch = Prefetcher(stream, args.batch_size)
    start = time.monotonic()
    last_eval = last_log = start
    next_snapshot = (int(previous_seconds // args.snapshot_seconds) + 1) * args.snapshot_seconds
    stop_reason = "time_budget"
    train_loss_sum = torch.zeros((), device=device)
    train_loss_steps = 0
    log_samples = samples

    def elapsed():
        return previous_seconds + time.monotonic() - start

    def save(name, metrics):
        checkpoint = {
            "feature_version": VERSION, "config": model.config,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step, "samples": samples, "train_seconds": elapsed(),
            "best_val_mse": best, "bad_checks": bad_checks, "metrics": metrics,
            "dataset_fingerprint": fingerprint, "numpy_rng": rng.bit_generator.state,
            "stream_state": stream_state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
        }
        atomic_save(checkpoint, output / f"{name}.pt")
        export_both(model, output / f"{name}.npz")

    def evaluate_and_save():
        nonlocal best, bad_checks, last_eval
        metrics = validate(model, validation, args.batch_size, device)
        metrics.update({"step": step, "samples": samples, "seconds": elapsed(),
            "epochs_equivalent": samples / len(training), "lr": optimizer.param_groups[0]["lr"]})
        improved = metrics["val_mse"] < best - 1e-6
        if improved:
            best = metrics["val_mse"]
            bad_checks = 0
        else:
            bad_checks += 1
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        if improved:
            save("best", metrics)
        save("last", metrics)
        last_eval = time.monotonic()
        print("validation " + json.dumps(metrics), flush=True)
        return metrics

    def snapshot(name, extra):
        export_both(model, output / f"{name}.npz")
        with (output / "snapshots.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"file": f"{name}.npz", "step": step, "seconds": elapsed(), "samples": samples} | extra) + "\n")

    metrics = evaluate_and_save()
    try:
        while elapsed() < args.hours * 3600:
            if (output / "STOP").exists():
                stop_reason = "STOP_file"
                break
            lr = cosine_lr(args, elapsed())
            for group in optimizer.param_groups:
                group["lr"] = lr
            (ids, extras, target), state = prefetch.next(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prediction = model(ids, extras)
                loss = (prediction.float() - target).square().mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            stream_state = state
            step += 1
            samples += len(target)
            train_loss_sum += loss.detach()  # no per-step GPU sync
            train_loss_steps += 1
            now = time.monotonic()
            if now - last_log >= args.log_seconds:
                train_mse = train_loss_sum.item() / train_loss_steps
                if not math.isfinite(train_mse):
                    raise RuntimeError("Non-finite loss; last checkpoint remains available")
                message = {"step": step, "seconds": round(elapsed(), 1), "lr": lr,
                    "train_mse": train_mse, "epochs": round(samples / len(training), 3),
                    "positions_per_second": round((samples-log_samples)/(now-last_log)),
                    "gpu_peak_gb": round(torch.cuda.max_memory_allocated()/1e9, 3) if device.type == "cuda" else 0}
                print(json.dumps(message), flush=True)
                with (output / "progress.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(message) + "\n")
                last_log, log_samples = now, samples
                train_loss_sum.zero_()
                train_loss_steps = 0
            if now - last_eval >= args.eval_seconds:
                metrics = evaluate_and_save()
                if args.patience and bad_checks >= args.patience:
                    stop_reason = "validation_plateau"
                    break
            if elapsed() >= next_snapshot:
                snapshot(f"checkpoint-{int(elapsed()):06d}s", {"kind": "time"})
                next_snapshot += args.snapshot_seconds
            if step in SNAPSHOT_STEPS:
                snapshot(f"checkpoint-step{step:07d}", {"kind": "step"})
            if args.max_steps and step >= args.max_steps:
                stop_reason = "step_limit"
                break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    finally:
        prefetch.close()
    metrics = evaluate_and_save()
    summary = {"stop_reason": stop_reason, "seconds": elapsed(), "steps": step,
        "samples": samples, "best_val_mse": best, "latest_val_mse": metrics["val_mse"],
        "note": "best means lowest held-out loss, NOT established highest playing strength"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/lichess-full")
    p.add_argument("--out", default="runs/main")
    p.add_argument("--hours", type=float, default=10, help="Total run budget INCLUDING elapsed time saved in resumed checkpoint; the LR schedule spans it")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--min-lr", type=float, default=0.00002)
    p.add_argument("--warmup-seconds", type=float, default=60)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--eval-seconds", type=float, default=300)
    p.add_argument("--snapshot-seconds", type=float, default=3600)
    p.add_argument("--log-seconds", type=float, default=30)
    p.add_argument("--val-positions", type=int, default=100000)
    p.add_argument("--patience", type=int, default=0, help="Stop after this many non-improving validations; 0 (default) uses the full budget")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--resume")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()
    if min(args.hours, args.width, args.hidden, args.batch_size, args.eval_seconds, args.snapshot_seconds, args.log_seconds, args.val_positions, args.cpu_threads, args.lr) <= 0:
        p.error("Time, dimensions, batch size, learning rate and thread settings must be positive")
    if args.width > 512 or args.hidden > 128:
        p.error("The native engine supports width <= 512 and hidden <= 128")
    with run_lock(args.out):
        train(args)


if __name__ == "__main__":
    main()
