"""Rating-over-training-time curve: measure each checkpoint against Stockfish anchors.

For every checkpoint it probes Stockfish strength settings until the score is neither
a whitewash nor a blank, then plays a longer match there and converts the score into
an Elo estimate on Stockfish's own UCI_Elo scale.
"""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
FASTCHESS = ROOT / "tools/fastchess/fastchess-windows-x86-64/fastchess.exe"
STOCKFISH = ROOT / "tools/stockfish/sf_19/stockfish/stockfish-windows-x86-64-universal.exe"
ENGINE = ROOT / "native/scratchchess.exe"
BOOK = ROOT / "tools/books/8moves_v3.pgn"
LEVELS = [1320, 1500, 1700, 1900, 2100, 2300, 2500, 2700, 2900, 3100]
RESULT = re.compile(r"Games: (\d+), Wins: (\d+), Losses: (\d+), Draws: (\d+)")


def elo_diff(score):
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


def margin(score, games):
    """Rough 95% interval on the Elo difference, from the binomial error on the score."""
    score = min(max(score, 1e-6), 1 - 1e-6)
    sigma = math.sqrt(max(score * (1 - score), 1e-6) / games)
    low, high = max(1e-6, score - 2 * sigma), min(1 - 1e-6, score + 2 * sigma)
    return (elo_diff(high) - elo_diff(low)) / 2


def play(model, opponent, games, tc, concurrency, out_dir):
    """One match: our network vs a Stockfish level, or vs another of our networks."""
    engines = ["-engine", f"cmd={ENGINE}", f"args=--model {model}", "name=ours"]
    if isinstance(opponent, int):
        engines += ["-engine", f"cmd={STOCKFISH}", f"name=sf{opponent}", "option.UCI_LimitStrength=true",
                    f"option.UCI_Elo={opponent}", "option.Threads=1", "option.Hash=64"]
    else:
        engines += ["-engine", f"cmd={ENGINE}", f"args=--model {opponent}", "name=anchor"]
    command = [str(FASTCHESS), *engines, "-each", f"tc={tc}", "-rounds", str(games // 2), "-games", "2",
               "-repeat", "-concurrency", str(concurrency), "-openings", f"file={BOOK}", "format=pgn",
               "order=random", "-pgnout", f"file={out_dir}/games.pgn", "-recover"]
    text = subprocess.run(command, capture_output=True, text=True, cwd=ROOT).stdout
    found = RESULT.findall(text)
    if not found:
        raise RuntimeError(f"Could not read a result for {model} vs {opponent}")
    played, wins, losses, draws = (int(x) for x in found[-1])
    return (wins + draws / 2) / played, played


def measure(model, start, args, out_dir):
    """Probe for a level where the score is between 20% and 80%, then play a longer match."""
    index = min(range(len(LEVELS)), key=lambda i: abs(LEVELS[i] - start))
    seen = {}
    while True:
        level = LEVELS[index]
        if level not in seen:
            seen[level], _ = play(model, level, args.probe, args.tc, args.concurrency, out_dir)
            print(f"    probe vs {level}: {seen[level]*100:.0f}%", flush=True)
        score = seen[level]
        if score > 0.8 and index < len(LEVELS) - 1:
            index += 1
        elif score < 0.2 and index > 0:
            index -= 1
        else:
            break
    score, played = play(model, level, args.games, args.tc, args.concurrency, out_dir)
    combined = (score * args.games + seen[level] * args.probe) / (args.games + args.probe)
    games = played + args.probe
    if combined < 0.02 and level == LEVELS[0]:
        return {"level": level, "score": combined, "games": games, "elo": None,
                "note": "below Stockfish's weakest setting (1320)"}
    return {"level": level, "score": combined, "games": games,
            "elo": round(level + elo_diff(combined)), "margin": round(margin(combined, games))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/main")
    p.add_argument("--models", nargs="*", help="Checkpoint stems; default: a spread across the run")
    p.add_argument("--tc", default="5+0.05")
    p.add_argument("--probe", type=int, default=20)
    p.add_argument("--games", type=int, default=80)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--out", default="runs/curve")
    args = p.parse_args()
    run = ROOT / args.run
    default = ["initial", "checkpoint-step0000100", "checkpoint-step0001000", "checkpoint-step0010000",
               "checkpoint-step0100000", "checkpoint-003600s", "checkpoint-007200s", "checkpoint-014400s",
               "checkpoint-021600s", "checkpoint-028800s", "best"]
    names = args.models or [n for n in default if (run / f"{n}.scn").exists()]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    if (run / "snapshots.jsonl").exists():
        for line in (run / "snapshots.jsonl").read_text().splitlines():
            row = json.loads(line)
            snapshots[Path(row["file"]).stem] = row
    results = []
    start = 1320
    for name in names:
        model = run / f"{name}.scn"
        print(f"  {name}", flush=True)
        began = time.monotonic()
        row = measure(model.relative_to(ROOT), start, args, out)
        row.update({"name": name, "seconds_trained": snapshots.get(name, {}).get("seconds"),
                    "steps": snapshots.get(name, {}).get("step"), "minutes_measuring": round((time.monotonic()-began)/60, 1)})
        if row.get("elo"):
            start = row["elo"]
        results.append(row)
        print(f"    -> {row.get('elo') or row['note']} ({row['games']} games vs sf{row['level']})", flush=True)
        (out / "curve.json").write_text(json.dumps({"tc": args.tc, "run": args.run, "results": results}, indent=2), encoding="utf-8")
    print(f"\n{'checkpoint':<26}{'trained':>10}{'vs':>7}{'score':>8}{'Elo':>8}")
    for row in results:
        hours = f"{row['seconds_trained']/3600:.2f} h" if row.get("seconds_trained") else ("0" if row["name"] == "initial" else "10 h")
        elo = f"{row['elo']} ±{row['margin']}" if row.get("elo") else "<1320"
        print(f"{row['name']:<26}{hours:>10}{row['level']:>7}{row['score']*100:>7.0f}%{elo:>12}")


if __name__ == "__main__":
    main()
