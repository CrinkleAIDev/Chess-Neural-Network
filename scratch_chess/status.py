"""Print training progress without needing TensorBoard or a web service."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/main")
    args = p.parse_args()
    root = Path(args.run)
    config = json.loads((root / "config.json").read_text())
    rows = []
    for line in (root / "metrics.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a live process may be partway through appending the last row
    print(f"Run: {root}\nGPU: {config['gpu']}\nParameters: {config['parameters']:,}")
    print(f"Training positions: {config['train_positions']:,}")
    if rows:
        first, last = rows[0], rows[-1]
        best = min(rows, key=lambda r: r["val_mse"])
        print(f"Last validation: {last['seconds']/3600:.3f} training hours; step {last['step']:,}")
        print(f"Held-out MSE: initial {first['val_mse']:.6f}, latest {last['val_mse']:.6f}, best {best['val_mse']:.6f}")
        print("Validation loss measures teacher agreement; it does not establish Elo.")
    if (root / "summary.json").exists():
        summary = json.loads((root / "summary.json").read_text())
        print(f"Last completed session: {summary['stop_reason']}")
    print(f"Play model: {root / 'best.npz'}")


if __name__ == "__main__":
    main()

