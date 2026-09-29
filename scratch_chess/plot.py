"""Export an honest training-progress chart for the reel (not an Elo chart)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/main")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(args.run)
    rows = []
    for line in (root / "metrics.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not rows:
        raise ValueError("No validation records yet")
    times = [r["seconds"]/60 for r in rows]
    loss = [r["val_mse"] for r in rows]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12}):
        fig, ax = plt.subplots(figsize=(9, 6), facecolor="#10171e")
        ax.set_facecolor("#10171e")
        ax.plot(times, loss, color="#8bdac7", linewidth=3, marker="o", markersize=4)
        ax.set_xlabel("Training time (minutes)", color="white")
        ax.set_ylabel("Held-out evaluation error (MSE)", color="white")
        ax.set_title("My chess network learning from scratch", color="white", loc="left", pad=20)
        ax.tick_params(colors="#d5e0e8")
        for spine in ax.spines.values():
            spine.set_color("#446073")
        ax.grid(alpha=0.15, color="white")
        ax.set_ylim(bottom=0)
        fig.text(0.12, 0.025, "Lower error means closer to the teacher. Playing strength requires separate matches.", color="#b1c0cd", fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        out = Path(args.out) if args.out else root / "training-curve.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, facecolor=fig.get_facecolor())
        plt.close(fig)
    print(out.resolve())


if __name__ == "__main__":
    main()
