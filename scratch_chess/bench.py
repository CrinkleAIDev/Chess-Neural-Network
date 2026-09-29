"""Fixed-position native search benchmark: depth reached and nodes/second."""
import argparse
import subprocess

from .native import command

POSITIONS = [
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r2q1rk1/pp2bppp/2n1bn2/3p4/3P4/2NBBN2/PP3PPP/R2Q1RK1 w - - 5 11",
    "r1b2rk1/2q1bppp/p2p1n2/np2p3/3PP3/5N1P/PPBN1PP1/R1BQR1K1 w - - 1 13",
    "8/5pk1/6p1/3R4/5P2/r5P1/6K1/8 w - - 0 45",
    "2r3k1/pp3ppp/4p3/3pP3/3P1P2/P1R3P1/1P5P/6K1 b - - 0 28",
]


def run(model, movetime, pruning=True, engine=None):
    cmd = command(model)
    if engine:
        cmd[0] = engine
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    results = []
    try:
        proc.stdin.write(f"setoption name Pruning value {'true' if pruning else 'false'}\n")
        for fen in POSITIONS:
            proc.stdin.write(f"ucinewgame\nposition fen {fen}\ngo movetime {int(movetime*1000)}\n")
            proc.stdin.flush()
            last = None
            for line in proc.stdout:
                if line.startswith("info"):
                    last = line.split()
                elif line.startswith("bestmove"):
                    break
            field = lambda name: int(last[last.index(name) + 1]) if last and name in last else 0
            results.append((field("depth"), field("nodes"), field("nps")))
    finally:
        proc.stdin.write("quit\n")
        proc.stdin.close()
        proc.wait(timeout=30)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--movetime", type=float, default=1.0)
    p.add_argument("--no-pruning", action="store_true")
    p.add_argument("--engine", help="Alternative engine executable, e.g. a candidate build")
    args = p.parse_args()
    rows = run(args.model, args.movetime, not args.no_pruning, args.engine)
    for fen, (depth, nodes, nps) in zip(POSITIONS, rows):
        print(f"depth {depth:2d}  nodes {nodes:>9,}  nps {nps:>9,}  {fen}")
    print(f"mean nps {sum(r[2] for r in rows)//len(rows):,}  mean depth {sum(r[0] for r in rows)/len(rows):.1f}")


if __name__ == "__main__":
    main()
