"""Ask Stockfish to label positions from OUR games for a later training dataset."""
import argparse
import json
from pathlib import Path

import chess
import chess.engine
import chess.pgn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pgn", required=True)
    p.add_argument("--stockfish", required=True)
    p.add_argument("--out", default="data/own-game-labels.jsonl")
    p.add_argument("--nodes", type=int, default=100000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    args = p.parse_args()
    if min(args.nodes, args.threads, args.stride) < 1:
        p.error("nodes, threads and stride must be positive")
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("Output already exists; choose a new file")
    seen = set()
    count = 0
    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as engine, open(args.pgn, encoding="utf-8") as source, destination.open("w", encoding="utf-8") as output:
        engine.configure({"Threads": args.threads, "Hash": 256, "UCI_LimitStrength": False})
        while (game := chess.pgn.read_game(source)) is not None:
            for index, node in enumerate(game.mainline()):
                board = node.board()
                if index % args.stride or board.is_game_over() or board.is_check():
                    continue
                fen = " ".join(board.fen().split()[:4])
                if fen in seen:
                    continue
                seen.add(fen)
                info = engine.analyse(board, chess.engine.Limit(nodes=args.nodes))
                score = info["score"].white()
                pv = {"line": " ".join(m.uci() for m in info.get("pv", []))}
                if score.is_mate():
                    pv["mate"] = score.mate()
                else:
                    pv["cp"] = score.score()
                row = {"fen": fen, "evals": [{"depth": info.get("depth", 0), "knodes": info.get("nodes", 0)//1000, "pvs": [pv]}]}
                output.write(json.dumps(row) + "\n")
                output.flush()
                count += 1
                if count % 50 == 0:
                    print(f"Labeled {count} positions", flush=True)
    print(f"Saved {count} positions to {destination}. Convert with scratch_chess.prepare --source {destination} --min-depth 0 --out data/own-games")


if __name__ == "__main__":
    main()

