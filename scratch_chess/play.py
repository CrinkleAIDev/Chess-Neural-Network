"""Play against your exported network in a terminal; saves a PGN."""
import argparse
from pathlib import Path

import chess
import chess.engine
import chess.pgn
from threadpoolctl import threadpool_limits

from . import native
from .model import NumpyEvaluator
from .search import Searcher


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="runs/main/best.npz")
    p.add_argument("--seconds", type=float, default=1)
    p.add_argument("--black", action="store_true", help="Play as black")
    p.add_argument("--pgn", default="runs/human-game.pgn")
    p.add_argument("--python", action="store_true", help="Use the slow Python search instead of the native engine")
    args = p.parse_args()
    threadpool_limits(limits=1)
    native_engine = None if args.python else chess.engine.SimpleEngine.popen_uci(native.command(args.model))
    engine = Searcher(NumpyEvaluator(args.model)) if args.python else None
    board = chess.Board()
    human = chess.BLACK if args.black else chess.WHITE
    print("Enter SAN (Nf3), UCI (g1f3), or quit.")
    try:
        while not board.is_game_over(claim_draw=True):
            print("\n" + str(board))
            if board.turn == human:
                text = input("Your move: ").strip()
                if text.lower() in ("quit", "exit"):
                    break
                try:
                    move = board.parse_san(text)
                except ValueError:
                    try:
                        move = board.parse_uci(text)
                    except ValueError:
                        print("That move isn't legal in this position.")
                        continue
            elif native_engine:
                result = native_engine.play(board, chess.engine.Limit(time=args.seconds), info=chess.engine.INFO_BASIC)
                move = result.move
                print(f"Bot: {board.san(move)} | depth {result.info.get('depth')} | {result.info.get('nodes')} nodes")
            else:
                result = engine.search(board, args.seconds)
                move = result.move
                print(f"Bot: {board.san(move)} | depth {result.depth} | {result.nodes} nodes")
            board.push(move)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if native_engine:
            native_engine.quit()
    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "Human" if human else "ScratchChess"
    game.headers["Black"] = "ScratchChess" if human else "Human"
    game.headers["Result"] = board.result(claim_draw=True)
    path = Path(args.pgn)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(str(game) + "\n\n")
    print(f"Result: {game.headers['Result']}; saved to {path}")


if __name__ == "__main__":
    main()
