"""UCI adapter: play the exported network in a GUI or a match runner."""
import argparse
import sys
import threading

import chess
from threadpoolctl import threadpool_limits

from .model import NumpyEvaluator
from .search import MATE, Searcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    # Tiny matrix products are MUCH faster without spawning many BLAS threads.
    threadpool_limits(limits=1)
    searcher = Searcher(NumpyEvaluator(args.model))
    board = chess.Board()
    worker = None
    output_lock = threading.Lock()

    def emit(text):
        with output_lock:
            print(text, flush=True)

    def halt():
        nonlocal worker
        if worker is not None:
            searcher.stop.set()
            worker.join()
            worker = None

    def report(result):
        if abs(result.score) >= MATE-64:
            distance = (MATE-abs(result.score)+1)//2
            score = f"mate {distance if result.score > 0 else -distance}"
        else:
            score = f"cp {result.score}"
        pv = result.move.uci() if result.move else "0000"
        emit(f"info depth {result.depth} score {score} nodes {result.nodes} nps {int(result.nodes/max(result.seconds,0.001))} time {int(result.seconds*1000)} pv {pv}")

    def think(position, options):
        try:
            seconds = options.pop("seconds")
            result = searcher.search(position, seconds=seconds, callback=report, reset_stop=False, **options)
            emit("bestmove " + (result.move.uci() if result.move else "0000"))
        except Exception as error:
            print(f"Search error: {error}", file=sys.stderr, flush=True)
            emit("bestmove 0000")

    try:
        for line in sys.stdin:
            words = line.strip().split()
            if not words:
                continue
            command = words[0]
            try:
                if command == "uci":
                    emit("id name ScratchChess 0.1")
                    emit("id author Your original network and search")
                    emit("uciok")
                elif command == "isready":
                    emit("readyok")
                elif command == "ucinewgame":
                    halt()
                    searcher.clear()
                    board = chess.Board()
                elif command == "position":
                    halt()
                    move_index = words.index("moves") if "moves" in words else len(words)
                    board = chess.Board() if words[1] == "startpos" else chess.Board(" ".join(words[2:move_index]))
                    for move in words[move_index+1:]:
                        board.push_uci(move)
                elif command == "go":
                    halt()
                    if "ponder" in words or "searchmoves" in words:
                        emit("info string ponder and searchmoves are not supported")
                        emit("bestmove 0000")
                        continue
                    def number(name, default=None):
                        return int(words[words.index(name)+1]) if name in words else default
                    seconds = float("inf") if any(k in words for k in ("infinite", "depth", "nodes")) else 5.0
                    side = "w" if board.turn else "b"
                    clock = number(side+"time")
                    if clock is not None:
                        seconds = min(max(0.001, (clock-50)/1000), max(0.005, clock/1000/max(1, number("movestogo", 30)) + number(side+"inc", 0)/1000*0.7))
                    if "movetime" in words:
                        seconds = max(0.001, number("movetime")/1000*0.95)
                    options = {"seconds": seconds, "max_depth": number("depth", 64), "max_nodes": number("nodes")}
                    searcher.stop.clear()
                    worker = threading.Thread(target=think, args=(board.copy(), options))
                    worker.start()
                elif command == "stop":
                    halt()
                elif command == "quit":
                    break
            except (ValueError, IndexError) as error:
                emit(f"info string invalid command: {error}")
    finally:
        halt()


if __name__ == "__main__":
    main()
