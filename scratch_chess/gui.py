"""Local browser UI: play the native engine with any saved network.

Run: .\\.venv\\Scripts\\python.exe -m scratch_chess.gui   (then open http://127.0.0.1:8765)
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import random
import re
import threading
import time
import webbrowser

import chess
import chess.engine
import chess.pgn
import chess.svg

from . import native

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).with_name("gui.html")


def piece_svgs():
    return {p: chess.svg.piece(chess.Piece.from_symbol(p)) for p in "PNBRQKpnbrqk"}


def list_models(runs):
    """Networks the engine can load, newest run first; 'best' of each run listed first."""
    models = []
    for run in sorted((d for d in runs.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True):
        files = sorted(run.glob("*.scn"))
        def order(path):
            name = path.stem
            if name == "best":
                return (0, 0)
            if name == "last":
                return (1, 0)
            match = re.search(r"(\d+)", name)
            return (2 if "step" in name else 3 if name != "initial" else 4, int(match.group(1)) if match else 0)
        for path in sorted(files, key=order):
            label = path.stem.replace("checkpoint-", "")
            if re.fullmatch(r"\d+s", label):
                label = f"{int(label[:-1]) / 3600:.1f} h"
            elif label.startswith("step"):
                label = f"step {int(label[4:]):,}"
            models.append({"id": str(path.relative_to(ROOT)), "run": run.name, "label": label})
    return models


class Game:
    def __init__(self):
        self.lock = threading.Lock()
        self.board = chess.Board()
        self.engine = None
        self.model = None
        self.human = chess.WHITE
        self.movetime = 1.0
        self.info = {}
        self.thinking = False

    def open_engine(self, model):
        # Restart every new game: training keeps rewriting best.scn, so this always loads the latest.
        self.close()
        path = (ROOT / model).resolve()
        if not path.is_file() or ROOT not in path.parents:
            raise ValueError("Unknown model")
        self.engine = chess.engine.SimpleEngine.popen_uci(native.command(path))
        self.engine.configure({"Hash": 256})
        self.model = model

    def close(self):
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass
        self.engine = None

    def state(self):
        board = self.board
        outcome = board.outcome(claim_draw=True)
        replay = chess.Board()
        sans = []
        for move in board.move_stack:
            sans.append(replay.san(move))
            replay.push(move)
        king = board.king(board.turn) if board.is_check() else None
        return {
            "fen": board.fen(), "turn": "w" if board.turn else "b",
            "human": "w" if self.human else "b",
            "legal": [m.uci() for m in board.legal_moves] if not outcome else [],
            "san": sans, "last": board.move_stack[-1].uci() if board.move_stack else None,
            "check": chess.square_name(king) if king is not None else None,
            "result": outcome.result() if outcome else None,
            "termination": outcome.termination.name.replace("_", " ").lower() if outcome else None,
            "info": self.info, "model": self.model, "movetime": self.movetime,
        }

    def pgn(self):
        game = chess.pgn.Game.from_board(self.board)
        bot = f"ScratchChess ({self.model})"
        game.headers["Event"] = "ScratchChess GUI game"
        game.headers["White"] = "Human" if self.human else bot
        game.headers["Black"] = bot if self.human else "Human"
        game.headers["Result"] = self.board.result(claim_draw=True)
        game.headers["MoveTimeSeconds"] = str(self.movetime)
        return str(game)


GAME = Game()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, body, status=200, kind="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def json(self, payload, status=200):
        self.send(json.dumps(payload), status)

    def do_GET(self):
        if self.path == "/":
            page = PAGE.read_text(encoding="utf-8").replace("__PIECES__", json.dumps(piece_svgs()))
            return self.send(page, kind="text/html; charset=utf-8")
        if self.path == "/api/models":
            return self.json(list_models(ROOT / "runs"))
        if self.path == "/api/state":
            with GAME.lock:
                return self.json(GAME.state())
        if self.path == "/api/pgn":
            with GAME.lock:
                return self.send(GAME.pgn(), kind="text/plain; charset=utf-8")
        self.send("Not found", 404, "text/plain")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        try:
            if self.path == "/api/new":
                with GAME.lock:
                    GAME.open_engine(body["model"])
                    color = body.get("color", "w")
                    if color == "random":
                        color = random.choice("wb")
                    GAME.human = chess.WHITE if color == "w" else chess.BLACK
                    GAME.movetime = min(30.0, max(0.05, float(body.get("movetime", 1.0))))
                    GAME.board = chess.Board()
                    GAME.info = {}
                    return self.json(GAME.state())
            if self.path == "/api/move":
                with GAME.lock:
                    move = chess.Move.from_uci(body["uci"])
                    if GAME.board.turn != GAME.human or move not in GAME.board.legal_moves:
                        return self.json({"error": "Illegal move"}, 400)
                    GAME.board.push(move)
                    return self.json(GAME.state())
            if self.path == "/api/engine":
                with GAME.lock:
                    if GAME.board.turn == GAME.human or GAME.board.outcome(claim_draw=True):
                        return self.json(GAME.state())
                    started = time.monotonic()
                    result = GAME.engine.play(GAME.board, chess.engine.Limit(time=GAME.movetime),
                                              info=chess.engine.INFO_BASIC | chess.engine.INFO_SCORE)
                    score = result.info.get("score")
                    white = score.white() if score else None
                    GAME.info = {
                        "depth": result.info.get("depth"), "nodes": result.info.get("nodes"),
                        "cp": white.score() if white and not white.is_mate() else None,
                        "mate": white.mate() if white and white.is_mate() else None,
                        "seconds": round(time.monotonic() - started, 2),
                    }
                    GAME.board.push(result.move)
                    return self.json(GAME.state())
            if self.path == "/api/undo":
                with GAME.lock:
                    # Take back to the human's previous turn.
                    if GAME.board.move_stack:
                        GAME.board.pop()
                    while GAME.board.move_stack and GAME.board.turn != GAME.human:
                        GAME.board.pop()
                    GAME.info = {}
                    return self.json(GAME.state())
        except (KeyError, ValueError) as error:
            return self.json({"error": str(error)}, 400)
        except chess.engine.EngineError as error:
            GAME.close()
            return self.json({"error": f"Engine failed: {error}"}, 500)
        self.send("Not found", 404, "text/plain")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"ScratchChess UI running at {url}  (Ctrl+C to quit)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        GAME.close()


if __name__ == "__main__":
    main()
