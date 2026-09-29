"""Paired-opening matches and PGN recordings; no invented absolute Elo."""
import argparse
import json
import math
from pathlib import Path
import random
import sys

import chess
import chess.engine
import chess.pgn

from . import native

# Repeated with colors reversed. Variety beyond this small suite is recommended
# for final claims; --openings accepts a JSON list of UCI-move strings.
OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4",
    "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5",
    "g1f3 d7d5 g2g3 g8f6 f1g2 e7e6",
]


def engine_command(model, python=False):
    if python:
        return [sys.executable, "-m", "scratch_chess.uci", "--model", str(Path(model).resolve())]
    return native.command(model)


def score_summary(scores):
    n = len(scores)
    result = {"games": n, "wins": scores.count(1.0), "draws": scores.count(0.5), "losses": scores.count(0.0)}
    if n:
        p = sum(scores)/n
        result["score_fraction"] = p
        result["relative_elo_estimate"] = 400*math.log10(p/(1-p)) if 0 < p < 1 else None
        result["note"] = "Relative estimate vs this opponent at these settings, not FIDE/online Elo. No finite estimate at 0%/100%; small samples are noisy."
    return result


def run(args):
    if args.games < 2 or args.games % 2:
        raise ValueError("Use an even number of games >= 2 for color-reversed opening pairs")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "games.pgn").exists():
        raise ValueError("Match output already exists; choose a fresh directory")
    openings = json.loads(Path(args.openings).read_text()) if args.openings else OPENINGS.copy()
    if not openings:
        raise ValueError("Opening suite is empty")
    for opening in openings:
        board = chess.Board()
        for move in opening.split():
            board.push_uci(move)
    random.Random(args.seed).shuffle(openings)
    opponent = [args.stockfish] if args.stockfish else engine_command(args.opponent, args.python)
    scores = []
    records = []
    with chess.engine.SimpleEngine.popen_uci(engine_command(args.model, args.python), timeout=30) as ours, chess.engine.SimpleEngine.popen_uci(opponent, timeout=30) as other:
        if args.stockfish:
            options = {"Threads": 1, "Hash": 64}
            if args.stockfish_elo is not None:
                options.update({"UCI_LimitStrength": True, "UCI_Elo": args.stockfish_elo})
            other.configure(options)
        for number in range(args.games):
            opening = openings[(number//2) % len(openings)]
            board = chess.Board()
            for move in opening.split():
                board.push_uci(move)
            our_color = chess.WHITE if number % 2 == 0 else chess.BLACK
            game_id = object()
            truncated = False
            while not board.is_game_over(claim_draw=True):
                if board.ply() >= args.max_plies:
                    truncated = True
                    break
                engine = ours if board.turn == our_color else other
                played = engine.play(board, chess.engine.Limit(time=args.movetime), game=game_id)
                if played.move not in board.legal_moves:
                    raise RuntimeError("Engine returned an illegal move; match aborted, not silently scored")
                board.push(played.move)
            outcome = board.outcome(claim_draw=True)
            game = chess.pgn.Game.from_board(board)
            game.headers["Event"] = "ScratchChess checkpoint benchmark"
            game.headers["Round"] = str(number+1)
            our_name = Path(args.model).stem
            their_name = Path(args.stockfish or args.opponent).stem
            game.headers["White"] = our_name if our_color else their_name
            game.headers["Black"] = their_name if our_color else our_name
            game.headers["TimeControl"] = "-"
            game.headers["MoveTimeSeconds"] = str(args.movetime)
            game.headers["Result"] = outcome.result() if outcome else "*"
            game.headers["Termination"] = outcome.termination.name if outcome else "ply limit; unscored"
            with (out / "games.pgn").open("a", encoding="utf-8") as f:
                f.write(str(game) + "\n\n")
            if outcome:
                scores.append(0.5 if outcome.winner is None else float(outcome.winner == our_color))
            records.append({"round": number+1, "our_color": "white" if our_color else "black", "result": game.headers["Result"], "truncated": truncated})
            summary = score_summary(scores) | {"settings": vars(args), "records": records, "unscored": len(records)-len(scores)}
            (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Game {number+1}/{args.games}: {game.headers['Result']} | {score_summary(scores)}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--opponent", help="Another one of our exported .npz networks")
    group.add_argument("--stockfish", help="Path to Stockfish executable; only an opponent")
    p.add_argument("--stockfish-elo", type=int)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--movetime", type=float, default=0.2)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--openings")
    p.add_argument("--out", default="runs/match")
    p.add_argument("--python", action="store_true", help="Use the slow Python search instead of the native engine")
    args = p.parse_args()
    if args.movetime <= 0 or args.max_plies < 1:
        p.error("movetime and max-plies must be positive")
    run(args)


if __name__ == "__main__":
    main()

