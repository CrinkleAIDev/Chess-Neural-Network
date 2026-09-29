"""Native engine correctness: perft, incremental accumulator and Python eval parity."""
import os
from pathlib import Path
import random
import subprocess

import chess
import numpy as np
import pytest

from scratch_chess.model import NumpyEvaluator
from scratch_chess import native
from scratch_chess.native import export

ROOT = Path(__file__).resolve().parent.parent
# Overridable so candidate builds and other network widths can be checked.
ENGINE = Path(os.environ.get("SCRATCH_ENGINE", native.ENGINE))
MODEL = Path(os.environ.get("SCRATCH_MODEL", ROOT / "runs" / "ready-check" / "best.npz"))

pytestmark = pytest.mark.skipif(not ENGINE.exists(), reason="native engine not built")

# Standard perft suite (chessprogramming.org): castling, en passant, promotions, pins.
PERFT = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3, 89890),
]


def session(commands, model=None, timeout=120):
    args = [str(ENGINE)] + (["--model", str(model)] if model else [])
    result = subprocess.run(args, input="\n".join(commands + ["quit"]) + "\n",
                            capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def bestmove(model, position, go):
    """Wait for bestmove before quitting; quit would otherwise abort the search."""
    proc = subprocess.Popen([str(ENGINE), "--model", str(model)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True)
    try:
        proc.stdin.write(f"{position}\n{go}\n")
        proc.stdin.flush()
        for line in proc.stdout:
            if line.startswith("bestmove"):
                return line.split()[1]
    finally:
        proc.stdin.write("quit\n")
        proc.stdin.close()
        proc.wait(timeout=30)


@pytest.mark.parametrize("fen,depth,expected", PERFT)
def test_perft(fen, depth, expected):
    out = session([f"position fen {fen}", f"perft {depth}"])
    assert out[-1] == f"nodes {expected}"


def random_positions(count, seed=7):
    rng = random.Random(seed)
    boards = []
    while len(boards) < count:
        board = chess.Board()
        for _ in range(rng.randint(0, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over():
            boards.append(board)
    return boards


@pytest.fixture(scope="module")
def scn(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip("no trained model available")
    return export(MODEL, tmp_path_factory.mktemp("net") / "net.scn")


def test_legal_moves_match_python_chess():
    boards = random_positions(150)
    commands = []
    for b in boards:
        commands += [f"position fen {b.fen()}", "legal"]
    out = session(commands)
    assert len(out) == len(boards)
    for b, line in zip(boards, out):
        assert set(line.split()) == {m.uci() for m in b.legal_moves}, b.fen()


def test_incremental_accumulator_and_eval_parity(scn):
    boards = random_positions(150, seed=11)
    commands = []
    for b in boards:
        commands += [f"position fen {b.fen()}", "verifyacc", "eval"]
    out = session(commands, scn)
    python = NumpyEvaluator(MODEL)
    errors = [float(l.split()[1]) for l in out if l.startswith("error")]
    values = [float(l.split()[1]) for l in out if l.startswith("value")]
    assert len(errors) == len(values) == len(boards)
    assert max(errors) < 1e-3
    expected = np.array([python.value(b) for b in boards])
    assert np.max(np.abs(np.array(values) - expected)) < 1e-4


def test_searches_at_root_even_when_draw_is_claimable(scn):
    # Queen attacked by rook; halfmove clock 100 must not short-circuit the root.
    assert bestmove(scn, "position fen 3r2k1/5ppp/8/8/3Q4/8/5PPP/6K1 w - - 100 150", "go depth 4") == "d4d8"


def test_threefold_is_avoided_when_winning(scn):
    # White is a queen up; shuffling back to a twice-seen position would throw the win away.
    moves = "g1h1 g8h8 h1g1 h8g8 g1h1 g8h8 h1g1 h8g8"
    best = bestmove(scn, f"position fen 6k1/5ppp/8/8/3Q4/8/5PPP/6K1 w - - 0 1 moves {moves}", "go depth 6")
    assert best != "g1h1"
