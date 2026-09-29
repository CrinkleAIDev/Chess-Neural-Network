import json
import subprocess
import sys
import time

import chess
import chess.engine
import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from scratch_chess.features import PAD, encode, position_key, target_from_white
from scratch_chess.model import Evaluator, NumpyEvaluator, export_model
from scratch_chess.prepare import parse_record
from scratch_chess.search import MATE, Searcher


@pytest.fixture(scope="session", autouse=True)
def threads():
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        yield


def test_score_perspective_and_mate():
    assert target_from_white(cp=400, turn=True) > 0
    assert target_from_white(cp=400, turn=False) < 0
    assert target_from_white(mate=-3, turn=False) == 1
    assert target_from_white(mate=3, turn=False) == -1
    with pytest.raises(ValueError):
        target_from_white(mate=0)


def test_color_mirror_same_features_and_split():
    board = chess.Board()
    for move in ["e2e4", "c7c5", "g1f3", "d7d6", "f1b5"]:
        board.push_uci(move)
    a, x = encode(board)
    b, y = encode(board.mirror())
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(x, y)
    assert position_key(a, x) == position_key(b, y)
    assert a.min() >= 0 and a.max() <= PAD


def test_other_perspective_is_derived_exactly():
    import random
    from scratch_chess.features import other_perspective
    rng = random.Random(3)
    for _ in range(300):
        board = chess.Board()
        for _ in range(rng.randint(0, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        ids, _ = encode(board)
        derived = other_perspective(ids[0])
        np.testing.assert_array_equal(np.sort(derived), ids[1])
        tensor = other_perspective(torch.from_numpy(ids[0]).unsqueeze(0))
        np.testing.assert_array_equal(np.sort(tensor.numpy()[0]), ids[1])


def test_castling_and_en_passant_encoded():
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    _, extras = encode(board)
    assert extras[4+3] == 1
    _, start = encode(chess.Board())
    assert start[:4].tolist() == [1, 1, 1, 1]


def test_labels_choose_deepest_and_normalize_black():
    obj = {"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -", "evals": [
        {"depth": 12, "pvs": [{"cp": -200}]},
        {"depth": 22, "pvs": [{"cp": 100}, {"cp": 500}]},
    ]}
    row = parse_record(obj)
    assert row[2] == pytest.approx(-np.tanh(0.25))
    assert parse_record(obj, min_depth=30) is None


def test_export_matches_torch_and_padding_is_zero(tmp_path):
    torch.manual_seed(17)
    net = Evaluator(width=32, hidden=16)
    path = tmp_path / "net.npz"
    export_model(net, path)
    cpu = NumpyEvaluator(path)
    rng = np.random.default_rng(5)
    board = chess.Board()
    for i in range(30):
        ids, extras = encode(board)
        with torch.no_grad():
            expected = net(torch.from_numpy(ids[None]), torch.from_numpy(extras[None])).item()
        assert cpu.value(board) == pytest.approx(expected, abs=2e-6)
        if board.is_game_over():
            board.reset()
        board.push(list(board.legal_moves)[rng.integers(board.legal_moves.count())])
    assert torch.count_nonzero(net.embedding.weight[PAD]).item() == 0


def test_optimizer_learns_without_pretrained_weights():
    torch.manual_seed(0)
    model = Evaluator(width=32, hidden=16)
    board = chess.Board()
    ids, extras = encode(board)
    ids = torch.from_numpy(np.repeat(ids[None], 8, axis=0))
    extras = torch.from_numpy(np.repeat(extras[None], 8, axis=0))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = (model(ids, extras) - 0.6).square().mean().item()
    for _ in range(50):
        optimizer.zero_grad()
        loss = (model(ids, extras) - 0.6).square().mean()
        loss.backward()
        optimizer.step()
    assert loss.item() < before * 0.05
    assert torch.count_nonzero(model.embedding.weight[PAD]).item() == 0


@pytest.mark.parametrize("fen", [
    "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
    "8/8/8/8/8/6k1/5q2/7K b - - 0 1",
])
def test_search_finds_mate_with_either_color(fen):
    board = chess.Board(fen)
    original = board.fen()
    result = Searcher(lambda b: 0).search(board, seconds=2, max_depth=2)
    assert board.fen() == original
    assert result.score >= MATE-2
    board.push(result.move)
    assert board.is_checkmate()


def test_terminal_draw_and_checkmate():
    engine = Searcher(lambda b: 999)
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    result = engine.search(stalemate)
    assert result.score == 0 and result.move is None
    mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1")
    assert engine.search(mate).score == -MATE
    board = chess.Board()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        board.push_uci(move)
    assert engine.search(board).score == 0


def test_quiescence_searches_quiet_check_evasions():
    board = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
    assert board.is_check()
    visited = []
    engine = Searcher(lambda b: visited.append(b.fen()) or 0)
    engine.nodes = 0
    engine.max_nodes = 10000
    engine.deadline = time.monotonic()+5
    engine.quiescence(board, -32000, 32000, 0)
    assert visited
    assert all(not chess.Board(fen).is_check() for fen in visited)


def test_timeout_and_node_budget_preserve_board():
    board = chess.Board()
    board.push_uci("e2e4")
    before = board.fen(), list(board.move_stack)
    result = Searcher(lambda b: 0).search(board, seconds=10, max_nodes=10)
    assert result.move in board.legal_moves
    assert result.nodes <= 11
    assert (board.fen(), board.move_stack) == before


def test_uci_handshake_play_and_stop(tmp_path):
    path = tmp_path / "net.npz"
    export_model(Evaluator(width=32, hidden=16), path)
    command = [sys.executable, "-m", "scratch_chess.uci", "--model", str(path)]
    with chess.engine.SimpleEngine.popen_uci(command, timeout=20) as engine:
        board = chess.Board()
        played = engine.play(board, chess.engine.Limit(time=0.05))
        assert played.move in board.legal_moves
        with engine.analysis(board) as analysis:
            time.sleep(0.05)
            analysis.stop()
            result = analysis.wait()
            assert result.move in board.legal_moves


def test_prepare_end_to_end_and_split_disjoint(tmp_path):
    from argparse import Namespace
    from scratch_chess.prepare import prepare
    from scratch_chess.train import load_data
    rng = np.random.default_rng(12)
    board = chess.Board()
    source = tmp_path / "sample.jsonl"
    rows = []
    for _ in range(350):
        if board.is_game_over():
            board.reset()
        rows.append({"fen": board.fen(), "evals": [{"depth": 20, "pvs": [{"cp": 50}]}]})
        board.push(list(board.legal_moves)[rng.integers(board.legal_moves.count())])
    rows.extend(rows[:40])
    source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    root = tmp_path / "prepared"
    prepare(Namespace(source=str(source), out=str(root), shard_size=20, min_depth=16, val_fraction=0.2, positions=1000))
    from scratch_chess.features import COMPACT_DTYPE, other_perspective
    train, val = load_data(root, "train"), load_data(root, "val")
    assert train.dtype == val.dtype == COMPACT_DTYPE

    def keys(rows):
        full = [np.stack((p, np.sort(other_perspective(p.astype(np.int64))))) for p in rows["pieces"]]
        return [position_key(ids, extras) for ids, extras in zip(full, rows["extras"])]

    train_keys, val_keys = keys(train), keys(val)
    # Duplicated source lines are kept, but always in the same split.
    assert set(train_keys).isdisjoint(set(val_keys))
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(train) + len(val) == manifest["scanned"] - manifest["skipped"]
    assert len(set(train_keys) | set(val_keys)) < len(train) + len(val)
