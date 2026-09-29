import json
from pathlib import Path
import subprocess
import sys

import chess
import numpy as np
import torch
import pytest

from scratch_chess.features import DTYPE, VERSION, encode, position_key
from scratch_chess.model import load_checkpoint
from scratch_chess.train import run_lock


def test_run_lock_rejects_second_trainer_and_releases(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already owns"):
            with run_lock(tmp_path):
                pass
    with run_lock(tmp_path):
        pass


def test_training_save_resume_and_safe_stop(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rows = np.empty(32, dtype=DTYPE)
    board = chess.Board()
    for i in range(32):
        ids, extras = encode(board)
        rows[i] = ids, extras, 0.25, position_key(ids, extras)
        board.push(list(board.legal_moves)[i % board.legal_moves.count()])
    np.save(data / "train.npy", rows[:24])
    np.save(data / "val.npy", rows[24:])
    manifest = {"feature_version": VERSION, "shards": {
        "train": [{"file": "train.npy", "count": 24}], "val": [{"file": "val.npy", "count": 8}]}}
    (data / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "run"
    base = [sys.executable, "-m", "scratch_chess.train", "--device", "cpu", "--data", str(data),
        "--out", str(out), "--hours", "0.1", "--batch-size", "8", "--width", "16", "--hidden", "8", "--cpu-threads", "1"]
    subprocess.run(base + ["--max-steps", "3"], check=True, capture_output=True, timeout=40)
    _, first = load_checkpoint(out / "last.pt")
    initial_steps = first["step"]
    assert initial_steps == 3
    assert (out / "initial.npz").exists() and (out / "best.npz").exists()
    subprocess.run(base + ["--resume", str(out / "last.pt"), "--max-steps", "6"], check=True, capture_output=True, timeout=40)
    _, second = load_checkpoint(out / "last.pt")
    assert second["step"] == 6
    assert second["samples"] == 48
    assert second["train_seconds"] >= first["train_seconds"]
    assert not torch.equal(first["model"]["out.weight"], second["model"]["out.weight"])
    (out / "STOP").touch()
    subprocess.run(base + ["--resume", str(out / "last.pt")], check=True, capture_output=True, timeout=40)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["stop_reason"] == "STOP_file"
    assert summary["steps"] == 6
