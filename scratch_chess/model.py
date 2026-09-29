"""Our own king-bucketed sparse evaluator; no downloaded network parameters."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import FEATURES, PAD, EXTRAS, VERSION, SCALE, encode


class Evaluator(nn.Module):
    def __init__(self, width=128, hidden=32, factorized=True):
        super().__init__()
        self.config = {"width": width, "hidden": hidden, "factorized": factorized}
        self.embedding = nn.EmbeddingBag(FEATURES + 1, width, mode="sum", padding_idx=PAD)
        # Shared piece-square features teach general patterns across king buckets.
        # Folded into the main table on export, so they cost nothing in search.
        self.shared = nn.EmbeddingBag(769, width, mode="sum", padding_idx=768) if factorized else None
        self.bias = nn.Parameter(torch.zeros(width))
        self.fc1 = nn.Linear(width * 2 + EXTRAS, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        nn.init.normal_(self.embedding.weight, std=0.025)
        if self.shared is not None:
            nn.init.normal_(self.shared.weight, std=0.025)
        with torch.no_grad():
            self.embedding.weight[PAD].zero_()
            if self.shared is not None:
                self.shared.weight[768].zero_()

    def forward(self, pieces, extras):
        batch = pieces.shape[0]
        x = self.embedding(pieces.reshape(batch * 2, 32)) + self.bias
        if self.shared is not None:
            shared_ids = torch.where(pieces == PAD, 768, pieces % 768)
            x = x + self.shared(shared_ids.reshape(batch * 2, 32))
        x = x.clamp(0, 1).reshape(batch, -1)
        x = torch.cat((x, extras), dim=1)
        x = self.fc1(x).clamp(0, 1)
        x = self.fc2(x).clamp(0, 1)
        return torch.tanh(self.out(x).squeeze(-1))


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload["feature_version"] != VERSION:
        raise ValueError("Unsupported feature version")
    config = dict(payload["config"])
    config.setdefault("factorized", False)  # earliest pilot checkpoint compatibility
    model = Evaluator(**config).to(device)
    model.load_state_dict(payload["model"])
    return model, payload


def export_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    if model.shared is not None:
        embedding = arrays["embedding.weight"].copy()
        embedding[:FEATURES] += np.tile(arrays.pop("shared.weight")[:768], (16, 1))
        arrays["embedding.weight"] = embedding
    arrays["feature_version"] = np.array(VERSION)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez(f, **arrays)
    temp.replace(path)


class NumpyEvaluator:
    """CPU inference without calling any external chess engine."""
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.weights = {k: data[k].copy() for k in data.files}
        if int(self.weights["feature_version"]) != VERSION:
            raise ValueError("Unsupported feature version")

    def value(self, board):
        ids, extras = encode(board)
        w = self.weights
        x = np.clip(w["embedding.weight"][ids].sum(axis=1) + w["bias"], 0, 1)
        x = np.concatenate((x.ravel(), extras))
        for name in ("fc1", "fc2"):
            x = np.clip(w[name + ".weight"] @ x + w[name + ".bias"], 0, 1)
        return float(np.tanh((w["out.weight"] @ x + w["out.bias"])[0]))

    def __call__(self, board):
        return int(SCALE * np.arctanh(np.clip(self.value(board), -0.99999, 0.99999)))
