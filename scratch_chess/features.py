"""Shared training/inference encoding. Scores are always side-to-move relative."""
import hashlib
import math

import chess
import numpy as np

VERSION = 1
KING_BUCKETS = 16
FEATURES = KING_BUCKETS * 12 * 64
PAD = FEATURES
MAX_PIECES = 32
EXTRAS = 12  # four relative castling rights and eight en-passant files
SCALE = 400.0
DTYPE = np.dtype([
    ("pieces", "<u2", (2, MAX_PIECES)), ("extras", "u1", (EXTRAS,)),
    ("target", "<f4"), ("key", "<u8"),
])
# Row format 2: side-to-move perspective only (the other is derived exactly by
# `other_perspective`), no stored key. 80 bytes instead of 152, so the full
# ~400M-position Lichess export fits on disk.
COMPACT_DTYPE = np.dtype([
    ("pieces", "<u2", (MAX_PIECES,)), ("extras", "u1", (EXTRAS,)), ("target", "<f4"),
])
ENEMY_KING_PLANE = 11


def other_perspective(ids, pad=None):
    """Opponent-perspective feature ids from side-to-move ids (torch or numpy, [..., 32]).

    Both perspectives mirror squares vertically relative to each other (XOR 56), swap
    own/enemy planes, and bucket by the opponent's king, which is the stm view's
    enemy-king feature.
    """
    pad = FEATURES if pad is None else pad
    valid = ids != pad
    local = ids % 768
    plane, square = local // 64, local % 64
    king = (plane == ENEMY_KING_PLANE) & valid
    king_square = (square * king).sum(-1, keepdims=True) ^ 56  # exactly one enemy king per row
    bucket = (king_square // 8 // 2) * 4 + (king_square % 8) // 2
    other = bucket * 768 + ((plane + 6) % 12) * 64 + (square ^ 56)
    return other * valid + pad * (~valid)


def encode(board):
    pieces = list(board.piece_map().items())
    if len(pieces) > MAX_PIECES:
        raise ValueError("Only standard chess positions with at most 32 pieces are supported")
    ids = np.full((2, MAX_PIECES), PAD, dtype=np.int64)
    for row, side in enumerate((board.turn, not board.turn)):
        king = board.king(side)
        if king is None:
            raise ValueError("Position must contain both kings")
        flip = 0 if side == chess.WHITE else 56
        king ^= flip
        bucket = (chess.square_rank(king) // 2) * 4 + chess.square_file(king) // 2
        for col, (square, piece) in enumerate(pieces):
            plane = piece.piece_type - 1 + (0 if piece.color == side else 6)
            ids[row, col] = bucket * 768 + plane * 64 + (square ^ flip)
        ids[row].sort()  # canonical keys, independent of piece-map iteration order
    extra = np.zeros(EXTRAS, dtype=np.float32)
    for i, side in enumerate((board.turn, not board.turn)):
        extra[2*i] = board.has_kingside_castling_rights(side)
        extra[2*i+1] = board.has_queenside_castling_rights(side)
    if board.has_legal_en_passant():
        extra[4 + chess.square_file(board.ep_square)] = 1
    return ids, extra


def position_key(ids, extras):
    """Canonical color-mirrored positions go into the SAME split."""
    raw = ids.astype("<u2").tobytes() + extras.astype("u1").tobytes()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def target_from_white(cp=None, mate=None, turn=chess.WHITE):
    if mate is not None:
        if mate == 0:
            raise ValueError("Terminal mate-zero labels are excluded")
        value = 1.0 if mate > 0 else -1.0
    elif cp is not None and math.isfinite(float(cp)):
        value = math.tanh(float(cp) / SCALE)
    else:
        raise ValueError("Missing or invalid score")
    return value if turn == chess.WHITE else -value

