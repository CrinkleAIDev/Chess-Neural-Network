"""Original iterative-deepening negamax, alpha-beta/PVS and quiescence search."""
from dataclasses import dataclass
import threading
import time

import chess

MATE = 30000
INF = 32000
PIECE_ORDER = [0, 100, 320, 330, 500, 900, 20000]  # move ordering ONLY, never evaluation


class SearchStopped(Exception):
    pass


@dataclass
class SearchResult:
    move: chess.Move | None
    score: int
    depth: int
    nodes: int
    seconds: float


class Searcher:
    def __init__(self, evaluator, cache_size=100_000):
        self.evaluate = evaluator
        self.cache_size = cache_size
        self.evals = {}
        self.moves = {}
        self.killers = {}
        self.history = {}
        self.stop = threading.Event()

    def clear(self):
        self.evals.clear()
        self.moves.clear()
        self.killers.clear()
        self.history.clear()

    def static(self, board):
        key = board._transposition_key()
        value = self.evals.get(key)
        if value is None:
            value = self.evaluate(board)
            if len(self.evals) >= self.cache_size:
                self.evals.clear()
            self.evals[key] = value
        return value

    def tick(self):
        self.nodes += 1
        if self.stop.is_set() or self.nodes > self.max_nodes or time.monotonic() >= self.deadline:
            raise SearchStopped

    def ordered(self, board, moves, ply):
        preferred = self.moves.get(board._transposition_key())
        killer = self.killers.get(ply)
        def priority(move):
            if move == preferred:
                return 10_000_000
            promotion = PIECE_ORDER[move.promotion] if move.promotion else 0
            if board.is_capture(move):
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                attacker = board.piece_type_at(move.from_square)
                return 1_000_000 + 16*PIECE_ORDER[victim] - PIECE_ORDER[attacker] + promotion
            if move.promotion:
                return 900_000 + promotion
            if move == killer:
                return 800_000
            return self.history.get((board.turn, move.from_square, move.to_square), 0)
        return sorted(moves, key=priority, reverse=True)

    def terminal(self, board, moves, ply):
        if not moves:
            return -MATE + ply if board.is_check() else 0
        # Claimable draws are taken by this engine. Mate takes precedence.
        if board.is_insufficient_material() or board.halfmove_clock >= 100 or board.is_repetition(3):
            return 0
        return None

    def quiescence(self, board, alpha, beta, ply, qdepth=0):
        self.tick()
        legal = list(board.legal_moves)
        end = self.terminal(board, legal, ply)
        if end is not None:
            return end
        checked = board.is_check()
        if ply >= 64:
            return self.static(board)
        if not checked:
            stand = self.static(board)
            if stand >= beta:
                return stand
            alpha = max(alpha, stand)
            if qdepth >= 8:
                return alpha
            legal = [m for m in legal if board.is_capture(m) or m.promotion]
        # Never stand pat in check: search ALL legal evasions.
        for move in self.ordered(board, legal, ply):
            board.push(move)
            try:
                score = -self.quiescence(board, -beta, -alpha, ply+1, qdepth+1)
            finally:
                board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def negamax(self, board, depth, alpha, beta, ply):
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)
        self.tick()
        legal = list(board.legal_moves)
        end = self.terminal(board, legal, ply)
        if end is not None:
            return end
        if ply >= 64:
            return self.static(board)
        best_move = None
        best_score = -INF
        for index, move in enumerate(self.ordered(board, legal, ply)):
            quiet = not board.is_capture(move) and not move.promotion
            board.push(move)
            try:
                if index == 0:
                    score = -self.negamax(board, depth-1, -beta, -alpha, ply+1)
                else:
                    score = -self.negamax(board, depth-1, -alpha-1, -alpha, ply+1)
                    if alpha < score < beta:
                        score = -self.negamax(board, depth-1, -beta, -alpha, ply+1)
            finally:
                board.pop()
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                if quiet:
                    self.killers[ply] = move
                    key = (board.turn, move.from_square, move.to_square)
                    self.history[key] = min(700_000, self.history.get(key, 0) + depth*depth)
                break
        if len(self.moves) >= self.cache_size:
            self.moves.clear()
        # Store ordering hints ONLY. Reusing score bounds across different repetition
        # histories can incorrectly erase a draw; this first engine avoids that risk.
        self.moves[board._transposition_key()] = best_move
        return best_score

    def search(self, board, seconds=1.0, max_depth=64, max_nodes=None, callback=None, reset_stop=True):
        start = time.monotonic()
        if reset_stop:
            self.stop.clear()
        self.deadline = start + max(0.001, seconds)
        self.max_nodes = max_nodes if max_nodes is not None else float("inf")
        self.nodes = 0
        self.killers.clear()
        self.history.clear()
        legal = list(board.legal_moves)
        best = SearchResult(legal[0] if legal else None, 0, 0, 0, 0)
        end = self.terminal(board, legal, 0)
        if end is not None:
            best.score = end
            return best
        for depth in range(1, max_depth+1):
            try:
                score = self.negamax(board, depth, -INF, INF, 0)
            except SearchStopped:
                break
            move = self.moves.get(board._transposition_key(), best.move)
            best = SearchResult(move, score, depth, self.nodes, time.monotonic()-start)
            if callback:
                callback(best)
            if abs(score) >= MATE-64:
                break
        best.nodes = self.nodes
        best.seconds = time.monotonic()-start
        return best

