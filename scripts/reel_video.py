"""Render the Instagram reel (1080x1920, 30 fps) from the project's real data.

Uses: runs/main/progress.jsonl (training log), runs/curve*/curve.json (ratings),
runs/curve/games.pgn (untrained game), runs/reel/hour1.pgn (1-hour game),
runs/reel/music.wav (scripts/reel_music.py). Captions only, no narration.

Usage: python scripts/reel_video.py [--out runs/reel/reel.mp4] [--preview 12.5 ...]
"""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess

import chess
import chess.pgn
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS = 1080, 1920, 30
DURATION = 46.0
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()  # the ffmpeg on PATH here is an old build that rejects standard options

BG = (11, 13, 18)
PANEL = (21, 25, 34)
TEXT = (242, 242, 238)
MUTED = (138, 147, 166)
GREEN = (91, 228, 155)
GOLD = (245, 196, 81)
RED = (255, 107, 107)
LIGHT_SQ = (233, 228, 214)
DARK_SQ = (125, 155, 118)


# ---------- drawing helpers ----------

@lru_cache(maxsize=None)
def font(name, size):
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


HEAVY = lambda s: font("seguibl.ttf", s)
BOLD = lambda s: font("segoeuib.ttf", s)
REG = lambda s: font("segoeui.ttf", s)
MONO = lambda s: font("consola.ttf", s)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def appear(t, start, dur=0.35):
    return ease((t - start) / dur)


def text(frame, xy, s, fnt, fill=TEXT, alpha=1.0, anchor="mm", rise=0):
    if alpha <= 0.01:
        return
    x, y = xy
    y += (1 - alpha) * rise
    probe = ImageDraw.Draw(frame)
    box = probe.textbbox((0, 0), s, font=fnt, anchor=anchor)
    pad = 6
    layer = Image.new("RGBA", (box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((pad - box[0], pad - box[1]), s, font=fnt, fill=fill + (255,), anchor=anchor)
    if alpha < 1:
        layer.putalpha(layer.getchannel("A").point(lambda a: int(a * alpha)))
    frame.paste(layer, (int(x + box[0] - pad), int(y + box[1] - pad)), layer)


def rounded(frame, box, radius, fill, alpha=1.0):
    layer = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle((0, 0, layer.width - 1, layer.height - 1), radius, fill=fill + (int(255 * alpha),))
    frame.paste(layer, box[:2], layer)


def header(frame, t, start, label, title, title_color=TEXT, y=300):
    a = appear(t, start)
    text(frame, (W / 2, y), label, BOLD(34), GREEN, a, rise=20)
    text(frame, (W / 2, y + 72), title, HEAVY(70), title_color, appear(t, start + 0.12), rise=24)


# ---------- chess board ----------

GLYPH_FILLED = {"k": "\u265a", "q": "\u265b", "r": "\u265c", "b": "\u265d", "n": "\u265e", "p": "\u265f"}
GLYPH_OUTLINE = {"k": "\u2654", "q": "\u2655", "r": "\u2656", "b": "\u2657", "n": "\u2658", "p": "\u2659"}


@lru_cache(maxsize=None)
def sprite(symbol, size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    f = font("seguisym.ttf", int(size * 0.86))
    d = ImageDraw.Draw(img)
    glyph = GLYPH_FILLED[symbol.lower()]
    box = d.textbbox((0, 0), glyph, font=f)
    at = ((size - (box[2] - box[0])) / 2 - box[0], (size - (box[3] - box[1])) / 2 - box[1] - size * 0.01)
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text((at[0] + size * 0.02, at[1] + size * 0.035), glyph, font=f, fill=(0, 0, 0, 110))
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(size * 0.03)))
    if symbol.isupper():
        d.text(at, glyph, font=f, fill=(250, 250, 246, 255))
        d.text(at, GLYPH_OUTLINE[symbol.lower()], font=f, fill=(28, 28, 30, 255))
    else:
        d.text(at, glyph, font=f, fill=(30, 30, 34, 255), stroke_width=max(1, size // 60), stroke_fill=(30, 30, 34, 255))
    return img


class Replay:
    """Animate a PGN game on a board: per-ply timings, sliding pieces, move highlights."""

    def __init__(self, game, start, ply_times, size, white_bottom=True):
        self.moves = list(game.mainline_moves())
        self.boards = [chess.Board()]
        for move in self.moves:
            board = self.boards[-1].copy()
            board.push(move)
            self.boards.append(board)
        self.times = []
        at = start
        for k in range(len(self.moves)):
            self.times.append(at)
            at += ply_times(k)
        self.end = at
        self.size = size
        self.sq = size // 8
        self.white_bottom = white_bottom

    def xy(self, square):
        f, r = chess.square_file(square), chess.square_rank(square)
        col, row = (f, 7 - r) if self.white_bottom else (7 - f, r)
        return col * self.sq, row * self.sq

    @lru_cache(maxsize=512)
    def static(self, ply, hidden):
        board = self.boards[ply]
        img = Image.new("RGBA", (self.size, self.size))
        d = ImageDraw.Draw(img)
        last = self.moves[ply - 1] if ply else None
        for square in chess.SQUARES:
            x, y = self.xy(square)
            light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
            color = LIGHT_SQ if light else DARK_SQ
            if last and square in (last.from_square, last.to_square):
                color = tuple(int(c * 0.55 + g * 0.45) for c, g in zip(color, (246, 214, 90)))
            d.rectangle((x, y, x + self.sq, y + self.sq), fill=color)
        if board.is_checkmate():
            king = board.king(board.turn)
            x, y = self.xy(king)
            glow = Image.new("RGBA", (self.sq, self.sq), (0, 0, 0, 0))
            ImageDraw.Draw(glow).ellipse((4, 4, self.sq - 4, self.sq - 4), fill=RED + (200,))
            img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(self.sq * 0.12)), (x, y))
        for square, piece in board.piece_map().items():
            if square == hidden:
                continue
            img.alpha_composite(sprite(piece.symbol(), self.sq), self.xy(square))
        return img

    def banner_y(self):
        """Vertical centre (board-relative) for an end-of-game banner that keeps the mated king visible."""
        board = self.boards[-1]
        king = board.king(board.turn)
        _, y = self.xy(king)
        return int(self.size * (0.72 if y < self.size / 2 else 0.28))

    def render(self, t):
        done = sum(1 for s in self.times if s <= t)
        if done == 0:
            return self.static(0, -1)
        k = done - 1
        progress = ease((t - self.times[k]) / 0.12)
        if progress >= 1:
            return self.static(done, -1)
        move = self.moves[k]
        img = self.static(done, move.to_square).copy()
        (x0, y0), (x1, y1) = self.xy(move.from_square), self.xy(move.to_square)
        piece = self.boards[done].piece_at(move.to_square)
        img.alpha_composite(sprite(piece.symbol(), self.sq), (int(x0 + (x1 - x0) * progress), int(y0 + (y1 - y0) * progress)))
        return img


def paste_board(frame, img, x, y, alpha=1.0):
    shadow = Image.new("RGBA", (img.width + 80, img.height + 80), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((40, 50, img.width + 40, img.height + 50), 16, fill=(0, 0, 0, int(150 * alpha)))
    frame.paste(shadow.filter(ImageFilter.GaussianBlur(22)), (x - 40, y - 40), shadow.filter(ImageFilter.GaussianBlur(22)))
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, img.width - 1, img.height - 1), 14, fill=int(255 * alpha))
    frame.paste(img.convert("RGB"), (x, y), mask)


# ---------- charts (drawn at 2x for smooth lines) ----------

class LogChart:
    def __init__(self, box, xmin, xmax, ymin, ymax, xlog=True):
        self.box = box
        self.xmin, self.xmax, self.ymin, self.ymax, self.xlog = xmin, xmax, ymin, ymax, xlog

    def px(self, x, y, scale=1):
        l, t, r, b = self.box
        fx = (math.log10(x) - math.log10(self.xmin)) / (math.log10(self.xmax) - math.log10(self.xmin)) if self.xlog \
            else (x - self.xmin) / (self.xmax - self.xmin)
        fy = (y - self.ymin) / (self.ymax - self.ymin)
        return ((l + fx * (r - l)) * scale, (b - fy * (b - t)) * scale)


def smooth_polyline(frame, chart, points, color, width, alpha=1.0, glow=True):
    """Anti-aliased polyline: draw at 2x on a cropped layer, then downsample."""
    if len(points) < 2:
        return
    l, t, r, b = chart.box
    pad = 40
    region = (l - pad, t - pad, r + pad, b + pad)
    layer = Image.new("RGBA", ((region[2] - region[0]) * 2, (region[3] - region[1]) * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    pts = [((x - region[0]) * 2, (y - region[1]) * 2) for x, y in (chart.px(*p) for p in points)]
    if glow:
        g = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        ImageDraw.Draw(g).line(pts, fill=color + (int(90 * alpha),), width=width * 6, joint="curve")
        layer.alpha_composite(g.filter(ImageFilter.GaussianBlur(14)))
    d.line(pts, fill=color + (int(255 * alpha),), width=width * 2, joint="curve")
    layer = layer.resize((region[2] - region[0], region[3] - region[1]), Image.LANCZOS)
    frame.paste(layer, region[:2], layer)


def dashed(frame, y, x0, x1, color, alpha, dash=16, gap=12, width=3):
    d = ImageDraw.Draw(frame, "RGBA")
    x = x0
    while x < x1:
        d.line((x, y, min(x + dash, x1), y), fill=color + (int(255 * alpha),), width=width)
        x += dash + gap


# ---------- data ----------

def load_games(path, limit=None):
    games = []
    with open(path, encoding="utf-8") as f:
        while (g := chess.pgn.read_game(f)) is not None:
            games.append(g)
            if limit and len(games) >= limit:
                break
    return games


def load_data():
    progress = [json.loads(l) for l in (ROOT / "runs/main/progress.jsonl").read_text().splitlines()]
    progress = [p for p in progress if p["seconds"] <= 3600]
    ratings = []
    snaps = {Path(json.loads(l)["file"]).stem: json.loads(l)["seconds"]
             for l in (ROOT / "runs/main/snapshots.jsonl").read_text().splitlines()}
    for name in ("runs/curve/curve.json", "runs/curve-early/curve.json"):
        for row in json.loads((ROOT / name).read_text())["results"]:
            if row.get("elo") and snaps.get(row["name"], 1e9) <= 3601:
                ratings.append((snaps[row["name"]], row["elo"]))
    ratings.sort()
    # Untrained game: shortest loss by the initial network (first 100 games of the curve run).
    first = load_games(ROOT / "runs/curve/games.pgn", 100)
    untrained = min((g for g in first if chess.pgn.Game.end(g).board().is_checkmate()),
                    key=lambda g: len(list(g.mainline_moves())))
    untrained_white = untrained.headers["White"] == "ours"
    # 1-hour game: shortest checkmate win for our bot as White.
    hour = [g for g in load_games(ROOT / "runs/reel/hour1.pgn")
            if g.headers["White"].startswith("ScratchChess") and g.headers["Result"] == "1-0" and g.end().board().is_checkmate()]
    showcase = min(hour, key=lambda g: len(list(g.mainline_moves())))
    return progress, ratings, untrained, untrained_white, showcase


# ---------- scenes ----------

class Reel:
    def __init__(self):
        self.progress, self.ratings, untrained, untrained_white, showcase = load_data()
        self.board_size = 912
        self.board_x = (W - self.board_size) // 2
        n_untrained = len(list(untrained.mainline_moves()))
        self.untrained = Replay(untrained, 5.2, lambda k: 3.9 / n_untrained, self.board_size, white_bottom=untrained_white)
        self.untrained_moves = (n_untrained + 1) // 2
        n_show = len(list(showcase.mainline_moves()))
        fast = n_show - 10
        self.showcase = Replay(showcase, 31.9, lambda k: 0.12 if k < fast else 0.34, self.board_size, white_bottom=True)
        self.showcase_moves = (n_show + 1) // 2
        self.start_board = Replay(untrained, 1e9, lambda k: 1, 1024).static(0, -1)
        self.hour_samples = 2_340_880_384

    def scene_alpha(self, t, start, end, fade=0.3):
        return min(ease((t - start) / fade), ease((end - t) / fade))

    def frame(self, t):
        img = Image.new("RGB", (W, H), BG)
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(glow).ellipse((-300, 380, W + 300, 1700), fill=(40, 80, 60, 55))
        img.paste(glow.filter(ImageFilter.GaussianBlur(160)), (0, 0), glow.filter(ImageFilter.GaussianBlur(160))) if t < 0 else None
        for start, end, draw in ((0, 4.6, self.hook), (4.6, 10.6, self.random), (10.6, 15.6, self.training),
                                 (15.6, 20.6, self.how), (20.6, 31.6, self.graph), (31.6, 42.0, self.game),
                                 (42.0, DURATION, self.outro)):
            if start <= t < end:
                layer = Image.new("RGB", (W, H), BG)
                draw(layer, t)
                a = self.scene_alpha(t, start, end) if end < DURATION else min(ease((t - start) / 0.3), ease((DURATION - 0.1 - t) / 0.6))
                img = Image.blend(img, layer, a)
        # Thin progress bar at the very top keeps viewers oriented.
        ImageDraw.Draw(img).rectangle((0, 0, int(W * t / DURATION), 6), fill=GREEN)
        return img

    def hook(self, f, t):
        faint = self.start_board.copy()
        faint.putalpha(faint.getchannel("A").point(lambda a: int(a * 0.10)))
        f.paste(faint, ((W - 1024) // 2, 520), faint)
        text(f, (W / 2, 640), "I trained a chess AI", HEAVY(80), TEXT, appear(t, 0.15), rise=30)
        text(f, (W / 2, 740), "from scratch.", HEAVY(80), TEXT, appear(t, 0.45), rise=30)
        text(f, (W / 2, 960), "1 hour later it plays at", BOLD(54), MUTED, appear(t, 1.7), rise=24)
        text(f, (W / 2, 1070), "GRANDMASTER", HEAVY(118), GOLD, appear(t, 2.1), rise=30)
        text(f, (W / 2, 1180), "strength.", HEAVY(80), TEXT, appear(t, 2.35), rise=30)

    def random(self, f, t):
        header(f, t, 4.7, "TRAINING TIME: 0 SECONDS", "Random weights")
        paste_board(f, self.untrained.render(t), self.board_x, 480, appear(t, 4.8))
        text(f, (W / 2, 1480), "It knows the rules. Nothing else.", BOLD(46), TEXT, appear(t, 5.6), rise=16)
        if t >= self.untrained.end + 0.15:
            a = appear(t, self.untrained.end + 0.15, 0.25)
            cy = 480 + self.untrained.banner_y()
            rounded(f, (W // 2 - 330, cy - 70, W // 2 + 330, cy + 70), 20, (40, 12, 14), 0.9 * a)
            text(f, (W / 2, cy), f"Checkmated in {self.untrained_moves} moves", HEAVY(52), RED, a)

    def training(self, f, t):
        header(f, t, 10.7, "THE TRAINING", "Studying chess positions")
        p = min(1.0, max(0.0, (t - 11.0) / 4.2))
        rows = self.progress[: max(1, int(p * len(self.progress)))]
        # Terminal panel with the real training log.
        a = appear(t, 10.9)
        rounded(f, (60, 480, W - 60, 830), 22, PANEL, a)
        text(f, (100, 520), "training.log  ·  RTX 3080", MONO(28), MUTED, a, anchor="lm")
        for i, row in enumerate(rows[-7:]):
            line = f"step {row['step']:>7,}  {row['positions_per_second']:>7,} pos/s  loss {row['train_mse']:.4f}"
            text(f, (100, 575 + i * 36), line, MONO(29), GREEN if i == len(rows[-7:]) - 1 else (170, 200, 180), a, anchor="lm")
        # Live loss curve (the network's error, falling).
        chart = LogChart((140, 900, W - 90, 1200), 0, 60, 0.06, 0.16, xlog=False)
        rounded(f, (60, 860, W - 60, 1270), 22, PANEL, a)
        text(f, (100, 890), "error vs Stockfish's evaluations", REG(28), MUTED, a, anchor="lm")
        text(f, (W - 100, 1240), "minutes", REG(26), MUTED, a, anchor="rm")
        pts = [(r["seconds"] / 60, min(0.16, r["train_mse"])) for r in rows]
        smooth_polyline(f, chart, pts, GREEN, 4, a)
        # Counter.
        seen = int(self.hour_samples * p)
        text(f, (W / 2, 1360), f"{seen:,}", HEAVY(84), TEXT, appear(t, 11.0))
        text(f, (W / 2, 1440), "positions studied", BOLD(40), MUTED, appear(t, 11.0))
        if p >= 1:
            text(f, (W / 2, 1500), "in the first hour", BOLD(40), GOLD, appear(t, 15.2, 0.2))

    def how(self, f, t):
        header(f, t, 15.7, "HOW IT WORKS", "Everything built from scratch")
        cards = [
            ("THE BRAIN", "6.7M-parameter neural network", "Starts from random numbers"),
            ("THE TEACHER", "360M positions from Lichess", "Each scored by Stockfish"),
            ("THE SEARCH", "Our own C++ engine", "Looks ~12 moves ahead"),
        ]
        for i, (label, main, sub) in enumerate(cards):
            a = appear(t, 16.1 + i * 0.55)
            y = 520 + i * 300 + int((1 - a) * 30)
            rounded(f, (80, y, W - 80, y + 250), 26, PANEL, a)
            text(f, (130, y + 60), label, BOLD(32), GREEN, a, anchor="lm")
            text(f, (130, y + 130), main, HEAVY(52), TEXT, a, anchor="lm")
            text(f, (130, y + 195), sub, REG(38), MUTED, a, anchor="lm")

    def graph(self, f, t):
        reveal = t >= 29.0
        header(f, t, 20.7, "RATING VS TRAINING TIME", "How strong did it get?" if not reveal else "")
        if reveal:
            text(f, (W / 2, 372), "2600 after 1 hour", HEAVY(76), GOLD, appear(t, 29.0), rise=20)
        chart = LogChart((150, 520, W - 70, 1330), 2, 5000, 800, 2800)
        a = appear(t, 20.9)
        d = ImageDraw.Draw(f)
        for y in range(1200, 2801, 400):
            _, py = chart.px(2, y)
            d.line((150, py, W - 70, py), fill=(38, 43, 55), width=2)
            text(f, (130, py), f"{y}", REG(28), MUTED, a, anchor="rm")
        for x, label in ((2, "2s"), (10, "10s"), (60, "1 min"), (600, "10 min"), (3600, "1 h")):
            px, _ = chart.px(x, 800)
            text(f, (px, 1370), label, REG(30), MUTED, a)
        for level, label, color in ((2200, "Master", MUTED), (2500, "Grandmaster", GOLD)):
            _, py = chart.px(2, level)
            dashed(f, py, 150, W - 70, color, 0.8 * appear(t, 21.4))
            # Left side: the late data points and callouts live on the right.
            text(f, (160, py - 26), label, BOLD(30), color, appear(t, 21.4), anchor="lm")
        # Line draws left to right over log time (slightly past 1 h so the last point lands).
        p = min(1.0, max(0.0, (t - 21.8) / 6.6))
        lx0, lx1 = math.log10(2), math.log10(3700)
        cut = 10 ** (lx0 + (lx1 - lx0) * p)
        pts = [(s, e) for s, e in self.ratings if s <= cut]
        if pts and cut > pts[-1][0]:
            nxt = next(((s, e) for s, e in self.ratings if s > cut), None)
            if nxt:
                (s0, e0), (s1, e1) = pts[-1], nxt
                u = (math.log10(cut) - math.log10(s0)) / (math.log10(s1) - math.log10(s0))
                pts.append((cut, e0 + (e1 - e0) * u))
        smooth_polyline(f, chart, pts, GREEN, 5)
        callouts = {8: "8 sec", 76: "76 sec", 758: "13 min", 3600: "1 hour"}
        offsets = {8: (0, 75), 76: (0, -75), 758: (-40, -80), 3600: (-60, 90)}
        for s, e in self.ratings:
            if s > cut:
                continue
            px, py = chart.px(s, e)
            d.ellipse((px - 11, py - 11, px + 11, py + 11), fill=GREEN, outline=BG, width=4)
            key = next((k for k in callouts if abs(k - s) / k < 0.1), None)
            if key:
                ca = appear(t, 21.8 + 6.6 * (math.log10(s) - lx0) / (lx1 - lx0), 0.3)
                dx, dy = offsets[key]
                bx, by = min(max(px + dx, 135), W - 135), py + dy
                rounded(f, (int(bx - 115), int(by - 42), int(bx + 115), int(by + 42)), 14, (30, 36, 48), ca)
                text(f, (bx, by - 12), callouts[key], BOLD(26), MUTED, ca)
                text(f, (bx, by + 16), f"{e}", HEAVY(34), GOLD if e >= 2500 else TEXT, ca)
        text(f, (W / 2, 1440), "Rated vs Stockfish at fixed strength levels · 100 games per point", REG(27), MUTED, appear(t, 22.5))
        text(f, (W / 2, 1500), "10 more hours of training added only ~70 points.", BOLD(34), TEXT, appear(t, 29.8), rise=12)

    def game(self, f, t):
        header(f, t, 31.7, "AFTER 1 HOUR OF TRAINING", "vs Stockfish 2500")
        by = 500
        text(f, (self.board_x, by - 30), "Stockfish (2500)", BOLD(32), MUTED, appear(t, 31.8), anchor="lm")
        paste_board(f, self.showcase.render(t), self.board_x, by, appear(t, 31.8))
        text(f, (self.board_x, by + self.board_size + 34), "Our AI · 1 hour of training", BOLD(32), GREEN, appear(t, 31.8), anchor="lm")
        text(f, (W / 2, 1515), "Scores 64% vs Stockfish 2500 over 100 games", BOLD(38), TEXT, appear(t, 33.0), rise=12)
        if t >= self.showcase.end + 0.1:
            a = appear(t, self.showcase.end + 0.1, 0.25)
            cy = by + self.showcase.banner_y()
            rounded(f, (W // 2 - 300, cy - 80, W // 2 + 300, cy + 80), 24, (12, 30, 20), 0.92 * a)
            text(f, (W / 2, cy - 18), "CHECKMATE", HEAVY(84), GREEN, a)
            text(f, (W / 2, cy + 47), f"in {self.showcase_moves} moves", BOLD(34), TEXT, a)

    def outro(self, f, t):
        text(f, (W / 2, 700), "Random weights", HEAVY(76), MUTED, appear(t, 42.1), rise=24)
        text(f, (W / 2, 800), "\u2193", HEAVY(76), MUTED, appear(t, 42.3))
        text(f, (W / 2, 930), "2600", HEAVY(150), GOLD, appear(t, 42.5), rise=30)
        text(f, (W / 2, 1060), "in 1 hour.", HEAVY(76), TEXT, appear(t, 42.8), rise=24)
        text(f, (W / 2, 1220), "Own network · own engine · one GPU", BOLD(38), GREEN, appear(t, 43.4), rise=12)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/reel/reel.mp4")
    p.add_argument("--music", default="runs/reel/music.wav")
    p.add_argument("--preview", type=float, nargs="*", help="Write PNG stills at these times instead of a video")
    args = p.parse_args()
    reel = Reel()
    if args.preview:
        for t in args.preview:
            path = ROOT / f"runs/reel/preview-{t:05.1f}.png"
            reel.frame(t).save(path)
            print(path)
        return
    command = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-", "-i", str(ROOT / args.music), "-c:v", "libx264", "-preset", "medium",
               "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
               "-movflags", "+faststart", str(ROOT / args.out)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    frames = int(DURATION * FPS)
    for i in range(frames):
        encoder.stdin.write(reel.frame(i / FPS).tobytes())
        if i % 150 == 0:
            print(f"frame {i}/{frames}", flush=True)
    encoder.stdin.close()
    encoder.wait()
    print(ROOT / args.out)


if __name__ == "__main__":
    main()
