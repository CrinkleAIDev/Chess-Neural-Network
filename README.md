# ScratchChess

A chess evaluation network trained **from random weights** on Stockfish-analysed
Lichess positions, played by **our own native C++ engine**. Stockfish supplies
training labels and serves as a test opponent. It never picks our bot's moves. No
pretrained weights, opening book or tablebases are used by the bot.

## Quick start: play the bot

The trained network (`runs/main/best.scn`) and a prebuilt Windows engine
(`native/scratchchess.exe`) are included. The engine needs a 64-bit CPU with AVX2
(almost any CPU from 2015 onwards).

**In the browser** (needs Python 3.11+):

```powershell
python -m pip install "chess==1.999" numpy
python -m scratch_chess.gui          # opens http://127.0.0.1:8765
```

**In any chess GUI** (Cute Chess, Arena, BanksiaGUI, ...): add `native\scratchchess.exe`
as a UCI engine with the argument `--model "C:\full\path\to\runs\main\best.scn"`.

**Other platforms**: build the engine from `native/engine.cpp` (single file, C++17),
e.g. `g++ -std=c++17 -O3 -march=native -pthread native/engine.cpp -o scratchchess`.

Everything below is for retraining, which needs an NVIDIA GPU, `.\setup.ps1`, and
the Lichess data (see [Data](#data)).

## Start the 10-hour run

```powershell
# Foreground (Ctrl+C once to stop cleanly):
.\train.ps1
# Or in the background, with logs:
.\start-training.ps1
# Progress from another terminal:
.\.venv\Scripts\python.exe -m scratch_chess.status --run runs/main
Get-Content runs/main/train.log -Tail 10 -Wait
```

Defaults: `data/lichess-full`, width 512, batch 16384, 10 hours. The learning rate
warms up for a minute, then follows a cosine decay across the **whole** budget, so
the run uses all 10 hours instead of stopping early. `best.npz` is the lowest
validation loss, not proven highest Elo; confirm with matches.

Stop: create `runs/main/STOP` (or Ctrl+C). Resume: delete `STOP`, then
`.\train.ps1 -Resume`. The hour budget includes time already trained.

Keep the PC awake. Training uses the GPU plus ~3 CPU cores, so engine work and
matches can run at the same time (they slow training a little).

## Saved automatically (reel material)

- `initial.npz/.scn`: the untrained network.
- `checkpoint-step0000100` … `checkpoint-step3000000`: log-spaced early snapshots.
- `checkpoint-XXXXXXs`: hourly snapshots.
- `best` / `last`: `.pt` (resumable), `.npz` (Python), `.scn` (native engine).
- `metrics.jsonl`, `progress.jsonl`, `snapshots.jsonl`, `config.json`.

## Play and test

```powershell
.\play.ps1 -Model runs/main/best.npz          # terminal game vs the native engine
.\.venv\Scripts\python.exe -m scratch_chess.bench --model runs/main/best.npz
```

In a chess GUI, add `native\scratchchess.exe` as a UCI engine with the argument
`--model "C:\full\path\to\runs\main\best.scn"`. Options: `Hash` (MB), `Pruning`.

### Engine and network A/B tests (fastchess, SPRT)

```powershell
# Engine change: build a candidate, compare with the saved baseline build.
.\.venv\Scripts\python.exe -m scratch_chess.native build
.\ab.ps1 -Name my-change                                     # new exe vs native\baseline.exe
# Two networks in the same engine:
.\ab.ps1 -Base native\scratchchess.exe -NewModel runs\a\best.scn -BaseModel runs\b\best.scn -Name a-vs-b
```

Results print every 100 games (`Elo`, `LOS`, `LLR`). SPRT stops automatically at
LLR ±2.94. Openings come from `tools/books/8moves_v3.pgn`, played in colour-reversed pairs.

Against Stockfish at a limited strength (a reference point, not a rating):

```powershell
$sf = (Get-Content tools/stockfish/source.json -Raw | ConvertFrom-Json).executables[0]
.\.venv\Scripts\python.exe -m scratch_chess.match --model runs/main/best.npz --stockfish $sf --stockfish-elo 2000 --games 100 --movetime 0.2 --out runs/vs-sf2000
```

`UCI_Elo` is calibrated at longer time controls. Hundreds of games, several
opponents and stated conditions are needed before quoting a rating.

## What is ours

- `features.py`: king-bucketed piece-square features from both players' views.
  The second view is derived exactly from the first (`other_perspective`).
- `model.py`: embedding (512 wide by default) → 32 → 32 → 1, trained on
  `tanh(cp/400)`. A shared piece-square table is folded in at export.
- `native/engine.cpp`: board, legal move generation (perft-verified), incremental
  accumulator updates, iterative deepening PVS with aspiration windows, a
  transposition table, null-move pruning, reverse futility, futility and late move
  pruning, logarithmic LMR, internal iterative reduction, killer/history ordering,
  check extensions, quiescence with delta pruning, and repetition/50-move handling.
- `train.py` / `dataset.py`: streamed, shuffled shards from disk; mixed precision;
  atomic, resumable checkpoints.
- `search.py` / `uci.py`: the original Python engine, kept as a readable
  reference (`--python` flags on `play`/`match`).

## Data

Source: [Lichess evaluation export](https://database.lichess.org/#evals), CC0,
about 410M positions. The importer streams the compressed file and parses with
10 processes:

```powershell
.\.venv\Scripts\python.exe -m scratch_chess.prepare --positions 1000000000 --workers 10 --out data/lichess-full
```

It takes the deepest analysis (minimum depth 16) and drops invalid, terminal and
in-check positions. Scores are converted to side-to-move and stored as 80-byte rows
(side-to-move features only). Positions go to train or validation (2%) by a hash of
the canonical position, so duplicates and colour mirrors never straddle the split.
There is no global dedup (duplicates are ~0.5%). `data/lichess-100m` and older
datasets use the previous 152-byte format, which is still readable.

## License

MIT, see [LICENSE](LICENSE). This covers the code and the trained network.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Covers perft (6 standard positions), legal moves vs python-chess, incremental
accumulator and eval parity with Python, derived-perspective exactness, root draw
handling, repetition avoidance, data split separation and checkpoint resume.
`SCRATCH_ENGINE` / `SCRATCH_MODEL` point the native tests at other builds or networks.
