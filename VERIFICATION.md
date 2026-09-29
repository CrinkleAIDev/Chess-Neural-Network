# Verified on this PC (2026-09-16)

RTX 3080 10 GB, Ryzen 9 9900X, 32 GB RAM, Python 3.12, PyTorch 2.6 (CUDA).

## Native engine

- Perft matches the standard values on 6 reference positions (castling, en passant,
  promotions, pins). Legal moves match python-chess on 150 random positions.
- Incremental accumulator equals a full rebuild after every legal move, and the
  evaluation matches the Python network within 1e-4 on 150 positions (widths 128 and 512).
- Speed (1 s/position, 5-position bench): the Python engine did ~20k nodes/s at depth ~5.
  The native engine now does ~470k nodes/s at depth ~12.8 (width 256) and ~435k at
  depth ~12.6 (width 512).

Search changes, each SPRT-tested against the previous build (fastchess, 5+0.05,
8moves_v3 book, same network on both sides):

| Change | Result |
|---|---|
| TT cutoffs at non-PV nodes, no stalemate probe in qsearch, soft time limit | +196 ± 78 Elo (100 games) |
| Reverse futility pruning, logarithmic LMR | +147 ± 60 (100 games) |
| Late move pruning, futility pruning, qsearch delta pruning | +196 ± 60 (100 games) |
| History malus + gravity, internal iterative reduction | +33 ± 28 (400 games, LOS 99%) |
| SIMD-friendly accumulator/eval loops (identical output) | update 7,140 → 69 ns at width 256; ~3x nodes/s |

Gains at this very fast time control overstate gains at slower ones; each change was
still clearly positive.

## Network width (20-minute equal-time runs, 100M positions, run in parallel)

| Width | Best val MSE | Positions seen |
|---|---|---|
| 128 | 0.0875 | 336M |
| 256 | 0.0818 | 252M |
| 512 | 0.0793 | 171M |

In the engine, width 512 vs width 256 (10+0.1, 196 games): -2 ± 41 Elo, i.e. equal,
despite 512 having seen far fewer positions. 512 was chosen for the 12-hour run because
its lower loss should matter more with long training on 4x the data. This is a judgement
call, not yet a measured result.

## Training pipeline

- Streaming shuffled shards from disk; the derived opponent view is exact (300 random
  positions, numpy and torch).
- Width 512, batch 16384: ~680k positions/s, GPU-bound (97% utilisation). Batch 32768
  was only ~6% faster.
- 26 automated tests pass.
