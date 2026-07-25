"""Stage 0b: does deck composition rescue loop closure? (Tests H1.)

Uniform deck gave 0.67 loops/board on radius 3 -- failing the gate.
Tile 0 is all-span-1 (three tight turns). Sweep weightings.
"""

import random
import numpy as np
from collections import Counter

from geometry import hex_board, canonical_tiles
from graph import Board

TILES = canonical_tiles()
# tile 0: spans (1,1,1)  -- all tight turns
# tile 1: spans (1,2,2)
# tile 2: spans (1,1,3)
# tile 3: spans (2,2,3)
# tile 4: spans (3,3,3)  -- all straight


def run_weighted(radius, weights, n_trials, seed=0):
    rng = random.Random(seed)
    cells = hex_board(radius)
    idx = list(range(len(TILES)))
    loop_counts = []
    loop_lengths = []
    for _ in range(n_trials):
        b = Board(cells)
        for cell in cells:
            t = rng.choices(idx, weights=weights)[0]
            b.place(cell, TILES[t]["matching"], rng.randrange(6))
        loops = b.loops()
        loop_counts.append(len(loops))
        loop_lengths.extend(l["length"] for l in loops)
    lc = np.array(loop_counts)
    ll = np.array(loop_lengths) if loop_lengths else np.array([0])
    return {
        "mean_loops": lc.mean(),
        "p_zero": (lc == 0).mean(),
        "max_loops": lc.max(),
        "mean_len": ll.mean(),
        "frac_minimal": float((ll == 3).mean()) if loop_lengths else 0.0,
    }


SCENARIOS = [
    ("uniform",            [1, 1, 1, 1, 1]),
    ("no all-straight",    [1, 1, 1, 1, 0]),
    ("span1 heavy 2x",     [2, 1, 1, 1, 1]),
    ("span1 heavy 4x",     [4, 1, 1, 1, 1]),
    ("span1 heavy 8x",     [8, 1, 1, 1, 1]),
    ("tiles 0+2 only",     [1, 0, 1, 0, 0]),
    ("tile 0 only",        [1, 0, 0, 0, 0]),
    ("0,1,2 balanced",     [1, 1, 1, 0, 0]),
    ("orbit-weighted",     [2, 6, 3, 3, 1]),
]


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    for radius in (3, 4):
        print(f"\n{'='*76}")
        print(f"DECK SWEEP -- radius {radius} ({len(hex_board(radius))} cells), "
              f"{n} fills each")
        print(f"{'='*76}")
        print(f"{'deck':<20} {'mean loops':>11} {'P(zero)':>9} "
              f"{'max':>5} {'mean len':>9} {'% minimal':>10}")
        print("-" * 76)
        for name, w in SCENARIOS:
            r = run_weighted(radius, w, n, seed=hash(name) % 10000 + radius)
            print(f"{name:<20} {r['mean_loops']:>11.2f} {r['p_zero']:>9.3f} "
                  f"{r['max_loops']:>5} {r['mean_len']:>9.2f} "
                  f"{r['frac_minimal']*100:>9.1f}%")
