"""Stage 0d: broader deck weight search.

Balance priority means we want: low P(zero), moderate-to-high mean loops,
real spread in loop length AND area, and multi-closure events present but
not dominant. Search corners, pairs, and interior points of the 5-simplex.
"""

import random
import itertools
import numpy as np
from collections import Counter

from geometry import hex_board, canonical_tiles
from graph import Board
from stage0 import enclosed_cells

TILES = canonical_tiles()
N = len(TILES)


def evaluate(radius, weights, n_trials, seed, area_every=15):
    rng = random.Random(seed)
    cells = hex_board(radius)
    idx = list(range(N))
    loop_counts, loop_lengths, areas = [], [], []
    multi = Counter()

    for trial in range(n_trials):
        b = Board(cells)
        placed = []
        for cell in cells:
            t = rng.choices(idx, weights=weights)[0]
            rot = rng.randrange(6)
            b.place(cell, TILES[t]["matching"], rot)
            placed.append((cell, t, rot))
        loops = b.loops()
        loop_counts.append(len(loops))
        loop_lengths.extend(l["length"] for l in loops)
        if trial % area_every == 0:
            for l in loops:
                areas.append(len(enclosed_cells(b, l)))
        if trial % 8 == 0:
            order = list(range(len(cells)))
            rng.shuffle(order)
            b2 = Board(cells)
            prev = 0
            for i in order:
                cell, t, rot = placed[i]
                b2.place(cell, TILES[t]["matching"], rot)
                n = len(b2.loops())
                multi[n - prev] += 1
                prev = n

    lc = np.array(loop_counts)
    ll = np.array(loop_lengths) if loop_lengths else np.array([0.0])
    ar = np.array(areas) if areas else np.array([0.0])
    tm = sum(multi.values()) or 1
    return {
        "mean_loops": lc.mean(),
        "p_zero": float((lc == 0).mean()),
        "frac_minimal": float((ll == 3).mean()),
        "mean_area": ar.mean(),
        "area_std": ar.std(),
        "len_std": ll.std(),
        "multi_rate": (tm - multi.get(0, 0) - multi.get(1, 0)) / tm,  # 2+ closures
        "any_closure_rate": (tm - multi.get(0, 0)) / tm,
    }


def score(r):
    """Composite balance score: reward hitting targets, penalize misses.
    Higher is better. This is a heuristic for ranking, not a real utility.
    """
    s = 0.0
    # P(zero) should be low
    s -= 15 * r["p_zero"]
    # mean loops: target band 3-8, penalize outside
    if r["mean_loops"] < 3:
        s -= (3 - r["mean_loops"]) * 2
    elif r["mean_loops"] > 10:
        s -= (r["mean_loops"] - 10) * 0.5
    else:
        s += 2
    # want NOT all-minimal -- reward length spread
    s += min(r["len_std"], 5) * 1.5
    # want area spread
    s += min(r["area_std"], 4) * 1.5
    # want some multi-closures but not too many (drama without chaos)
    if 0.03 < r["multi_rate"] < 0.15:
        s += 3
    else:
        s -= abs(r["multi_rate"] - 0.07) * 10
    return s


# Corners and pairs (as before, for continuity) plus new candidates
WEIGHT_SETS = {
    "0+2 only":        [1, 0, 1, 0, 0],
    "0+3 only":        [1, 0, 0, 1, 0],
    "0+1 only":        [1, 1, 0, 0, 0],
    "0+4 only":        [1, 0, 0, 0, 1],
    "2+3 only":        [0, 0, 1, 1, 0],
    "0,2,3 equal":      [1, 0, 1, 1, 0],
    "0,1,2 (3:1:2)":    [3, 1, 2, 0, 0],
    "0,1,2 (2:1:2)":    [2, 1, 2, 0, 0],
    "0,2 (2:1)":        [2, 0, 1, 0, 0],
    "0,2 (3:1)":        [3, 0, 1, 0, 0],
    "0,2 (1:2)":        [1, 0, 2, 0, 0],
    "0,2,4 (3:2:1)":    [3, 0, 2, 0, 1],
    "0,1,2,3 equal":    [1, 1, 1, 1, 0],
    "all equal":        [1, 1, 1, 1, 1],
    "0 heavy w/ 4":     [4, 0, 1, 0, 1],
    "0,3 only":         [1, 0, 0, 1, 0],
    "0,2,3 (2:1:1)":    [2, 0, 1, 1, 0],
    "0,2,3 (3:2:1)":    [3, 0, 2, 1, 0],
}

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    results = {}
    for radius in (3, 4):
        print(f"\n{'='*100}")
        print(f"radius {radius} ({len(hex_board(radius))} cells), n={n}")
        print(f"{'='*100}")
        print(f"{'deck':<20}{'loops':>7}{'P(0)':>7}{'%min':>7}{'area':>7}"
              f"{'areaSD':>8}{'lenSD':>8}{'2+clos':>8}{'score':>8}")
        print("-" * 100)
        rows = []
        for name, w in WEIGHT_SETS.items():
            r = evaluate(radius, w, n, seed=hash((name, radius)) % 100000)
            sc = score(r)
            rows.append((sc, name, w, r))
        rows.sort(reverse=True)
        for sc, name, w, r in rows:
            print(f"{name:<20}{r['mean_loops']:>7.2f}{r['p_zero']:>7.3f}"
                  f"{r['frac_minimal']*100:>6.1f}%{r['mean_area']:>7.2f}"
                  f"{r['area_std']:>8.2f}{r['len_std']:>8.2f}"
                  f"{r['multi_rate']:>8.3f}{sc:>8.2f}")
        results[radius] = rows
