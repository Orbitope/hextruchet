"""Stage 0f: do blank/partial spacer tiles suppress minimal-loop dominance?

Hypothesis: minimal 3-loops require 3 mutually-adjacent tiles to each
contribute a specific tight arc. A spacer tile (0 or 2 arcs) breaks that
adjacency requirement structurally -- it can sit in the middle of what
would have been a 3-loop and simply not connect, forcing any closure that
does happen to route around it and be longer.
"""

import random
import numpy as np
from collections import Counter

from geometry import hex_board, canonical_tiles, rotate_matching
from geometry_ext import blank_tile, partial_matchings
from graph import Board
from stage0 import enclosed_cells

FULL_TILES = canonical_tiles()
BLANK = blank_tile()
PARTIALS = partial_matchings()


def make_deck_sampler(rng, full_weights, blank_weight, partial_weight):
    """Returns a function that samples one tile's matching."""
    full_ids = list(range(len(FULL_TILES)))
    total_full = sum(full_weights)
    total = total_full + blank_weight + partial_weight

    def sample():
        r = rng.random() * total
        if r < total_full:
            # pick among full tiles by weight
            t = rng.choices(full_ids, weights=full_weights)[0]
            return FULL_TILES[t]["matching"]
        r -= total_full
        if r < blank_weight:
            return BLANK
        return rng.choice(PARTIALS)
    return sample


def evaluate(radius, full_weights, blank_w, partial_w, n_trials, seed, area_every=15):
    rng = random.Random(seed)
    cells = hex_board(radius)
    sampler = make_deck_sampler(rng, full_weights, blank_w, partial_w)

    loop_counts, loop_lengths, areas = [], [], []
    multi = Counter()
    for trial in range(n_trials):
        b = Board(cells)
        placed = []
        for cell in cells:
            m = sampler()
            rot = rng.randrange(6)
            b.place(cell, m, rot)
            placed.append((cell, m, rot))
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
                cell, m, rot = placed[i]
                b2.place(cell, m, rot)
                n2 = len(b2.loops())
                multi[n2 - prev] += 1
                prev = n2

    lc = np.array(loop_counts)
    ll = np.array(loop_lengths) if loop_lengths else np.array([0.0])
    ar = np.array(areas) if areas else np.array([0.0])
    tm = sum(multi.values()) or 1
    return {
        "mean_loops": lc.mean(),
        "p_zero": float((lc == 0).mean()),
        "frac_minimal": float((ll == 3).mean()) if len(loop_lengths) else 1.0,
        "mean_area": ar.mean(),
        "area_std": ar.std(),
        "len_std": ll.std(),
        "multi_rate": (tm - multi.get(0, 0) - multi.get(1, 0)) / tm,
        "n_loops": len(loop_lengths),
    }


# full_weights indexed by FULL_TILES[0..4], plus blank_w, partial_w
SCENARIOS = [
    ("baseline 0:2=1:2, no spacer",        [1,0,2,0,0], 0,  0),
    ("0:2=1:2 + 20% blank",                [1,0,2,0,0], 5,  0),
    ("0:2=1:2 + 40% blank",                [1,0,2,0,0], 10, 0),
    ("0:2=1:2 + 60% blank",                [1,0,2,0,0], 15, 0),
    ("0:2=1:2 + 20% partial",              [1,0,2,0,0], 0,  5),
    ("0:2=1:2 + 40% partial",              [1,0,2,0,0], 0,  10),
    ("0:2=1:2 + 20% blank + 20% partial",  [1,0,2,0,0], 5,  5),
    ("tile2-only + 20% blank",             [0,0,1,0,0], 2,  0),
    ("tile2-only + 40% blank",             [0,0,1,0,0], 4,  0),
    ("tile2-only + 20% partial",           [0,0,1,0,0], 0,  2),
    ("tile2-only + 40% partial",           [0,0,1,0,0], 0,  4),
    ("tile0-only + 40% blank",             [1,0,0,0,0], 4,  0),
    ("tile0-only + 60% blank",             [1,0,0,0,0], 9,  0),
]

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    for radius in (3, 4):
        print(f"\n{'='*100}")
        print(f"radius {radius}, n={n}")
        print(f"{'='*100}")
        print(f"{'scenario':<38}{'loops':>7}{'P(0)':>7}{'%min':>7}{'area':>7}"
              f"{'areaSD':>8}{'lenSD':>8}{'2+clos':>8}")
        print("-"*100)
        for name, fw, bw, pw in SCENARIOS:
            r = evaluate(radius, fw, bw, pw, n, seed=hash((name,radius))%99999)
            print(f"{name:<38}{r['mean_loops']:>7.2f}{r['p_zero']:>7.3f}"
                  f"{r['frac_minimal']*100:>6.1f}%{r['mean_area']:>7.2f}"
                  f"{r['area_std']:>8.2f}{r['len_std']:>8.2f}{r['multi_rate']:>8.3f}")
