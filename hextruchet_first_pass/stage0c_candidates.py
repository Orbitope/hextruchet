"""Stage 0c: full distributions for candidate decks that passed the sweep."""

import random
import numpy as np
from collections import Counter

from geometry import hex_board, canonical_tiles
from graph import Board
from stage0 import enclosed_cells

TILES = canonical_tiles()

CANDIDATES = [
    ("tiles 0+2 only",  [1, 0, 1, 0, 0]),
    ("span1 heavy 4x",  [4, 1, 1, 1, 1]),
    ("0,1,2 balanced",  [1, 1, 1, 0, 0]),
    ("span1 heavy 8x",  [8, 1, 1, 1, 1]),
]


def detail(radius, weights, n_trials, seed, area_every=20):
    rng = random.Random(seed)
    cells = hex_board(radius)
    idx = list(range(len(TILES)))
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

        # multi-closure check on incremental replay
        if trial % 5 == 0:
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

    return loop_counts, loop_lengths, areas, multi


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    for radius in (3, 4):
        for name, w in CANDIDATES:
            lc, ll, ar, multi = detail(radius, w, n, seed=radius * 77 + len(name))
            lc = np.array(lc); ll = np.array(ll); ar = np.array(ar)
            print(f"\n{'='*68}")
            print(f"r{radius} | {name}  ({n} fills)")
            print(f"{'='*68}")
            print(f"loops/board: mean {lc.mean():.2f}  median {np.median(lc):.0f}  "
                  f"P(zero) {(lc==0).mean():.3f}  max {lc.max()}")
            c = Counter(ll.tolist()); tot = len(ll)
            print("loop length distribution:")
            for k in sorted(c):
                if k <= 15:
                    bar = "#" * int(60 * c[k] / tot)
                    print(f"  {k:3d}: {c[k]/tot:6.3f} {bar}")
            big = sum(v for k, v in c.items() if k > 15)
            if big:
                print(f"  >15: {big/tot:6.3f}")
            if len(ar):
                ca = Counter(ar.tolist()); ta = len(ar)
                print(f"enclosed area: mean {ar.mean():.2f}  median {np.median(ar):.0f}  max {ar.max()}")
                print("  area distribution:", {k: round(ca[k]/ta, 3) for k in sorted(ca) if k <= 8})
            tm = sum(multi.values())
            print("loops closed per placement:",
                  {k: round(v/tm, 4) for k, v in sorted(multi.items()) if k != 0})
