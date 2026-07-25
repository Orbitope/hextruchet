"""Stage 0e: attack the minimal-loop-dominance problem directly.

Hypothesis: span-1 arcs (tile 0's specialty) are necessary for ANY closure,
but every span-1 arc is also a chance to complete a 3-loop immediately.
Test: does removing tile 0 entirely kill closures, or can tile 2 alone
(spans 1,1,3) still close loops without so much minimality? Also test
fractional/non-corner ratios more finely, and a completely different
lever: does board radius/shape change the minimal fraction independent
of deck (i.e. is this a structural ceiling)?
"""

import random
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
        "n_loops_total": len(loop_lengths),
    }


print("### Does tile 2 ALONE close loops without tile 0? ###")
for radius in (3, 4):
    r = evaluate(radius, [0, 0, 1, 0, 0], 500, seed=radius+1)
    print(f"r{radius} tile-2-only: loops={r['mean_loops']:.2f} "
          f"P(0)={r['p_zero']:.3f} %min={r['frac_minimal']*100:.1f} "
          f"area={r['mean_area']:.2f}")

print("\n### tile 2 + tile 3 (both have non-adjacent long arcs, no tile 0) ###")
for radius in (3, 4):
    r = evaluate(radius, [0, 0, 1, 1, 0], 500, seed=radius+2)
    print(f"r{radius} 2+3: loops={r['mean_loops']:.2f} P(0)={r['p_zero']:.3f} "
          f"%min={r['frac_minimal']*100:.1f} area={r['mean_area']:.2f}")

print("\n### Fine ratio sweep of 0:2, looking for minimum %minimal "
      "subject to P(zero)<0.05 ###")
ratios = [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,8),
          (2,3), (2,5), (3,4), (3,5), (3,7), (2,7), (1,10)]
for radius in (3, 4):
    print(f"\n-- radius {radius} --")
    rows = []
    for a, b_ in ratios:
        w = [a, 0, b_, 0, 0]
        r = evaluate(radius, w, 400, seed=hash((radius, a, b_)) % 99999)
        rows.append((a, b_, r))
    rows.sort(key=lambda x: x[2]["frac_minimal"])
    for a, b_, r in rows:
        flag = "" if r["p_zero"] < 0.05 else "  [P(0) too high]"
        print(f"  0:2 = {a}:{b_:<3} loops={r['mean_loops']:>5.2f} "
              f"P(0)={r['p_zero']:.3f} %min={r['frac_minimal']*100:>5.1f} "
              f"area={r['mean_area']:.2f} n={r['n_loops_total']}{flag}")

print("\n### Does board radius alone (fixed uniform deck) change %minimal? ###")
print("(tests whether minimality is a deck property or a structural ceiling)")
for radius in (2, 3, 4, 5):
    r = evaluate(radius, [1,1,1,1,1], 400, seed=radius+50)
    print(f"  r{radius} uniform: loops={r['mean_loops']:.2f} "
          f"%min={r['frac_minimal']*100:.1f} n={r['n_loops_total']}")
