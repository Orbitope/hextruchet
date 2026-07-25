"""Stage 0: geometric baseline. Random fill, measure loop statistics."""

import random
import json
from collections import Counter
import numpy as np

from geometry import hex_board, canonical_tiles, neighbor, opposite_edge
from graph import Board

TILES = canonical_tiles()


def random_fill(cells, rng, deck=None):
    """Fill every cell with a random canonical tile at random rotation."""
    b = Board(cells)
    if deck is None:
        for cell in cells:
            t = rng.randrange(len(TILES))
            b.place(cell, TILES[t]["matching"], rng.randrange(6))
    else:
        d = list(deck)
        rng.shuffle(d)
        for cell, t in zip(cells, d):
            b.place(cell, TILES[t]["matching"], rng.randrange(6))
    return b


def enclosed_cells(board, loop):
    """Cells enclosed by a loop, via ray casting in the infinite plane.

    A loop is a closed curve made of arcs. We determine, for each cell on
    the board, whether it is inside. We use the crossing-parity method along
    a path of cells: walk from the cell outward in a fixed direction,
    counting how many times we cross an arc belonging to the loop.
    """
    loop_arcs = set(loop["arcs"])
    # Ports used by this loop
    loop_ports = set()
    for aid in loop_arcs:
        cell, ea, eb = board.arcs[aid]
        loop_ports.add((cell, ea))
        loop_ports.add((cell, eb))

    inside = []
    for cell in board.cells:
        # Walk in direction 0 until off-board, counting crossings.
        crossings = 0
        cur = cell
        steps = 0
        while steps < 200:
            # crossing edge 0 of cur into neighbor
            if (cur, 0) in loop_ports:
                crossings += 1
            nxt = neighbor(cur, 0)
            if nxt not in board.cells:
                break
            cur = nxt
            steps += 1
        if crossings % 2 == 1:
            inside.append(cell)
    return inside


def replay_closures(cells, tile_choices, rng):
    """Place tiles in random order, recording loops closed per placement."""
    order = list(range(len(cells)))
    rng.shuffle(order)
    b = Board(cells)
    prev_loops = 0
    per_placement = []
    for idx in order:
        cell = cells[idx]
        t, rot = tile_choices[idx]
        b.place(cell, TILES[t]["matching"], rot)
        n = len(b.loops())
        per_placement.append(n - prev_loops)
        prev_loops = n
    return per_placement


def run(radius, n_trials, seed=0, measure_area_every=50):
    rng = random.Random(seed)
    cells = hex_board(radius)
    n_cells = len(cells)

    loop_counts = []
    loop_lengths = []
    run_lengths = []
    terminal_arcs = []
    areas = []
    closures_per_placement = Counter()

    for trial in range(n_trials):
        tile_choices = [
            (rng.randrange(len(TILES)), rng.randrange(6)) for _ in cells
        ]
        b = Board(cells)
        for cell, (t, rot) in zip(cells, tile_choices):
            b.place(cell, TILES[t]["matching"], rot)

        comps = b.components()
        loops = [c for c in comps if c["is_loop"]]
        runs_ = [c for c in comps if not c["is_loop"]]

        loop_counts.append(len(loops))
        loop_lengths.extend(c["length"] for c in loops)
        run_lengths.extend(c["length"] for c in runs_)
        terminal_arcs.append(len(b.open_ports()))

        if trial % measure_area_every == 0:
            for l in loops:
                areas.append(len(enclosed_cells(b, l)))

        if trial % 10 == 0:
            for d in replay_closures(cells, tile_choices, rng):
                closures_per_placement[d] += 1

    return {
        "radius": radius,
        "n_cells": n_cells,
        "n_trials": n_trials,
        "loop_counts": loop_counts,
        "loop_lengths": loop_lengths,
        "run_lengths": run_lengths,
        "terminal_arcs": terminal_arcs,
        "areas": areas,
        "closures_per_placement": dict(closures_per_placement),
    }


def summarize(res):
    lc = np.array(res["loop_counts"])
    ll = np.array(res["loop_lengths"])
    rl = np.array(res["run_lengths"])
    ar = np.array(res["areas"])

    print(f"\n{'='*62}")
    print(f"STAGE 0 -- radius {res['radius']} ({res['n_cells']} cells), "
          f"{res['n_trials']} random fills")
    print(f"{'='*62}")

    print(f"\nLoops per board:")
    print(f"  mean {lc.mean():.2f}   median {np.median(lc):.0f}   "
          f"min {lc.min()}   max {lc.max()}")
    print(f"  P(zero loops) = {(lc == 0).mean():.3f}")

    if len(ll):
        print(f"\nLoop length (arcs):")
        print(f"  mean {ll.mean():.2f}   median {np.median(ll):.0f}   max {ll.max()}")
        c = Counter(ll.tolist())
        tot = len(ll)
        print("  distribution:")
        for k in sorted(c)[:12]:
            print(f"    {k:3d} arcs: {c[k]/tot:6.3f}  ({c[k]})")
        big = sum(v for k, v in c.items() if k > 12)
        if big:
            print(f"    >12 arcs: {big/tot:6.3f}  ({big})")

    if len(ar):
        print(f"\nEnclosed area (cells), sampled n={len(ar)}:")
        print(f"  mean {ar.mean():.2f}   median {np.median(ar):.0f}   max {ar.max()}")
        c = Counter(ar.tolist())
        tot = len(ar)
        for k in sorted(c)[:10]:
            print(f"    {k:3d} cells: {c[k]/tot:6.3f}")

    print(f"\nOpen run length (arcs):")
    print(f"  mean {rl.mean():.2f}   median {np.median(rl):.0f}   max {rl.max()}")

    print(f"\nLoops closed per placement:")
    cpp = res["closures_per_placement"]
    tot = sum(cpp.values())
    for k in sorted(cpp, key=int):
        print(f"    {k}: {cpp[k]/tot:6.4f}  ({cpp[k]})")

    ta = np.array(res["terminal_arcs"])
    print(f"\nOpen ports per board: mean {ta.mean():.1f}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    out = {}
    for radius in (3, 4):
        res = run(radius, n, seed=radius * 1000)
        summarize(res)
        out[radius] = {
            k: v for k, v in res.items()
            if k not in ("loop_lengths", "run_lengths", "areas")
        }
        out[radius]["loop_length_hist"] = dict(Counter(res["loop_lengths"]))
        out[radius]["area_hist"] = dict(Counter(res["areas"]))
        out[radius]["run_length_hist"] = dict(Counter(res["run_lengths"]))
    with open("results_stage0.json", "w") as f:
        json.dump(out, f)
    print("\nwrote results_stage0.json")
