"""Generate the datasets embedded in docs/ (the interactive article).

Everything the article claims numerically is produced here from the real
engine, so the figures can't drift away from the research. Writes a single
`docs/data.js` defining `window.HT`.

Run: python3 viz/build_article_data.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hex_truchet._hexcore import (  # noqa: E402
    Board,
    canonical_tiles,
    enclosed_cells,
    hex_board,
    orbit_size,
    span_multiset,
    tile_arcs,
)

L = 60.0  # hex edge-to-edge scale, matching build_viewer.py


# --------------------------------------------------------------------------
# geometry


def axial_to_px(q: int, r: int) -> tuple[float, float]:
    """Axial -> screen pixel, matching export_game.py's `center()` after its
    y-flip. Adjacent cells land exactly 2*apothem apart in both directions."""
    return (L * q + 0.5 * L * r, (math.sqrt(3) / 2.0) * L * r)


def board_geometry(radius: int) -> dict:
    """Board layout, reusing the *validated* edge ordering from game_data.json.

    export_game.py doesn't just place the six edge offsets at 60-degree steps
    -- it checks each against the engine's own `neighbor(cell, edge)` and
    remaps until edge index i really is the direction the engine thinks it is.
    Recomputing that here by hand would be a silent way to draw arcs that
    connect to the wrong neighbour, so the offsets are taken from the file
    that geometry already produced.
    """
    game = json.loads((ROOT / "viz" / "game_data.json").read_text())
    cells = hex_board(radius)
    assert [tuple(c["qr"]) for c in game["cells"]] == list(cells), (
        "game_data.json cell order no longer matches hex_board(); "
        "sample-board cell indices would be wrong"
    )
    pts = [axial_to_px(q, r) for (q, r) in cells]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = L * 0.75
    minx, miny = min(xs) - pad, min(ys) - pad
    return {
        "radius": radius,
        "L": L,
        "W": round((max(xs) + pad) - minx, 2),
        "H": round((max(ys) + pad) - miny, 2),
        "edge_off": game["edge_off"],
        "cells": [
            {"idx": i, "qr": list(c), "c": [round(pts[i][0] - minx, 2), round(pts[i][1] - miny, 2)]}
            for i, c in enumerate(cells)
        ],
    }


# --------------------------------------------------------------------------
# tiles

_TILE_INFO = canonical_tiles()
# The engine's tile identity is the matching itself; the dicts carry metadata.
TILES = [t["matching"] for t in _TILE_INFO]


def tile_table() -> list[dict]:
    """The 5 canonical tiles, with every rotation's arcs pre-sampled.

    `orbit` is the number of *visually distinct* rotations -- the orbit of the
    matching under the rotation group. Tile 4 has orbit 1, which is why the
    game's UI has to skip aliased rotations or the rotate key looks broken.
    """
    out = []
    for i, info in enumerate(_TILE_INFO):
        m = info["matching"]
        out.append(
            {
                "id": i,
                "spans": list(info["spans"]),
                "orbit": info["orbit"],
                "rots": [[list(p) for p in tile_arcs(m, k)] for k in range(6)],
            }
        )
    return out


# --------------------------------------------------------------------------
# deck experiments


def make_deck(kind: str, n: int, rng: random.Random) -> list[int]:
    if kind == "uniform":
        deck = [i % 5 for i in range(n)]
    elif kind == "tuned":
        n0 = round(n / 3)
        deck = [0] * n0 + [2] * (n - n0)
    else:
        raise ValueError(kind)
    rng.shuffle(deck)
    return deck


def fill_board(cells, deck, rng):
    """Fill every cell with a random rotation of its dealt tile.

    Cells are (q, r) tuples -- the engine keys on coordinates, not indices --
    so the caller maps back to indices for the renderer.
    """
    board = Board(cells)
    order = list(cells)
    rng.shuffle(order)
    placed = []
    for cell, tile in zip(order, deck):
        rot = rng.randrange(6)
        board.place(cell, TILES[tile], rot)
        placed.append((cell, tile, rot))
    return board, placed


def deck_experiment(radius: int, kind: str, trials: int, seed: int) -> dict:
    cells = hex_board(radius)
    rng = random.Random(seed)
    counts, zero, lengths, areas = [], 0, Counter(), []
    for _ in range(trials):
        deck = make_deck(kind, len(cells), rng)
        board, _ = fill_board(cells, deck, rng)
        loops = board.loops()
        counts.append(len(loops))
        if not loops:
            zero += 1
        for lp in loops:
            lengths[lp["length"]] += 1
            areas.append(len(enclosed_cells(board, lp)))
    total_loops = sum(lengths.values())
    return {
        "kind": kind,
        "radius": radius,
        "trials": trials,
        "mean_loops": round(sum(counts) / trials, 3),
        "p_zero": round(zero / trials, 4),
        "mean_area": round(sum(areas) / len(areas), 3) if areas else 0.0,
        "len_hist": {str(k): v for k, v in sorted(lengths.items())},
        "pct_minimal": round(100.0 * lengths[3] / total_loops, 1) if total_loops else 0.0,
    }


def sample_boards(radius: int, kind: str, n: int, seed: int) -> list[dict]:
    """Real filled boards, with their loops, for rendering on the page."""
    cells = hex_board(radius)
    index = {c: i for i, c in enumerate(cells)}
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        deck = make_deck(kind, len(cells), rng)
        board, placed = fill_board(cells, deck, rng)
        tiles = [
            {
                "cell": index[cell],
                "tile": tile,
                "arcs": [list(p) for p in tile_arcs(TILES[tile], rot)],
            }
            for (cell, tile, rot) in placed
        ]
        loops = []
        for lp in board.loops():
            # arc ids -> (cell index, edge_a, edge_b) for the renderer
            arcs = []
            for aid in lp["arcs"]:
                cell, ea, eb = board.arcs[aid]
                arcs.append([index[cell], ea, eb])
            encl = enclosed_cells(board, lp)
            loops.append(
                {
                    "arcs": arcs,
                    "len": lp["length"],
                    # cell indices, so the renderer can tint the captured area
                    "cells": sorted(index[c] for c in encl),
                    "area": len(encl),
                }
            )
        loops.sort(key=lambda d: -d["area"])
        out.append({"tiles": tiles, "loops": loops})
    return out


# --------------------------------------------------------------------------
# research results (from the recorded runs; see HANDOFF.md)

SELFPLAY = [
    {"update": 75, "vs_random": 0.76, "vs_greedy": 0.000, "margin": -29.8},
    {"update": 150, "vs_random": 0.93, "vs_greedy": 0.000, "margin": -32.7},
    {"update": 225, "vs_random": 0.90, "vs_greedy": 0.000, "margin": -23.8},
    {"update": 375, "vs_random": 0.86, "vs_greedy": 0.000, "margin": -25.6},
    {"update": 450, "vs_random": 0.86, "vs_greedy": 0.004, "margin": -24.8},
]

VS_GREEDY = [
    {"update": 20, "vs_greedy": 0.000, "margin": -26.6},
    {"update": 60, "vs_greedy": 0.000, "margin": -20.4},
    {"update": 120, "vs_greedy": 0.008, "margin": -14.2},
    {"update": 150, "vs_greedy": 0.055, "margin": -12.8},
    {"update": 175, "vs_greedy": 0.047, "margin": -12.1},
    {"update": 225, "vs_greedy": 0.086, "margin": -9.5},
    {"update": 300, "vs_greedy": 0.060, "margin": -9.6},
]

SWEEP = [
    {"k": 1, "depth": 0, "win": 0.467, "margin": 0.00, "sec": 0.009, "n": 30, "label": "greedy"},
    {"k": 2, "depth": 4, "win": 0.900, "margin": 9.10, "sec": 0.051, "n": 30},
    {"k": 3, "depth": 2, "win": 0.933, "margin": 10.20, "sec": 0.030, "n": 30, "label": "medium"},
    {"k": 3, "depth": 4, "win": 0.900, "margin": 9.67, "sec": 0.072, "n": 30},
    {"k": 3, "depth": 8, "win": 0.900, "margin": 12.43, "sec": 0.140, "n": 30, "label": "hard"},
    {"k": 4, "depth": 4, "win": 0.933, "margin": 10.23, "sec": 0.156, "n": 30},
    {"k": 5, "depth": 4, "win": 0.933, "margin": 10.23, "sec": 0.168, "n": 30},
    {"k": 5, "depth": 8, "win": 0.900, "margin": 13.03, "sec": 0.368, "n": 30},
    {"k": 6, "depth": 6, "win": 0.800, "margin": 8.43, "sec": 0.294, "n": 30},
    {"k": 8, "depth": 0, "win": 1.000, "margin": 22.64, "sec": 29.0, "n": 50, "label": "expert"},
]


# --------------------------------------------------------------------------


def main() -> None:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)

    print("geometry...")
    geo3 = board_geometry(3)

    print("tiles...")
    tiles = tile_table()

    print("deck experiments (this takes a minute)...")
    trials = 2000
    decks = {
        "uniform_r3": deck_experiment(3, "uniform", trials, seed=11),
        "tuned_r3": deck_experiment(3, "tuned", trials, seed=12),
        "uniform_r4": deck_experiment(4, "uniform", trials, seed=13),
        "tuned_r4": deck_experiment(4, "tuned", trials, seed=14),
    }
    for k, v in decks.items():
        print(f"  {k}: mean={v['mean_loops']} p0={v['p_zero']} minimal={v['pct_minimal']}%")

    print("sample boards...")
    samples = {
        "uniform": sample_boards(3, "uniform", 6, seed=21),
        "tuned": sample_boards(3, "tuned", 6, seed=22),
    }

    payload = {
        "geo": geo3,
        "tiles": tiles,
        "decks": decks,
        "samples": samples,
        "selfplay": SELFPLAY,
        "vs_greedy": VS_GREEDY,
        "sweep": SWEEP,
    }

    game_path = ROOT / "viz" / "game_data.json"
    if game_path.exists():
        payload["game"] = json.loads(game_path.read_text())

    out = docs / "data.js"
    out.write_text("window.HT = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
