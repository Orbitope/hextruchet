"""Export Hex Truchet games as JSON packs: ground truth for the GDScript port.

Two consumers:
  1. The Godot differential test (tests/test_rules.gd) replays each pack's move
     sequence through its own GameState and asserts identical per-step gained /
     scores / loop sets. This is what makes the GDScript rules engine trustable.
  2. Godot's replay mode + the web viewer, which render the same packs.

Also exports the STATIC tables (cell coords, neighbor map, per-tile arc edge
pairs for every rotation) so the port can verify its geometry matches exactly
rather than silently diverging on, say, edge-index conventions.

Usage:
    python training/export_pack.py --out hextruchet-godot/data/packs --games 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hex_truchet"))
import _hexcore as H  # noqa: E402

RADIUS = 3
L = 60.0  # cell spacing in px, matches the web viewer


# --------------------------------------------------------------------------
# static geometry (identical maths to viz/export_game.py, kept in sync)
# --------------------------------------------------------------------------

def build_geometry():
    cells = list(H.hex_board(RADIUS))
    cidx = {c: i for i, c in enumerate(cells)}

    def center(q, r):
        return (L * q + 0.5 * L * r, -math.sqrt(3) / 2 * L * r)

    D = []
    for i in range(6):
        ang = math.radians(60 * i)
        D.append((L * math.cos(ang), L * math.sin(ang)))

    c0 = center(0, 0)
    remap = []
    for i in range(6):
        nb = H.neighbor((0, 0), i)
        nc = center(*nb)
        v = (nc[0] - c0[0], nc[1] - c0[1])
        best = min(range(6), key=lambda k: (D[k][0] - v[0]) ** 2 + (D[k][1] - v[1]) ** 2)
        remap.append(best)

    centers = [center(q, r) for (q, r) in cells]
    xs = [c[0] for c in centers]
    ys = [-c[1] for c in centers]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    pad = L

    def to_screen(c):
        return [c[0] - minx + pad, -c[1] - miny + pad]

    # neighbor_idx[cell][edge] -> neighbor cell index, or -1 off-board
    neighbor_idx = []
    for c in cells:
        row = []
        for e in range(6):
            nb = H.neighbor(c, e)
            row.append(cidx.get(nb, -1))
        neighbor_idx.append(row)

    return {
        "L": L,
        "radius": RADIUS,
        "n_cells": len(cells),
        "W": maxx - minx + 2 * pad,
        "H": maxy - miny + 2 * pad,
        "cells": [{"idx": i, "qr": list(cells[i]), "center": to_screen(centers[i])}
                  for i in range(len(cells))],
        "edge_off": [[D[remap[i]][0] * 0.5, -D[remap[i]][1] * 0.5] for i in range(6)],
        "neighbor_idx": neighbor_idx,
    }


def build_tile_tables():
    """arcs[tile_type][rotation] -> list of 3 [edge_a, edge_b] pairs, plus the
    distinct-rotation class map (which raw rotations look identical)."""
    tiles = H.canonical_tiles()
    out = {}
    for t in range(len(tiles)):
        m = tiles[t]["matching"]
        per_rot = []
        seen = {}
        rot_class = []
        for rot in range(6):
            arcs = [list(pair) for pair in H.tile_arcs(m, rot)]
            per_rot.append(arcs)
            key = frozenset(tuple(sorted(p)) for p in arcs)
            if key not in seen:
                seen[key] = rot
            rot_class.append(seen[key])
        out[str(t)] = {
            "spans": list(tiles[t]["spans"]),
            "orbit": tiles[t]["orbit"],
            "arcs_by_rotation": per_rot,
            # rot_class[r] = the lowest rotation producing an identical arc set
            "rotation_class_rep": rot_class,
            "distinct_rotations": sorted(set(rot_class)),
        }
    return out


# --------------------------------------------------------------------------
# game generation
# --------------------------------------------------------------------------

CELLS = list(H.hex_board(RADIUS))
CIDX = {c: i for i, c in enumerate(CELLS)}
TILES = H.canonical_tiles()


def make_deck(rng, deck_counts):
    deck = []
    for tile_type, n in deck_counts.items():
        deck.extend([int(tile_type)] * n)
    rng.shuffle(deck)
    return deck


def legal_cell_indices(board, free_placement):
    placed = set(board.placed)
    if free_placement:
        return [i for i, c in enumerate(CELLS) if c not in placed]
    if not placed:
        return list(range(len(CELLS)))
    out = []
    for i, c in enumerate(CELLS):
        if c in placed:
            continue
        if any(H.neighbor(c, e) in placed for e in range(6)):
            out.append(i)
    return out


def greedy_move(board, hand, rng, free_placement):
    """Stage-2 greedy: max immediate area; ties -> first found in
    (hand_slot, cell, rotation) order; no positive move -> random legal."""
    legal = legal_cell_indices(board, free_placement)
    best, best_val = None, -1
    for hi, t in enumerate(hand):
        for ci in legal:
            for rot in range(6):
                recs, undo = board.try_place_and_get_new_loops(
                    CELLS[ci], TILES[t]["matching"], rot, H.enclosed_cells)
                val = sum(r["area"] for r in recs)
                undo()
                if val > best_val:
                    best_val = val
                    best = (hi, ci, rot)
    if best is None or best_val <= 0:
        return (rng.randrange(len(hand)), rng.choice(legal), rng.randrange(6))
    return best


def random_move(board, hand, rng, free_placement):
    legal = legal_cell_indices(board, free_placement)
    return (rng.randrange(len(hand)), rng.choice(legal), rng.randrange(6))


def loops_snapshot(board, loop_owner, player):
    loops = []
    for lp in board.loops():
        enc = H.enclosed_cells(board, lp)
        arc_edges = []
        for aid in lp["arcs"]:
            cell, ea, eb = board.arcs[aid]
            arc_edges.append([CIDX[cell], ea, eb])
        key = frozenset((CIDX[board.arcs[aid][0]],
                         min(board.arcs[aid][1], board.arcs[aid][2]),
                         max(board.arcs[aid][1], board.arcs[aid][2]))
                        for aid in lp["arcs"])
        owner = loop_owner.setdefault(key, player)
        loops.append({
            "arcs": sorted(arc_edges),
            "cells": sorted(CIDX[c] for c in enc),
            "area": len(enc),
            "length": lp["length"],
            "owner": owner,
        })
    loops.sort(key=lambda d: (d["arcs"][0] if d["arcs"] else [], d["area"]))
    return loops


def play_game(seed, policy="greedy", free_placement=False,
              deck_counts=None, hand_size=3, n_players=2):
    deck_counts = deck_counts or {"0": 12, "2": 25}
    total = sum(deck_counts.values())
    assert total == len(CELLS), f"deck must sum to {len(CELLS)}, got {total}"

    rng = random.Random(seed)
    deck = make_deck(rng, deck_counts)
    deck_initial = list(deck)

    hands = [[] for _ in range(n_players)]
    for p in range(n_players):
        for _ in range(hand_size):
            hands[p].append(deck.pop(0))

    board = H.Board(CELLS)
    scores = [0] * n_players
    loop_owner = {}
    steps = []

    move_fn = greedy_move if policy == "greedy" else random_move

    for t in range(len(CELLS)):
        p = t % n_players
        hi, ci, rot = move_fn(board, hands[p], rng, free_placement)
        tile = hands[p][hi]

        recs, undo = board.try_place_and_get_new_loops(
            CELLS[ci], TILES[tile]["matching"], rot, H.enclosed_cells)
        gained = sum(r["area"] for r in recs)
        undo()
        board.place(CELLS[ci], TILES[tile]["matching"], rot)
        scores[p] += gained

        arcs = [list(pair) for pair in H.tile_arcs(TILES[tile]["matching"], rot)]

        # hand update: remove played tile (preserving order), refill from deck
        hands[p].pop(hi)
        if deck:
            hands[p].append(deck.pop(0))

        steps.append({
            "t": t,
            "player": p,
            "hand_slot": hi,
            "cell": ci,
            "tile": tile,
            "rot": rot,
            "arcs": arcs,
            "gained": gained,
            "score": list(scores),
            "hands_after": [list(h) for h in hands],
            "loops": loops_snapshot(board, loop_owner, p),
        })

    return {
        "config": {
            "policy": policy,
            "free_placement": free_placement,
            "deck_counts": deck_counts,
            "hand_size": hand_size,
            "n_players": n_players,
            "radius": RADIUS,
            "scorer": "area_linear",
        },
        "seed": seed,
        "deck_initial": deck_initial,
        "steps": steps,
        "final_score": scores,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hextruchet-godot/data/packs")
    ap.add_argument("--games", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    geometry = build_geometry()
    tiles = build_tile_tables()
    with open(os.path.join(args.out, "static.json"), "w") as f:
        json.dump({"geometry": geometry, "tiles": tiles}, f)
    print(f"wrote static.json ({geometry['n_cells']} cells, {len(tiles)} tile types)")

    # Cover the rule-option matrix so the differential test exercises the
    # config paths, not just the spec default.
    variants = [
        ("greedy", False, {"0": 12, "2": 25}),
        ("random", False, {"0": 12, "2": 25}),
        ("greedy", True, {"0": 12, "2": 25}),      # free placement
        ("random", True, {"0": 12, "2": 25}),
        ("random", False, {"0": 8, "1": 7, "2": 8, "3": 7, "4": 7}),  # 5-tile
        ("greedy", False, {"0": 8, "1": 7, "2": 8, "3": 7, "4": 7}),
    ]

    manifest = []
    for i in range(args.games):
        policy, free, counts = variants[i % len(variants)]
        seed = 1000 + i
        pack = play_game(seed, policy=policy, free_placement=free, deck_counts=counts)
        name = f"pack_{i:02d}_{policy}{'_free' if free else ''}{'_5tile' if len(counts) > 2 else ''}.json"
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(pack, f)
        manifest.append({
            "file": name,
            "policy": policy,
            "free_placement": free,
            "tile_types": sorted(int(k) for k in counts),
            "seed": seed,
            "final_score": pack["final_score"],
            "n_loops_final": len(pack["steps"][-1]["loops"]),
        })
        print(f"  {name}: score {pack['final_score']}, "
              f"{len(pack['steps'][-1]['loops'])} loops")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"packs": manifest}, f, indent=2)
    print(f"wrote manifest.json ({len(manifest)} packs) -> {args.out}")


if __name__ == "__main__":
    main()
