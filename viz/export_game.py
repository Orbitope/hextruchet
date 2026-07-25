"""Generate one hex-truchet game and export geometry + trajectory + loops to
JSON for the web viewer. Uses the tested _hexcore (single-instance) directly,
with a greedy policy for both seats so the board fills with interesting loops.
Faithful to the Stage 3 rules: 2 players, hand=3, deck 12x tile0 / 25x tile2,
adjacency-required placement, area_linear scoring.
"""
import json, math, random, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hex_truchet"))
import _hexcore as H

RADIUS = 3
CELLS = list(H.hex_board(RADIUS))
CIDX = {c: i for i, c in enumerate(CELLS)}
TILES = H.canonical_tiles()
L = 60.0  # cell spacing (px)

# --- geometry: cell centers + 6 edge unit offsets (math coords, y up) --------
def center(q, r):
    return (L * q + 0.5 * L * r, -math.sqrt(3) / 2 * L * r)
D = []  # edge direction offset (edge-mid = center + 0.5*D[i])
for i in range(6):
    ang = math.radians(60 * i)
    D.append((L * math.cos(ang), L * math.sin(ang)))
# reconcile D with axial neighbor directions so edge i faces neighbor(cell,i):
# neighbor(cell,i) center - center should be ~ D[i]. Verify + remap if needed.
q0, r0 = CELLS[CIDX[(0, 0)]]
c0 = center(q0, r0)
remap = []
for i in range(6):
    nb = H.neighbor((0, 0), i)
    nc = center(*nb)
    v = (nc[0] - c0[0], nc[1] - c0[1])
    # find which D matches direction of v
    best = min(range(6), key=lambda k: (D[k][0]/L - v[0]/L)**2 + (D[k][1]/L - v[1]/L)**2)
    remap.append(best)
# remap[i] = index into D that physically points to neighbor across edge i
centers = [center(q, r) for (q, r) in CELLS]
# screen coords: flip y, shift to positive
xs = [c[0] for c in centers]; ys = [-c[1] for c in centers]
minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
pad = L
def to_screen(c):
    return [c[0] - minx + pad, -c[1] - miny + pad]
cells_out = [{"idx": i, "qr": list(CELLS[i]), "center": to_screen(centers[i])}
             for i in range(len(CELLS))]
edge_off = [[D[remap[i]][0] * 0.5, -D[remap[i]][1] * 0.5] for i in range(6)]  # screen-space, half
W = maxx - minx + 2 * pad; Hh = maxy - miny + 2 * pad

# --- game (greedy both seats, faithful rules) --------------------------------
def make_deck(rng):
    deck = [0] * 12 + [2] * 25
    rng.shuffle(deck)
    return deck

def legal_cells(board):
    if not board.placed:
        return list(range(len(CELLS)))
    placed = set(board.placed)
    out = []
    for i, c in enumerate(CELLS):
        if c in placed: continue
        if any(H.neighbor(c, e) in placed for e in range(6)):
            out.append(i)
    return out

def greedy_move(board, hand):
    legal = legal_cells(board)
    best = None; best_val = -1
    for hi, t in enumerate(hand):
        for ci in legal:
            for rot in range(6):
                recs, undo = board.try_place_and_get_new_loops(
                    CELLS[ci], TILES[t]["matching"], rot, H.enclosed_cells)
                val = sum(r["area"] for r in recs)
                undo()
                if val > best_val:
                    best_val = val; best = (hi, ci, rot)
    if best is None or best_val <= 0:
        # No loop-closing move available: fall back to a RANDOM legal placement
        # (matches the real Stage 2 greedy in stage2_screen.py, whose fallback
        # is rng.choice(legal) -- NOT lowest-index, which would make the whole
        # game fill in near cell-index order and hide the true placement order).
        hi = rng.randrange(len(hand))
        ci = rng.choice(legal)
        rot = rng.randrange(6)
        return hi, ci, rot
    return best

LOOP_OWNER = {}  # stable loop key -> player who first closed it

def loops_snapshot(board, player):
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
        owner = LOOP_OWNER.setdefault(key, player)  # first placer to close it
        loops.append({"arcs": arc_edges, "cells": [CIDX[c] for c in enc],
                      "area": len(enc), "length": lp["length"], "owner": owner})
    return loops

rng = random.Random(7)
deck = make_deck(rng)
hands = [[deck.pop() for _ in range(3)], [deck.pop() for _ in range(3)]]
board = H.Board(CELLS)
scores = [0, 0]
steps = []
for t in range(len(CELLS)):
    p = t % 2
    hi, ci, rot = greedy_move(board, hands[p])
    tile = hands[p][hi]
    recs, undo = board.try_place_and_get_new_loops(
        CELLS[ci], TILES[tile]["matching"], rot, H.enclosed_cells)
    gained = sum(r["area"] for r in recs)
    undo()
    board.place(CELLS[ci], TILES[tile]["matching"], rot)
    scores[p] += gained
    # rotated arc edge-pairs for this tile
    arcs = [list(pair) for pair in H.tile_arcs(TILES[tile]["matching"], rot)]
    hands[p].pop(hi)
    if deck: hands[p].append(deck.pop())
    steps.append({"t": t, "player": p, "cell": ci, "tile": tile, "rot": rot,
                  "arcs": arcs, "gained": gained, "score": list(scores),
                  "loops": loops_snapshot(board, p)})

out = {"L": L, "W": W, "H": Hh, "cells": cells_out, "edge_off": edge_off,
       "steps": steps, "final_score": scores}
with open(os.path.join(os.path.dirname(__file__), "game_data.json"), "w") as f:
    json.dump(out, f)
print(f"exported {len(steps)} steps, final score {scores}, "
      f"loops at end: {len(steps[-1]['loops'])}, "
      f"board {W:.0f}x{Hh:.0f}")
