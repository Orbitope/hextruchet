"""Stage 2 agents: random, greedy, greedy+denial, shallow lookahead.

All agents share an interface: given a game state, choose (cell, rotation)
for the tile they must place. Adjacency-required vs free placement is
handled by the legal-moves function passed in, not by the agent.
"""

import random
from geometry import hex_board, canonical_tiles
from graph import Board
from stage0 import enclosed_cells

TILES = canonical_tiles()


def legal_cells_free(board, cells_all):
    return [c for c in cells_all if c not in board.placed]


def legal_cells_adjacent(board, cells_all):
    """Adjacent-to-existing-tile only, or any cell if board is empty."""
    if not board.placed:
        return list(cells_all)
    out = []
    from geometry import neighbor
    placed_set = set(board.placed)
    for c in cells_all:
        if c in placed_set:
            continue
        if any(neighbor(c, e) in placed_set for e in range(6)):
            out.append(c)
    return out


def _clone_board(board):
    """Shallow dict-copy clone instead of full replay. Replaying every
    placement to build each candidate clone was the O(cells) cost per
    candidate that made greedy search cubic overall.
    """
    clone = Board.__new__(Board)
    clone.cells = board.cells
    clone.placed = dict(board.placed)
    clone.arcs = dict(board.arcs)
    clone.port_to_arc = dict(board.port_to_arc)
    clone._next_arc = board._next_arc
    if hasattr(board, "adjacency_required"):
        clone.adjacency_required = board.adjacency_required
    return clone


def score_delta_for_move(board, cell, tile_idx, rotation, player, scorer):
    """Simulate placing (tile, rotation) at cell; return (score_gained,
    new_loop_records) without mutating the real board.
    """
    clone = _clone_board(board)
    prev_keys = {frozenset(l["arcs"]) for l in clone.loops()}
    clone.place(cell, TILES[tile_idx]["matching"], rotation)
    loops = clone.loops()
    cur = {frozenset(l["arcs"]): l for l in loops}
    new_keys = set(cur) - prev_keys
    records = []
    for k in new_keys:
        l = cur[k]
        area = len(enclosed_cells(clone, l))
        records.append({"length": l["length"], "area": area, "owner": player})
    gained = scorer(records)
    return gained, records


def best_response_value(board, cell, tile_idx, rotation, player, scorer, n_players):
    """After placing at (cell, tile, rotation), what's the best any OTHER
    player could gain on their immediate next placement anywhere legal?
    Used for the denial agent. Cheap 1-ply opponent lookahead.
    """
    clone = Board(board.cells)
    for c, (m, r) in board.placed.items():
        clone.place(c, m, r)
    clone.place(cell, TILES[tile_idx]["matching"], rotation)

    legal = legal_cells_adjacent(clone, board.cells) if board.adjacency_required \
        else legal_cells_free(clone, board.cells)
    best = 0.0
    opp = (player + 1) % n_players
    for c2 in legal[:12]:  # cap branching for speed
        for t2 in range(len(TILES)):
            for r2 in (0, 2, 4):  # sample rotations for speed
                g, _ = score_delta_for_move(clone, c2, t2, r2, opp, scorer)
                if g > best:
                    best = g
    return best


class GameBoard(Board):
    """Board subclass that remembers the adjacency rule for convenience."""
    def __init__(self, cells, adjacency_required):
        super().__init__(cells)
        self.adjacency_required = adjacency_required


def scorer_area_linear(records):
    return sum(r["area"] for r in records)


def scorer_length_plus_area(records, k=1.0):
    return sum(r["length"] + k * r["area"] for r in records)


SCORERS = {
    "area_linear": scorer_area_linear,
    "length_plus_area": scorer_length_plus_area,
}


# ---------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------

def random_agent(board, hand, player, scorer, n_players, rng):
    legal = legal_cells_adjacent(board, board.cells) if board.adjacency_required \
        else legal_cells_free(board, board.cells)
    cell = rng.choice(legal)
    tile_idx = rng.choice(hand)
    rotation = rng.randrange(6)
    return cell, tile_idx, rotation


def greedy_agent(board, hand, player, scorer, n_players, rng):
    legal = legal_cells_adjacent(board, board.cells) if board.adjacency_required \
        else legal_cells_free(board, board.cells)
    best_move, best_val = None, -1
    for c in legal:
        for t in hand:
            for rot in range(6):
                g, _ = score_delta_for_move(board, c, t, rot, player, scorer)
                if g > best_val:
                    best_val = g
                    best_move = (c, t, rot)
    if best_move is None or best_val <= 0:
        return random_agent(board, hand, player, scorer, n_players, rng)
    return best_move


def denial_agent(board, hand, player, scorer, n_players, rng):
    """Maximize own gain minus best opponent response next turn."""
    legal = legal_cells_adjacent(board, board.cells) if board.adjacency_required \
        else legal_cells_free(board, board.cells)
    # Speed: pre-filter to top-K greedy candidates by own gain, then
    # evaluate denial only on those (full denial search is too slow).
    candidates = []
    for c in legal:
        for t in hand:
            for rot in (0, 1, 2, 3, 4, 5):
                g, _ = score_delta_for_move(board, c, t, rot, player, scorer)
                candidates.append((g, c, t, rot))
    candidates.sort(reverse=True, key=lambda x: x[0])
    top = candidates[:8]
    best_move, best_val = None, -1e9
    for g, c, t, rot in top:
        opp_best = best_response_value(board, c, t, rot, player, scorer, n_players)
        val = g - 0.5 * opp_best
        if val > best_val:
            best_val = val
            best_move = (c, t, rot)
    if best_move is None:
        return random_agent(board, hand, player, scorer, n_players, rng)
    return best_move


AGENTS = {
    "random": random_agent,
    "greedy": greedy_agent,
    "denial": denial_agent,
}
