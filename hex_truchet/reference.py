"""Readable single-instance reference implementation of hex_truchet.

Written from spec.md ONLY. Style: dataclass state, explicit ifs, no
vectorization, no premature abstraction, every rule traceable to a spec line.
All randomness via simulacrum.rng scalar draws with slots from hex_truchet.Slots.

Loop-closure and enclosed-area computation is delegated to the vendored,
unit-tested Stage 0-2 core (`_hexcore`) rather than reimplemented here — that
algorithm had subtle bugs historically (HANDOFF.md 4.1) and is now correct and
tested, so the reference leans on it deliberately. fast.py does NOT reuse
_hexcore; it reimplements loop closure as tensor ops from spec.md, and the
differential test validates the two against each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulacrum import ReferenceEnv, rng

from hex_truchet import (
    ACTION_SPACE_SIZE, DECK_TYPE0_COUNT, DECK_SIZE, HAND_SIZE, HIDDEN_SENTINEL,
    N_CELLS, N_REFILL_STEPS, N_ROTATIONS, ROT_NORM, SCORE_NORM, Slots,
    TILE_NORM,
)
from hex_truchet import _hexcore

# --- fixed board/tile data (deterministic; spec.md #Constants) ---------------
# CELLS is a fixed ordering of the 37 axial coords; state array index i always
# refers to CELLS[i]. TILES indexes canonical_tiles(); only types 0 and 2 occur.
CELLS = tuple(_hexcore.hex_board(3))
CELLS_SET = set(CELLS)
CELL_INDEX = {cell: i for i, cell in enumerate(CELLS)}
TILES = _hexcore.canonical_tiles()

# Per-cell on-board neighbor cell indices (for adjacency-required legality).
_NEIGHBOR_IDX = tuple(
    tuple(
        CELL_INDEX[nb]
        for nb in (_hexcore.neighbor(cell, e) for e in range(6))
        if nb in CELL_INDEX
    )
    for cell in CELLS
)


@dataclass
class State:
    t: int                    # spec: State space — tiles placed so far (RNG key)
    current_player: int       # spec: State space — whose turn (0 or 1)
    board_tile: list          # spec: State space — int8[37], {-1,0,2} at CELLS[i]
    board_rotation: list      # spec: State space — int8[37], [0,5]
    hand_p0: list             # spec: State space — int8[3], left-packed, {-1,0,2}
    hand_p1: list             # spec: State space — int8[3], left-packed
    score_p0: int             # spec: State space — cumulative area_linear score
    score_p1: int


# --- action encoding (spec.md #Actions) --------------------------------------

def _decode_action(action):
    """action in [0, 666) -> (hand_slot, cell_idx, rotation). spec: Actions."""
    hand_slot = action // (N_CELLS * N_ROTATIONS)
    rem = action % (N_CELLS * N_ROTATIONS)
    cell_idx = rem // N_ROTATIONS
    rotation = rem % N_ROTATIONS
    return hand_slot, cell_idx, rotation


def _hand_of(state, player):
    return state.hand_p0 if player == 0 else state.hand_p1


def _score_of(state, player):
    return state.score_p0 if player == 0 else state.score_p1


def _cell_is_legal(state, cell_idx):
    """spec: Actions — legality clause 2 (adjacency_required)."""
    if state.t == 0:
        return True                       # empty board: every cell legal
    if state.board_tile[cell_idx] != -1:
        return False                      # occupied
    for j in _NEIGHBOR_IDX[cell_idx]:     # adjacent to >=1 occupied cell
        if state.board_tile[j] != -1:
            return True
    return False


def _legal_action_mask(state):
    """spec: Observations field 8 / Actions legality — bool[666]."""
    hand = _hand_of(state, state.current_player)
    cell_legal = [_cell_is_legal(state, c) for c in range(N_CELLS)]
    mask = [False] * ACTION_SPACE_SIZE
    for a in range(ACTION_SPACE_SIZE):
        hand_slot, cell_idx, _rot = _decode_action(a)
        # spec: Actions — legal iff hand slot occupied AND cell legal
        # (rotation never affects legality).
        if hand[hand_slot] != -1 and cell_legal[cell_idx]:
            mask[a] = True
    return mask


def _first_legal(mask):
    """spec: Actions — illegal action redirected to smallest legal index."""
    for a in range(ACTION_SPACE_SIZE):
        if mask[a]:
            return a
    return None   # unreachable while t < 37 (spec: Termination guarantees one)


# --- deck draws (spec.md #RNG slots) -----------------------------------------

def _drawn_counts(board_tile, hand_p0, hand_p1):
    """(type0_drawn, total_drawn) — every drawn tile is conserved in board U
    hands (nothing discarded), so these are just live counts. spec: RNG slots.
    Must be called with state updated up to the moment of the draw."""
    type0 = (board_tile.count(0) + hand_p0.count(0) + hand_p1.count(0))
    total = (sum(v != -1 for v in board_tile)
             + sum(v != -1 for v in hand_p0)
             + sum(v != -1 for v in hand_p1))
    return type0, total


def _draw_tile(key, step, slot, index, board_tile, hand_p0, hand_p1):
    """One sequential Bernoulli-without-replacement draw. spec: RNG slots.

    Returns tile type 0 or 2. Uses draw_uniform(...) < p rather than
    draw_bernoulli because p varies per draw (and, in the batched port, per
    instance) — draw_bernoulli's scalar-p helper cannot express that, but
    draw_uniform(...) < p is exactly what draw_bernoulli computes internally,
    so this stays bit-identical to the batched draw_uniform_torch(...) < p."""
    type0_drawn, total_drawn = _drawn_counts(board_tile, hand_p0, hand_p1)
    remaining_type0 = DECK_TYPE0_COUNT - type0_drawn
    remaining_total = DECK_SIZE - total_drawn
    p = remaining_type0 / remaining_total
    draw_is_type0 = rng.draw_uniform(key, step, slot, index) < p
    return 0 if draw_is_type0 else 2


# --- scoring via vendored loop-closure core ----------------------------------

def _area_gained(state, cell_idx, tile_type, rotation):
    """area_linear score from placing (tile_type, rotation) at CELLS[cell_idx].

    Rebuilds a fresh _hexcore.Board from the current placed tiles, then uses
    the tested try_place_and_get_new_loops to get exactly the loops newly
    closed by THIS placement, with per-loop enclosed area. spec: Rewards /
    transition step 2. Placement order of the pre-existing tiles is irrelevant
    — loop structure and enclosed area are functions of the final geometry,
    not of arc-id assignment order.
    """
    board = _hexcore.Board(CELLS_SET)
    for i in range(N_CELLS):
        tt = state.board_tile[i]
        if tt != -1:
            board.place(CELLS[i], TILES[tt]["matching"], state.board_rotation[i])
    records, _undo = board.try_place_and_get_new_loops(
        CELLS[cell_idx], TILES[tile_type]["matching"], rotation,
        _hexcore.enclosed_cells)
    # spec: Rewards — area_linear = sum of enclosed-cell counts over new loops.
    return sum(r["area"] for r in records)


class HexTruchetReference(ReferenceEnv):
    def reset(self, seed: int, episode: int = 0) -> State:
        self.seed_episode(seed, episode)

        # spec: Reset — empty board, scores 0, current_player 0.
        board_tile = [-1] * N_CELLS
        board_rotation = [0] * N_CELLS
        hand_p0 = [-1] * HAND_SIZE
        hand_p1 = [-1] * HAND_SIZE

        # spec: Reset / RNG slots — deal 6 tiles, INITIAL_DEAL slot at step 0,
        # index 0..5, order p0s0,p0s1,p0s2,p1s0,p1s1,p1s2. Each draw is written
        # into its slot BEFORE the next draw's counts are computed.
        deal_targets = [(hand_p0, 0), (hand_p0, 1), (hand_p0, 2),
                        (hand_p1, 0), (hand_p1, 1), (hand_p1, 2)]
        for index, (hand, slot) in enumerate(deal_targets):
            tile = _draw_tile(self.key, 0, Slots.INITIAL_DEAL, index,
                              board_tile, hand_p0, hand_p1)
            hand[slot] = tile

        self.state = State(
            t=0, current_player=0,
            board_tile=board_tile, board_rotation=board_rotation,
            hand_p0=hand_p0, hand_p1=hand_p1,
            score_p0=0, score_p1=0,
        )
        return self.state

    def step(self, action: int) -> tuple[State, float, bool, dict]:
        state = self.state
        p = state.current_player
        k = state.t

        # spec: Actions — illegal action redirected to smallest legal index
        # (pure function of state, consumes no RNG).
        mask = _legal_action_mask(state)
        if not mask[action]:
            action = _first_legal(mask)
        hand_slot, cell_idx, rotation = _decode_action(action)

        hand = _hand_of(state, p)
        tile_type = hand[hand_slot]   # guaranteed != -1 after clamping

        # spec: transition step 2 — place, score newly-closed loops by area.
        gained = _area_gained(state, cell_idx, tile_type, rotation)
        board_tile = list(state.board_tile)
        board_rotation = list(state.board_rotation)
        board_tile[cell_idx] = tile_type
        board_rotation[cell_idx] = rotation
        score_p0 = state.score_p0
        score_p1 = state.score_p1
        if p == 0:
            score_p0 += gained
        else:
            score_p1 += gained

        # spec: transition step 3 — remove played tile, preserve left-packing:
        # shift the occupied suffix left by one; the last occupied slot opens up.
        hand_p0 = list(state.hand_p0)
        hand_p1 = list(state.hand_p1)
        new_hand = hand_p0 if p == 0 else hand_p1
        sz = sum(v != -1 for v in new_hand)
        for j in range(hand_slot, sz - 1):
            new_hand[j] = new_hand[j + 1]
        new_hand[sz - 1] = -1

        # spec: transition step 4 — refill from the deck for steps k in 0..30
        # (writing into the slot that just opened); no draw at all for k >= 31.
        if k < N_REFILL_STEPS:
            tile = _draw_tile(self.key, k, Slots.DECK_DRAW, 0,
                              board_tile, hand_p0, hand_p1)
            new_hand[sz - 1] = tile

        # spec: transition steps 5-8 — advance turn/counter, reward, terminate.
        current_player = 1 - p
        t = k + 1
        if t < N_CELLS:
            reward = 0.0
        else:
            reward = float(score_p0 - score_p1) if p == 0 \
                else float(score_p1 - score_p0)
        terminated = (t == N_CELLS)

        self.state = State(
            t=t, current_player=current_player,
            board_tile=board_tile, board_rotation=board_rotation,
            hand_p0=hand_p0, hand_p1=hand_p1,
            score_p0=score_p0, score_p1=score_p1,
        )
        return self.state, reward, terminated, {}

    def observe(self, state: State) -> np.ndarray:
        # spec: Observations — float32[749] for the acting player's view, each
        # division computed in float32 (cast int -> float32, then divide by the
        # float32 constant), fields concatenated in the listed order.
        cp = state.current_player
        opp = 1 - cp
        my_hand = _hand_of(state, cp)
        opp_hand = _hand_of(state, opp)
        tile_norm = np.float32(TILE_NORM)

        parts = []
        # 1. board_tile_norm (37)
        for i in range(N_CELLS):
            parts.append(np.float32(state.board_tile[i]) / tile_norm)
        # 2. board_rotation_norm (37)
        for i in range(N_CELLS):
            parts.append(np.float32(state.board_rotation[i]) / np.float32(ROT_NORM))
        # 3. my_hand_norm (3) — own tiles always shown truthfully
        for j in range(HAND_SIZE):
            parts.append(np.float32(my_hand[j]) / tile_norm)
        # 4. opponent_hand_norm (3) — private masking: empty shown, occupied
        #    replaced by the HIDDEN sentinel regardless of true type.
        for j in range(HAND_SIZE):
            v = opp_hand[j]
            if v == -1:
                parts.append(np.float32(-1) / tile_norm)
            else:
                parts.append(np.float32(HIDDEN_SENTINEL) / tile_norm)
        # 5. my_score_norm (1)
        parts.append(np.float32(_score_of(state, cp)) / np.float32(SCORE_NORM))
        # 6. opponent_score_norm (1)
        parts.append(np.float32(_score_of(state, opp)) / np.float32(SCORE_NORM))
        # 7. t_norm (1)
        parts.append(np.float32(state.t) / np.float32(N_CELLS))
        # 8. legal_action_mask (666)
        mask = _legal_action_mask(state)
        for a in range(ACTION_SPACE_SIZE):
            parts.append(np.float32(1.0) if mask[a] else np.float32(0.0))

        return np.array(parts, dtype=np.float32)

    def to_json(self, state: State) -> dict:
        return {
            "t": int(state.t),
            "current_player": int(state.current_player),
            "board_tile": [int(v) for v in state.board_tile],
            "board_rotation": [int(v) for v in state.board_rotation],
            "hand_p0": [int(v) for v in state.hand_p0],
            "hand_p1": [int(v) for v in state.hand_p1],
            "score_p0": int(state.score_p0),
            "score_p1": int(state.score_p1),
        }

    def from_json(self, obj: dict) -> State:
        return State(
            t=int(obj["t"]),
            current_player=int(obj["current_player"]),
            board_tile=[int(v) for v in obj["board_tile"]],
            board_rotation=[int(v) for v in obj["board_rotation"]],
            hand_p0=[int(v) for v in obj["hand_p0"]],
            hand_p1=[int(v) for v in obj["hand_p1"]],
            score_p0=int(obj["score_p0"]),
            score_p1=int(obj["score_p1"]),
        )
