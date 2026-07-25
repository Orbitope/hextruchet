"""hex_truchet: a simulacrum environment package (private-hand variant).

Stage 3 self-play environment. See spec.md for the full specification; the
constants below mirror its "Constants" block exactly.
"""

from enum import IntEnum

# --- fixed configuration (spec.md #Constants) --------------------------------
RADIUS = 3
N_CELLS = 37
N_PORTS_PER_CELL = 6           # a hex cell has 6 edges/ports
N_PLAYERS = 2
HAND_SIZE = 3
N_ROTATIONS = 6
DECK_TYPE0_COUNT = 12
DECK_TYPE2_COUNT = 25          # 12 + 25 = 37 = N_CELLS (Stage 0 deck ratio 1:2)
DECK_SIZE = DECK_TYPE0_COUNT + DECK_TYPE2_COUNT
N_REFILL_STEPS = N_CELLS - N_PLAYERS * HAND_SIZE   # = 31; refills at t in 0..30

# tile type ids: index into _hexcore.canonical_tiles(); only these two appear
TILE_TYPE_A = 0                # (1,1,1) matching
TILE_TYPE_B = 2               # (1,1,3) matching

# action encoding
ACTION_SPACE_SIZE = HAND_SIZE * N_CELLS * N_ROTATIONS   # 3 * 37 * 6 = 666

# observation normalization constants (spec.md #Observations)
TILE_NORM = 4.0               # board_tile / my_hand / opponent_hand divisor
ROT_NORM = 5.0
SCORE_NORM = 100.0
HIDDEN_SENTINEL = -2          # opponent-hand occupied-but-hidden raw value
OBS_SIZE = (N_CELLS + N_CELLS + HAND_SIZE + HAND_SIZE
            + 1 + 1 + 1 + ACTION_SPACE_SIZE)   # = 749


class Slots(IntEnum):
    """RNG slots — MUST mirror the table in spec.md."""
    INITIAL_DEAL = 0  # reset (step 0), index 0..5 -- the 6-card initial deal
    DECK_DRAW = 1     # every step t in 0..30, index 0 -- one refill draw
