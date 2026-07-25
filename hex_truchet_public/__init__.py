"""hex_truchet_public: a simulacrum environment package."""

from enum import IntEnum


class Slots(IntEnum):
    """RNG slots — MUST mirror the table in spec.md."""
    INITIAL_DEAL = 0  # reset (step 0), index 0..5 -- the 6-card initial deal
    DECK_DRAW = 1     # every step t in 0..30, index 0 -- one refill draw
