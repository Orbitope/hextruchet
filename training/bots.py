"""Non-RL bot roster for Hex Truchet -- a tunable search bot plus named
difficulty presets, for the playable game (see viz/GODOT_GAME_PLAN.md).

All bots here are deterministic search/heuristics -- no training, no weights,
no ML runtime. That makes them trivial to port to GDScript (the Godot game
needs its own native rules engine anyway) and means difficulty is a knob, not
a checkpoint.

The single tunable is `lookahead_action(env, K, depth)`:

  K     -- candidate first-moves tried per turn, ranked by immediate area gain.
           K=1 is EXACTLY greedy. Higher K = stronger and linearly more costly.
  depth -- plies rolled out (with greedy on both sides) after the candidate
           move, then scored by margin-so-far. depth=0 rolls to game end
           (strongest, priciest). Truncating mainly saves early-game cost: a
           full rollout at t=4 is ~33 plies but only ~7 at t=30.

Difficulty therefore spans a genuine range from "plays like the Stage 2 greedy
heuristic" up to "beat greedy 50/50 with a +22.6 margin" (HANDOFF.md 8.9),
with cost-per-move as the tradeoff -- measured by `sweep_bots.py`.

NOTE on tile types: the search itself is tile-type agnostic -- it enumerates
whatever legal actions the environment reports. The 2-tile restriction comes
from the RL env's deck (`hex_truchet/fast.py`, per spec.md's fixed 12:25
deck), NOT from these bots. A 5-tile game (the engine supports all 5 canonical
tiles) works with the same code once the environment/deck allows them; only
`train_selfplay.greedy_action`'s `_ROTATION_CLASS_REP` fast-path caching is
specialized to tiles {0,2} and would need the other three added.
"""
import torch

from hex_truchet import ACTION_SPACE_SIZE
from train_selfplay import greedy_action, MASK_DIM
from lookahead_bot import lookahead_action


# --- presets: (K, depth) -- MEASURED, see training/bot_sweep.log -------------
# vs plain greedy, n=30 games each, seats rotated; s/move is Python-on-CPU
# (a pessimistic proxy for a native GDScript port):
#
#   K  depth    win    margin   s/move
#   1    --    0.467   +0.00    0.009    <- IS greedy (baseline)
#   2     4    0.900   +9.10    0.051
#   3     2    0.933  +10.20    0.030    <- best strength-per-cost
#   3     4    0.900   +9.67    0.072
#   3     8    0.900  +12.43    0.140    <- best margin in the cheap range
#   4     4    0.933  +10.23    0.156
#   8     0    1.000  +22.64    ~29      <- offline only (n=50, HANDOFF 8.9)
#
# Two things this table says, and they shape the presets below:
#  1. The jump from greedy (47%) to ANY search (>=90%) is enormous; tuning
#     beyond that moves margin, not win rate. So difficulty tiers between
#     `medium` and `expert` are separated by HOW BADLY they beat you, not by
#     whether they do.
#  2. depth is NOT where the strength is -- (3,2) beats (3,4) while costing
#     less than half. The win comes from considering several candidates, not
#     from simulating far ahead. Raise K before raising depth.
PRESETS = {
    # Instant, weakest. Uniform over legal moves -- a true beginner mode.
    "random":  None,
    # Instant. Exactly the Stage 2 greedy heuristic (K=1 short-circuits search).
    "easy":    (1, 0),
    # Best strength-per-cost: ~93% vs greedy at ~30ms/move. Default opponent.
    "medium":  (3, 2),
    # Same win rate, noticeably bigger margins, still comfortably interactive.
    "hard":    (3, 8),
    # Full strength (HANDOFF 8.9: 100% vs greedy, +22.6). ~29 s/move --
    # OFFLINE USE ONLY (showcase replay packs); do NOT wire into live play.
    "expert":  (8, 0),
}
# Difficulties safe to offer in an interactive game (excludes `expert`).
INTERACTIVE_PRESETS = ["random", "easy", "medium", "hard"]


def random_action(env, obs=None):
    if obs is None:
        obs = env.observe()
    mask = obs[:, -MASK_DIM:] > 0.5
    u = torch.rand(env.n, ACTION_SPACE_SIZE, device=env.device).masked_fill(~mask, -1)
    return u.argmax(-1)


def make_bot(difficulty="medium"):
    """Return a `fn(env_single_instance) -> action tensor[1]` for the named
    preset. The env must be a single-instance HexTruchetBatched (n=1), which is
    what interactive play uses."""
    if difficulty not in PRESETS:
        raise ValueError(f"unknown difficulty {difficulty!r}; "
                         f"choose from {sorted(PRESETS)}")
    cfg = PRESETS[difficulty]
    if cfg is None:
        return lambda env: random_action(env)
    K, depth = cfg
    if K <= 1:
        return lambda env: greedy_action(env, env.observe())  # no search needed
    return lambda env: lookahead_action(env, K=K, depth=depth)


__all__ = ["PRESETS", "INTERACTIVE_PRESETS", "make_bot", "random_action",
           "lookahead_action", "greedy_action"]
