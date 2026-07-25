# hex_truchet_public — environment spec

**This package is a delta on `../hex_truchet/spec.md` — read that one first.**
It is identical in every respect (state space, actions, rewards, termination,
reset, invariants, RNG slots) except **Observations**, where the opponent's
hand is fully visible instead of masked. Everything below either restates the
shared content briefly (so this file alone satisfies the spec-contract
check) or gives the one real difference in full.

**Scope and rationale.** Built alongside `hex_truchet` (not yet
implemented/validated as of this writing — see that package's history) to
let a later training run compare a self-play policy's strength under hidden
vs. fully-visible opponent hands. `hex_truchet` is the private-hand variant
and is being built first, full pipeline (reference → fast → validate →
train), since it's the more interesting POMDP case and matches how the
existing Stage 2 heuristic agents (`agents.py`/`stage2_screen.py`) already
behave — they never look at the opponent's actual hand contents. This
package exists so the public-hand variant's design is locked in now, even
though it won't be implemented until `hex_truchet` is proven out.

## State space

Identical to `hex_truchet/spec.md` — same fields, same dtypes, same bounds:
`t`, `current_player`, `board_tile`, `board_rotation`, `hand_p0`, `hand_p1`,
`score_p0`, `score_p1`. `schema.json` in this package mirrors that table
identically (see this package's `schema.json`).

## Actions

Identical to `hex_truchet/spec.md` — same `[0, 666)` flat encoding decoding
to `(hand_slot, cell_idx, rotation)`, same legality rule (occupied hand slot
+ `adjacency_required` legal cell), same deterministic smallest-legal-action
fallback for illegal submissions.

## Observations

**The one real difference.** Same flat `float32[749]` layout, same field
order, same computation for every field **except step 4**
(`opponent_hand_norm`):

4. **`opponent_hand_norm`** (3): for `j in 0..2`, let `v =
   hand[1 - current_player][j]`. **Public-hand encoding (this package):**
   emit `float32(v) / float32(4)` directly — the true tile type (or `-1` for
   a genuinely empty slot), exactly the same computation as `my_hand_norm`
   in step 3. No `HIDDEN` sentinel is ever emitted; unlike
   `../hex_truchet/spec.md`'s step 4, this package's opponent-hand encoding
   is indistinguishable in form from its own-hand encoding.

All other fields (`board_tile_norm`, `board_rotation_norm`, `my_hand_norm`,
`my_score_norm`, `opponent_score_norm`, `t_norm`, `legal_action_mask`) are
computed identically to `hex_truchet/spec.md`, in the same order, same
dtype, same normalization constants.

Because the true opponent hand is now always visible, `deck_remaining_count`
*by type* would technically be exactly derivable here (unlike the private
variant, where it isn't) — it is still deliberately **not** added as an
observation field, so the two packages' observation shapes stay identical
and directly comparable, and because it remains fully recoverable from
already-included fields (`board_tile_norm` + `my_hand_norm` +
`opponent_hand_norm`), so including it would still violate "anything
derivable doesn't belong."

## Rewards

Identical to `hex_truchet/spec.md`: `0.0` every non-terminal step; at the
terminal step, `reward = float(score[p] - score[1-p])` for the acting player
`p`. Same known asymmetry (only the 37th-placement actor gets a nonzero
`step()` reward), same resolution left to the training script.

## Termination

Identical to `hex_truchet/spec.md`: `terminated = (t == 37)`, a fixed
37-step horizon regardless of actions or RNG outcomes, for the same
board-connectivity and hand-scheduling reasons.

## Reset

Identical to `hex_truchet/spec.md`: `t=0`, `current_player=0`, both scores
`0`, empty board, both hands dealt to `HAND_SIZE=3` via the same 6-draw
`INITIAL_DEAL` sequence.

## Invariants

Identical to `hex_truchet/spec.md`'s 12 invariants (step counter bounds,
turn alternation, tile-count conservation total and by-type, left-packed
hands, non-negative scores, reset-state shape, terminal-state shape). None
of them reference hand *visibility* — they're all about the true underlying
state, which is unaffected by this package's change (only `observe()`
differs).

## RNG slots

Identical to `hex_truchet/spec.md`: `INITIAL_DEAL` (slot 0, reset, indices
`0..5`) and `DECK_DRAW` (slot 1, steps `t=0..30`, index `0`), same
sequential-Bernoulli-without-replacement mechanism, same
`deck_remaining_count(t) = 37 - 6 - min(t, 31)` closed form. Making the
opponent's hand visible in `observe()` doesn't change how tiles are drawn or
dealt — those are ground-truth-state mechanics, identical across both
packages.
