# hex_truchet — environment spec

Single source of truth. Both `reference.py` and `fast.py` are written from
this document — never from each other. Every rule below must be traceable in
both implementations.

**Scope.** This package specs the **private-hand** variant: 2 players, the
`hand` drafting mechanism (hold `h=3` tiles, play one, refill), placement
restricted to `adjacency_required=True`, scorer `area_linear`, on a fixed
radius-3 hex board (37 cells). This is Stage 3 of a larger game-design
research project — see `/Users/mwburke/projects/hextruchet/hextruchet_first_pass/HANDOFF.md`
§7.3 for why: Stage 2 screening found a hand-crafted "denial" agent
(greedy + 1-ply opponent-lookahead) statistically indistinguishable from
plain greedy in head-to-head play, despite provably choosing different
moves. This environment exists to train a self-play agent that can settle
whether that's a shallow strategic ceiling or just weak hand-crafted
heuristics. A **public-hand** sibling variant is specced separately in
`../hex_truchet_public/spec.md` (identical in every respect except
observation — see that file) and will be built once this one is validated
end-to-end.

**Self-play framing.** This is a 2-player turn-based game modeled as a
single-agent-per-step environment: `state.current_player` says whose turn it
is, `observe(state)` returns that player's view (so the same policy network
plays both seats symmetrically across an episode), and `step()`'s reward is
attributed to whichever player just acted. See Rewards for the resulting
asymmetry this creates in the raw per-step reward stream, and why fixing it
is a training-script concern, not an environment one.

**Constants** (all fixed for this package — no runtime configuration; see
"why one behavior per package" in the session notes this spec was written
from):
`RADIUS = 3`, `N_CELLS = 37`, `N_PLAYERS = 2`, `HAND_SIZE = 3`,
`N_ROTATIONS = 6`, `DECK_TYPE0_COUNT = 12`, `DECK_TYPE2_COUNT = 25`
(`DECK_TYPE0_COUNT + DECK_TYPE2_COUNT = N_CELLS = 37`, matching Stage 0's
established winning deck ratio, tile 0 : tile 2 = 1:2, from
`stage0.py`/`stage2_screen.py::make_deck`), `SCORE_NORM = 100.0`,
`ACTION_SPACE_SIZE = HAND_SIZE * N_CELLS * N_ROTATIONS = 666`.

`CELLS = hex_board(RADIUS)` from the existing, stable `geometry.py` — a
fixed, deterministic ordering (37 axial `(q, r)` tuples); state array index
`i` always refers to `CELLS[i]`. Tile types reference `canonical_tiles()`
from the same module: **tile type `0`** is the `(1,1,1)` matching (orbit
size 2 — only 2 of its 6 rotations are visually distinct), **tile type `2`**
is the `(1,1,3)` matching (orbit size 3 — 3 of 6 rotations distinct). Both
are stable, already-tested Stage 0 code; this spec does not redefine them,
only references them by index. Rotation aliasing (multiple raw rotation
values producing the identical resulting arc pattern for these two
tile types) is preserved as-is — it already existed in the screened Stage 2
engine (`agents.py`/`stage2_screen.py` both loop `for rot in range(6)`
unconditionally) and isn't something this port changes.

## State space

| field | dtype | bounds | meaning |
|---|---|---|---|
| `t` | int64 | `[0, 37]` | in-episode step counter = number of tiles placed so far. RNG key. |
| `current_player` | int8 | `[0, 1]` | whose turn it is to act next |
| `board_tile` | int8[37] | `{-1, 0, 2}` | tile type at `CELLS[i]`; `-1` = empty |
| `board_rotation` | int8[37] | `[0, 5]` | rotation at `CELLS[i]`; `0` (canonical unused value) when `board_tile[i] == -1` |
| `hand_p0` | int8[3] | `{-1, 0, 2}` | player 0's held tile types, **always left-packed** (occupied slots first, `-1` padding after); `-1` = empty slot |
| `hand_p1` | int8[3] | `{-1, 0, 2}` | player 1's held tile types, same left-packed convention |
| `score_p0` | int32 | `[0, 500]` | player 0's cumulative `area_linear` score (loose upper bound; observed Stage 2 maxima on this board were ~30) |
| `score_p1` | int32 | `[0, 500]` | player 1's cumulative score |

Serialized form: `schema.json` `$defs/state`, mirroring this table exactly.
`deck_remaining_count` (how many tiles are left undrawn) is deliberately
**not** a state field — see RNG slots below for why it's fully derivable
from `t` and doesn't need to be tracked or stored.

## Actions

Integer in `[0, 666)`, decoding a `(hand_slot, cell_idx, rotation)` triple:

```
hand_slot = action // (N_CELLS * N_ROTATIONS)      # in [0, 3)
rem       = action %  (N_CELLS * N_ROTATIONS)
cell_idx  = rem // N_ROTATIONS                       # in [0, 37)
rotation  = rem %  N_ROTATIONS                       # in [0, 6)
```

`hand_slot` indexes into `current_player`'s hand (`hand_p0` or `hand_p1`,
whichever is `current_player`). `cell_idx` indexes into `CELLS`.

**Legality.** An action is legal iff both hold:
1. `hand[current_player][hand_slot] != -1` (the slot is occupied).
2. `CELLS[cell_idx]` is a legal placement cell under `adjacency_required`:
   if the board is empty (`t == 0`), every cell is legal; otherwise the cell
   must be currently empty (`board_tile[cell_idx] == -1`) AND adjacent to at
   least one occupied cell (reusing `geometry.neighbor`/existing
   `legal_cells_adjacent` semantics — unchanged from the screened Stage 2
   engine, `agents.py`).

`rotation` never affects legality (matches the existing engine: legal-cell
functions don't filter by rotation).

**Illegal actions.** The action space is large (666) but usually only a
handful of the decoded triples are legal, especially early game — so the
observation includes a `legal_action_mask` (see Observations) and a
well-behaved policy should never submit an illegal action. But
`simulacrum`'s validation battery deliberately submits random (frequently
illegal) actions, so `step()` must define illegal-action behavior precisely
and deterministically: **an illegal action is silently replaced by the
smallest legal action index** (`argmin` over the true/false
`legal_action_mask`, i.e. scan `action = 0, 1, 2, ...` and take the first
legal one) before the transition proceeds. This is a pure function of state
(no RNG involved — doesn't consume an RNG slot), so it doesn't affect
reference/fast RNG parity. `legal_action_mask` is never all-false when
`t < 37` — see Termination for why a legal action always exists.

## Observations

`observe(state)` returns the view for `state.current_player` (so the same
policy plays both seats across an episode via self-play). All fields below
are computed **in the order listed**, cast to `float32` at each division
(matching the toywalk convention: cast the int to float32, then divide by
the float32 constant — this ordering is what makes reference/fast
bit-identical), and concatenated into one flat `float32` array of length
`37 + 37 + 3 + 3 + 1 + 1 + 1 + 666 = 749`:

1. **`board_tile_norm`** (37): `float32(board_tile[i]) / float32(4)` for
   `i in 0..36`, in `CELLS` order. (`4` is `TILES` index range, not a tuned
   constant — only `-1, 0, 2` ever occur given the fixed deck, giving
   values `{-0.25, 0.0, 0.5}`.)
2. **`board_rotation_norm`** (37): `float32(board_rotation[i]) / float32(5)`.
3. **`my_hand_norm`** (3): `float32(hand[current_player][j]) / float32(4)`
   for `j in 0..2` — the acting player's own hand, always true values
   (never masked — you always see your own tiles).
4. **`opponent_hand_norm`** (3): for `j in 0..2`, let `v =
   hand[1 - current_player][j]`. **Private-hand masking (this package):**
   if `v == -1` (slot genuinely empty — hand size is public information,
   physically countable even with hidden tile identities) emit
   `float32(-1) / float32(4)`; otherwise (slot occupied, tile type hidden)
   emit a fixed `HIDDEN` sentinel, `float32(-2) / float32(4) = -0.5`,
   **regardless of the tile's true type**. This is the only place this
   package's observation differs from `../hex_truchet_public/spec.md`.
5. **`my_score_norm`** (1): `float32(score[current_player]) /
   float32(SCORE_NORM)`.
6. **`opponent_score_norm`** (1): `float32(score[1 - current_player]) /
   float32(SCORE_NORM)`.
7. **`t_norm`** (1): `float32(t) / float32(N_CELLS)`.
8. **`legal_action_mask`** (666): `float32(1.0)` where the action is legal
   per the Actions section, else `float32(0.0)`, in raw action-index order.
   Computed only from public board state + the acting player's own hand —
   safe to expose regardless of hand-visibility mode, never leaks opponent
   hand contents.

Note: `deck_remaining_count` is deliberately **not** included as an
observation field (unlike `t_norm`, which is). Under private-hand masking,
the exact remaining-tile-count *by type* is not knowable to an observer (the
opponent's hidden hand slots could be either type), so exposing it would
either leak hidden information or require an inexact/estimated field — see
RNG slots. The plain *total* remaining count is knowable but is already
recoverable from `t` (`N_CELLS - 6 - min(t, 31)`, worked out below), so
including it as its own field would violate "anything derivable doesn't
belong."

## Rewards

`0.0` at every non-terminal step. At the terminal step (the step where `t`
reaches `37`): `reward = float(score[p] - score[1 - p])`, where `p` is the
**pre-step** `current_player` — i.e. the player who made the 37th and final
placement. Exact integer arithmetic promoted to float at the end; no
computation that could diverge between scalar and vectorized
implementations, so no `x-atol` is needed for this or any other field.

**Known asymmetry, deliberately left to the training script.** Only the
player who happens to place the 37th tile receives a nonzero reward from
`step()` directly — the other player's own last action (the 36th placement)
occurs at a non-terminal step and gets `reward = 0.0`, even though the same
final-margin outcome is just as much a consequence of their play. This
environment defines `step()`'s literal per-call reward only; a self-play
training script is expected to construct each player's own trajectory from
the shared episode and resolve the terminal margin onto **both** players'
final action (a standard post-processing step for turn-based self-play, not
an environment-spec concern — the environment's job is to simulate
transitions correctly, not to pre-solve credit assignment).

## Termination

`terminated = (t == 37)`. This game **always** terminates in exactly 37
steps regardless of actions (illegal actions are redirected to a legal
fallback, never a no-op — see Actions) or RNG outcomes: the board has
exactly 37 cells, one tile is placed per step, and the hex region is fully
connected so a legal cell always exists for a non-full board (the
frontier — cells adjacent to an already-placed tile — is always non-empty
when `0 < t < 37`, and at `t == 0` every cell is legal by definition).
`current_player`'s hand is also guaranteed non-empty whenever `t < 37` — see
the worked-out draw schedule under RNG slots. So termination is a fixed
horizon, not just "bounded" — a stronger guarantee than the framework's
minimum requirement, and it makes this environment simple for the
determinism/auto-reset battery.

## Reset

`t = 0`, `current_player = 0`, `score_p0 = score_p1 = 0`, `board_tile` all
`-1`, `board_rotation` all `0`. Both hands are dealt to `HAND_SIZE = 3` via
6 sequential draws (see RNG slots, slot `INITIAL_DEAL`, indices `0..5`) in
fixed order: player 0 slot 0, slot 1, slot 2, then player 1 slot 0, slot 1,
slot 2 — each draw immediately written into the corresponding hand slot
before the next draw's probability is computed (see RNG slots for why this
ordering matters).

## Invariants

1. `0 <= t <= 37`.
2. `current_player in {0, 1}`.
3. `current_player == t % 2` (strict alternation — no stalls are possible
   on this board, see Termination).
4. Exactly `t` entries of `board_tile` are `!= -1`, and `37 - t` are `== -1`
   (occupied-cell count matches the step counter exactly).
5. `board_tile[i] in {-1, 0, 2}` for all `i` (only the two fixed deck tile
   types ever appear).
6. `board_rotation[i] == 0` whenever `board_tile[i] == -1` (canonical unused
   value — no undefined states).
7. **Tile conservation (total):** `count(board_tile != -1) +
   count(hand_p0 != -1) + count(hand_p1 != -1) + deck_remaining_count(t) ==
   37` at all times, where `deck_remaining_count(t) = 37 - 6 - min(t, 31)`
   (derived, not stored — see RNG slots).
8. **Tile conservation (by type):** `count(board_tile == 0) +
   count(hand_p0 == 0) + count(hand_p1 == 0) + deck_remaining_type0(t) ==
   12`, and the same equation with `2`/`25` in place of `0`/`12`. Catches
   any bug that draws from the wrong type distribution.
9. `hand_p0` and `hand_p1` are each left-packed: occupied slots (`!= -1`)
   form a contiguous prefix, `-1` padding only after it. No internal gaps.
10. `score_p0 >= 0` and `score_p1 >= 0` (areas are always non-negative).
11. `t == 0` implies: `board_tile` is all `-1`, both hands have exactly 3
    occupied slots, both scores are `0`, `current_player == 0`.
12. `terminated` (`t == 37`) implies `board_tile` has no `-1` entries and
    both hands are fully empty (all slots `-1`) — validates that the
    tile-count bookkeeping (37 = 6 dealt + 31 drawn, exhausting exactly when
    the board fills) is correct, not just that the step counter reached 37.

Each becomes an `@invariant` on the batched env, checked batch-wide every
step when `debug=True`.

## RNG slots

Every tile draw (whether during the initial deal or a later hand refill) is
modeled as a **live sequential draw without replacement** from the known
fixed multiset `{12 × type 0, 25 × type 2}`, rather than pre-shuffling and
storing a 37-length deck order in state. This is distributionally identical
to "shuffle once, deal from the front" (a standard fact about sampling
without replacement) but avoids ever storing a full permutation array in
state, and fits the framework's per-step RNG-keying convention much more
naturally than a single big step-0 shuffle would.

At each draw, let `type0_drawn` = total type-0 draws resolved so far this
episode (dealt into a hand, later placed or still held — doesn't matter
which), and `total_drawn` = total draws resolved so far (any type). Both are
derivable at the moment of the draw as `count(board_tile == 0) +
count(hand_p0 == 0) + count(hand_p1 == 0)` (type0_drawn) and the
`!= -1` analogue (total_drawn), **as long as state is updated immediately
after each draw, before the next draw's probability is computed** — true
both within the 6-draw initial deal (each of the 6 draws writes into its
hand slot before the next draw happens) and across the game's later
single-draw-per-step refills. The draw's outcome:

```
remaining_type0 = DECK_TYPE0_COUNT - type0_drawn
remaining_total = (DECK_TYPE0_COUNT + DECK_TYPE2_COUNT) - total_drawn
draw_is_type0 = rng.draw_bernoulli(key, step, slot, index,
                                    p = remaining_type0 / remaining_total)
# tile type drawn = 0 if draw_is_type0 else 2
```

`deck_remaining_count(t)`, used in invariant 7 and deliberately excluded
from state and observations, is a closed-form function of `t` alone (no RNG
state needed to compute it): `37 - 6 - min(t, 31)`. This holds because —
worked out from the mechanics below — exactly `31` refill draws occur
total, one per step for steps `t = 0` through `t = 30` (`k < N_CELLS -
N_PLAYERS*HAND_SIZE = 31`), and zero for `t = 31` through `36`.

| slot | name | used at | distribution |
|---|---|---|---|
| 0 | `INITIAL_DEAL` | reset (step 0), index `0..5` | `Bernoulli(p = remaining_type0/remaining_total)`, sequential — see above. Index order: p0 slot0, p0 slot1, p0 slot2, p1 slot0, p1 slot1, p1 slot2. |
| 1 | `DECK_DRAW` | every step `t = k` for `k in 0..30`, index `0` | Same Bernoulli rule, drawn once per such step (no draw at all for `k in 31..36` — the deck is exhausted, `step()` skips this slot entirely that step, consuming no randomness) |

`INITIAL_DEAL` and `DECK_DRAW` are deliberately separate slots even though
both draw at step `0` (the reset deal happens "at" step 0, and the first
`step()` call's possible refill draw, for `k=0`, is also keyed at step 0) —
same step, different slot, no collision, per the framework's RNG contract.

The full per-step transition, precisely, given pre-step state
`(t=k, current_player=p, ...)` and a (possibly-clamped-to-legal) action
decoding to `(hand_slot, cell_idx, rotation)`:

1. `tile_type = hand[p][hand_slot]` (guaranteed `!= -1` after clamping).
2. Place: `board_tile[cell_idx] = tile_type`, `board_rotation[cell_idx] =
   rotation`. Compute newly-closed loops from this placement using the
   existing, tested `graph.py` `Board`/union-find loop-closure logic
   (unchanged — this port reuses that algorithm's *rules*, not its Python
   object, since `fast.py` needs a masked-tensor version of the same
   logic). `score[p] += area_linear(newly_closed_loop_records)` (existing
   `stage0.enclosed_cells` area rule, unchanged).
3. Remove the played tile from the hand, **preserving left-packedness**:
   shift `hand[p][hand_slot+1 : sz]` left by one (where `sz` is the current
   occupied-slot count), so the gap at the old position `sz-1` opens up.
4. If `k < 31`: draw one tile via `DECK_DRAW` (index `0`, keyed at step
   `k`) and write it into `hand[p][sz-1]` (the slot that just opened at
   step 3), restoring `sz` occupied slots. If `k >= 31`: leave
   `hand[p][sz-1] = -1` (occupied count drops to `sz-1`; this is exactly
   the last-few-turns hand-drain the existing reference engine exhibits).
5. `current_player = 1 - p`.
6. `t = k + 1`.
7. `reward = 0.0` if `t < 37`, else `float(score[p] - score[1-p])` (`p` is
   the actor from this step, i.e. the value *before* step 5's flip).
8. `terminated = (t == 37)`.
