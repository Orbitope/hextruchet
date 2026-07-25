# Hex Truchet — Godot Game Implementation Plan (Stage 4)

Plan for a **playable** Hex Truchet game in Godot, with three modes:

1. **Play (hot-seat)** — two humans alternate on one screen.
2. **Play vs Bot** — human vs a bot: `greedy` (the Stage 2 heuristic) or
   `policy` (the trained Stage 3 neural net).
3. **Replay** — load a recorded bot-vs-bot game (a trajectory pack) and scrub
   it, exactly like the current web viewer.

This supersedes the earlier viewer-only plan. The key difference from a viewer:
Godot must implement the **game rules natively** (legality, loop detection,
scoring, deck/hand), not just draw a pre-computed trajectory. The rendering
work (hex geometry, the tangent-continuous arc math) carries over unchanged
from the validated web viewer (`viz/viewer.html` / `build_viewer.py`).

---

## 0. Engine version

**Godot 4.x, latest stable (4.5 / 4.6 line), GDScript.** Godot 4 has no
formally-branded LTS today (unlike 3.5) — "latest stable 4.x" is the real
target; pin the exact build in `project.godot` → `config/features`. Nothing
here needs 4.6-only APIs (4.3+ is fine). GDScript over C# to keep the toolchain
dependency-free and web-export-friendly.

---

## 1. The rules engine is the heart of this — and it must match the spec

The single source of truth is `hex_truchet/spec.md` (the Stage 3 env spec).
Port it to GDScript as a pure, UI-free `GameState` — not from the Python code,
from the spec, so the port is an independent implementation we can
differentially test (same discipline that validated `reference.py` vs
`fast.py`).

**The Godot game is deliberately MORE general than the RL env.** `spec.md`
locks one fixed configuration (2 tile types, adjacency-required, 2 players)
because simulacrum bakes one behavior per package. The playable game should be
configurable — see §1.2. Where the game is run in the spec's exact
configuration, it must match the spec bit-for-bit (that's what the
differential test in §1.3 checks); the extra options are supersets, not
deviations.

`GameState.gd` (Resource or RefCounted), fully headless:

```
config:                                # NEW -- see 1.2
  tile_types   : Array[int]            # e.g. [0,2] (spec deck) or all 5
  deck_counts  : Array[int]            # copies of each type; must sum to 37
  free_placement : bool                # false = adjacency-required (spec), true = anywhere
  n_players    : int                   # 2 (3 is a later stretch)
state:
  board_tile   : PackedInt32Array (37)   # -1 empty, else a tile-type id
  board_rot    : PackedInt32Array (37)   # 0..5
  hands        : [PackedInt32Array(3), ...]  # left-packed, -1 pad
  scores       : [int, ...]
  current_player, t
  deck         : PackedInt32Array        # a real shuffled deck (see below)
api:
  legal_actions() -> Array           # (hand_slot, cell, rotation) triples, or a mask
  legal_cells() -> PackedInt32Array  # adjacency frontier, OR all empty cells if free_placement
  apply(hand_slot, cell, rotation) -> int   # returns area gained; mutates state
  loops() -> Array[Loop]             # {arc_edges, enclosed_cells, area, length}
  clone() -> GameState               # for bot lookahead
  is_terminal() -> bool              # t == 37
```

**Deck:** for interactive play use a real shuffled deck array (simpler than the
spec's sequential-Bernoulli-without-replacement draw, and distributionally
identical). Only replaying an existing pack must reproduce exact draws; a fresh
game just needs a valid deck matching `deck_counts`. Store the RNG seed in the
pack for reproducibility.

### 1.2 Configurable rules (both are game modes, not spec violations)

**(a) Free placement.** `free_placement = true` makes every empty cell legal
instead of only the adjacency frontier. This is not new logic to invent — it's
the `legal_cells_free` rule that already exists in `agents.py` and was screened
in Stage 2 (HANDOFF.md §7). It changes the game's feel a lot (you can start
loops anywhere, so the board develops in disconnected clusters), and it makes
the early game much wider, so it's worth offering as a mode toggle. Only two
things care about it: `legal_cells()` and the ghost-preview highlight.

**(b) More tile types.** The engine already supports **all 5** canonical
tiles — `_hexcore.canonical_tiles()` returns them, and `Board.place()` accepts
any of their matchings. The 2-tile restriction is purely a *deck* choice from
Stage 0 (tile 0 : tile 2 = 1:2 was the ratio that passed the loop-closure
gate), not an engine limit. The five:

| id | arc spans | character | distinct rotations |
|---|---|---|---|
| 0 | (1,1,1) | three tight turns | 2 |
| 1 | (1,2,2) | one tight + two wide | 6 |
| 2 | (1,1,3) | two tight + one straight | 3 |
| 3 | (2,2,3) | two wide + one straight | 3 |
| 4 | (3,3,3) | three straights | 1 |

Making `tile_types`/`deck_counts` config means the game can ship the validated
2-tile deck as the default *and* offer a 5-tile variant. **Caveat worth
testing before shipping a 5-tile mode as "the" game:** Stage 0 found deck
composition is the dominant lever on loop-closure rate — a uniform 5-tile deck
*failed* the gate badly (0.67 loops/board, 51% of boards closed zero loops)
which is why the 1:2 two-tile deck exists at all (§2.2–2.3). A 5-tile mode
therefore needs its own deck-ratio tuning pass, or it risks being a
significantly worse game. Treat "support all 5" as an engine/config
requirement (cheap, do it), and "ship a 5-tile deck" as a design question
requiring the Stage 0 sweep re-run for whichever ratio is chosen.

Rotation aliasing note: several tiles have fewer than 6 visually-distinct
rotations (tile 4 has exactly 1). The UI should ideally skip redundant
rotations when cycling with R, using the per-type distinct-rotation classes
(`train_selfplay._ROTATION_CLASS_REP` has these for tiles 0 and 2; the other
three need the same table computed once from `tile_arcs`).

### 1.3 Loop detection (the one genuinely tricky port)

Port the union-find loop-closure + ray-cast area exactly as specified. Two
proven references to follow (pick the clearer one to translate, then validate
against the other):
- `hex_truchet/_hexcore.py` — the readable union-find `Board` (arcs, port
  adjacency, `components()`, `enclosed_cells`).
- `hex_truchet/fast.py::_total_loop_area` — the batched tensor version (label
  propagation + ray-cast parity) if a more array-oriented port fits GDScript
  better.

Loop = connected component of arcs (linked across shared cell boundaries) in
which every arc's both ports are matched to a neighbor (no open ports).
Enclosed area = ray-cast crossing parity along the fixed edge-0 rays. The board
is tiny (≤111 arcs), so a plain union-find recompute per placement is instant —
no need for the batched approach's cleverness.

### 1.4 Validation (do this before building any UI on top)

Differential-test the GDScript engine against Python, the same way the env was
validated:
1. Python (`export_pack.py`) dumps N random games as packs: the move sequence
   **and** per-step `gained` / `loops` / `scores`.
2. A Godot headless test scene replays each pack's moves through `GameState`
   and asserts identical `gained`, loop sets, and final scores at every step.
3. Only when that passes is the engine trusted. Run it headless
   (`godot --headless --script res://tests/test_rules.gd`) so it can go in CI.

---

## 2. Bots — all non-ML, one tunable search

**No neural net is needed for a strong opponent.** This is the biggest change
from the first draft of this plan, and it's grounded in measurement
(HANDOFF.md §8.9, `training/sweep_bots.py`): a plain rollout search beats the
greedy heuristic overwhelmingly, and — critically — a *cheap* configuration of
it is both strong and fast enough for interactive play. The trained RL policy,
by contrast, plateaued at ~8% win rate vs greedy. So the whole roster is
deterministic search: no weights to export, no inference code, no ML runtime,
and difficulty is a knob rather than a checkpoint. Trivial to port to GDScript.

Bots share one interface: `func choose(state: GameState) -> Array` returning
`(hand_slot, cell, rotation)`.

### 2.1 The one algorithm (everything else is a preset of it)

**Rollout search** — at the bot's turn:
1. Rank legal `(hand_slot, cell, rotation)` actions by immediate area gain
   (this ranking IS the greedy heuristic) and take the top **K**.
2. For each of those K candidates: apply it to a cloned state, then play
   **both** sides forward with plain greedy for **depth** plies (or to game end
   if `depth == 0`).
3. Score each candidate by the resulting margin from the bot's perspective;
   play the best one.

This is a *rollout algorithm* (one step of policy improvement over the greedy
base policy). Two knobs:
- **K** — candidates tried. `K = 1` short-circuits to exactly greedy.
  Dominant strength *and* cost lever, roughly linear in both.
- **depth** — plies simulated. `0` = to game end (strongest, priciest).
  Truncating mainly saves *early*-game cost: a full rollout at `t=4` is ~33
  plies but only ~7 at `t=30`.

Reference implementation: `training/lookahead_bot.py::lookahead_action`
(Python; the GDScript port is direct — it needs only `clone()`, `apply()` and
the greedy ranking, all already in `GameState`).

### 2.2 Difficulty presets (measured, see `training/bot_sweep.log`)

Measured vs plain greedy, seats rotated, on the spec's 2-tile /
adjacency-required configuration. `s/move` is Python-on-CPU and is a
**pessimistic** proxy for GDScript — treat the *relative* costs as the signal:

| preset | K | depth | win vs greedy | margin | s/move (py) |
|---|---|---|---|---|---|
| `random` | — | — | ~0 | very negative | ~0 |
| `easy` | 1 | — | 0.467 (is greedy) | +0.00 | 0.009 |
| `medium` | 3 | 2 | **0.933** | +10.20 | **0.030** |
| `hard` | 3 | 8 | 0.900 | **+12.43** | 0.140 |
| `expert` | 8 | 0 | **1.000** (n=50) | +22.64 | ~29 ⚠️ |

Full sweep (`training/bot_sweep.log`, n=30/config, seats rotated):

| K | depth | win | margin | s/move |
|---|---|---|---|---|
| 1 | — | 0.467 | +0.00 | 0.009 |
| 2 | 4 | 0.900 | +9.10 | 0.051 |
| **3** | **2** | **0.933** | +10.20 | **0.030** |
| 3 | 4 | 0.900 | +9.67 | 0.072 |
| 3 | 8 | 0.900 | +12.43 | 0.140 |
| 4 | 4 | 0.933 | +10.23 | 0.156 |
| 5 | 4 | 0.933 | +10.23 | 0.168 |
| 5 | 8 | 0.900 | **+13.03** | 0.368 |
| 6 | 6 | 0.800 | +8.43 | 0.294 |

Presets are defined in `training/bots.py::PRESETS` — the shared source of truth
to keep the GDScript port in sync with.

Four things this measurement says that shape the design:
- **⚠️ Read this table for magnitudes, not for ranking.** At n=30 the standard
  error on a win rate near 0.9 is ~5.5 points, so *every* search config from
  0.80 to 0.93 is statistically indistinguishable. The one solid, huge
  difference is greedy (0.467) vs any-search (0.80–0.93). Don't tune presets
  on the ordering here without a larger sample — this project has already been
  burned once by trusting a small-sample win rate (HANDOFF.md §7.4).
- **More search is not reliably better.** (K=6, depth=6) scored *lowest* of the
  search configs while costing 10× `medium`. Probably noise, but it does rule
  out "just crank K and depth" as a strategy — there is no meaningful strength
  left to buy above `medium` in this range.
- **Depth is not where the strength is.** `(K=3, depth=2)` matches or beats
  every deeper config at a fraction of the cost. The value is in *considering
  several candidates*, not simulating far ahead — greedy's flaw is myopia about
  alternatives, not a short horizon. **Raise K before raising depth.** Depth
  does appear to buy *margin* (the less-noisy metric: +12.4 at depth 8 vs +10.2
  at depth 2), i.e. deeper search wins by more, not more often.
- **The difficulty ladder needs handicapping, not more search.** Since every
  search config saturates at ~90%, `medium`/`hard`/`expert` differ in **how
  badly they beat you**, not whether they do. A genuinely *competitive* tier for
  a human player will need explicit handicapping — K=2, occasionally taking the
  2nd-best candidate, or a probability of playing the plain greedy move.
  **Untested; the most important open design question for the vs-bot mode.**
- **`expert` is a research config, not a shippable one** at ~29 s/move. Use it
  offline only (generating showcase replay packs).

### 2.3 Tile-type generality

The search is **tile-type agnostic** — it enumerates whatever legal actions the
state reports, so a 5-tile game (§1.2b) needs no bot changes. The only
2-tile-specific thing in the Python code is a caching fast-path
(`train_selfplay._ROTATION_CLASS_REP`, which memoizes per-tile distinct
rotations for tiles 0 and 2); a GDScript port should just compute that table
for all configured tile types at startup from `tile_arcs`.

### 2.4 Optional later: a distilled net

If a *strong-and-instant* opponent is ever wanted (e.g. for a web export, or
mobile), the path is distillation, not RL from scratch: generate
`(observation, expert-action)` pairs from the `expert` preset and train a small
net by supervised imitation. Scaffolding for this already exists
(`training/generate_lookahead_data.py`, plus `lookahead_action(...,
return_value=True)` which returns a free Monte-Carlo value target). **Not
needed for v1** — `medium` is already fast enough — so treat this as a
performance optimization to reach for only if measurement says it's necessary.

Bot moves run on a short timer / `await` a frame so the human sees the move
land with the same draw-on animation as their own.

---

## 3. Rendering (carried over from the web viewer)

Unchanged concept, ported to a Godot `_draw()` (see the arc helper below). One
addition for interactive play: a **ghost preview** of the currently-selected
hand tile at the hovered legal cell, rotating with input, before commit.

Hex corners: 6 points at `30° + 60°k`, radius `L/sqrt(3)`. Arc between edges
`ea,eb` of a cell — tangent-continuous circular arc, **sampled into points**
(never an SVG-style arc command; that was the bug that threw arcs outside the
cell). GDScript:

```gdscript
func arc_points(center, ea, eb) -> PackedVector2Array:
    var Pa := center + edge_off[ea]
    var Pb := center + edge_off[eb]
    if hex_span(ea, eb) == 3:
        return PackedVector2Array([Pa, Pb])          # straight-through
    var oc = line_intersect(Pa, edge_off[ea].orthogonal(),
                            Pb, edge_off[eb].orthogonal())
    if oc == null: return PackedVector2Array([Pa, Pb])
    var r : float = oc.distance_to(Pa)
    var a0 := (Pa - oc).angle()
    var d := wrapf((Pb - oc).angle() - a0, -PI, PI)    # short arc, stays inside
    var pts := PackedVector2Array()
    for i in 19: pts.append(oc + Vector2(cos(a0 + d*i/18.0), sin(a0 + d*i/18.0)) * r)
    return pts
```

Draw with `draw_polyline(pts, color, width, true)`. Loop arcs in the loop's
ramp color (thicker), runs dim/player-tinted, enclosed cells filled at low
alpha. Two theme token sets (dark/light) in a `Palette` autoload, mirroring the
web viewer's CSS variables and 8-color loop ramp.

---

## 4. Interaction (human play)

- **Hand UI**: the current player's 3 tiles shown as buttons/thumbnails at the
  bottom; click to select. (Opponent's hand shows backs/count only, honoring
  the private-hand rule — matters when playing vs bot.)
- **Placement**: hover the board → legal cells highlight (from
  `legal_cells()`); the selected tile ghosts at the hovered cell. Scroll / R
  key rotates the ghost; click commits `apply(...)`, which animates the
  draw-on and any loop-seal flash, updates scores, advances the turn, refills
  the hand.
- **Illegal feedback**: clicking a non-legal cell does nothing (or a tiny
  shake) — the game never needs the env's "redirect to smallest legal action"
  rule, because a human is choosing among surfaced legal options.
- **Camera**: `Camera2D`, fit-to-view on start, scroll-zoom + drag-pan for
  dense end-states.
- **End of game**: full board → banner with final scores + margin, "who won",
  play-again / change-mode.

---

## 5. Modes & flow

`Main.gd` owns a `GameState` and a mode:
- **Hot-seat**: both seats are human input. Hidden-hands should be an *option*,
  not forced — on one shared screen, showing both hands is often the nicer
  experience.
- **Vs Bot**: seat 0 human, seat 1 a bot at a chosen difficulty (§2.2); after
  the human commits, the bot `choose()`s and plays on a timer.
- **Replay**: no `GameState` mutation from input — instead load a pack (§6) and
  drive the board from its `steps[]` via the transport bar (play/pause/scrub),
  exactly the current web viewer. Reuses the same `Board` renderer.

**Rule options, orthogonal to mode** (all three modes honor them; see §1.2):
- **Placement**: *adjacency-required* (default, the screened/validated rule) or
  *free placement* (place anywhere empty).
- **Tile set**: the validated 2-tile deck (default) or a 5-tile variant — with
  the deck-ratio caveat in §1.2b.

A start menu picks mode + rule options (and for vs-bot the difficulty; for
replay, the game from a `manifest.json`). Rule options should be recorded into
any pack the game saves, so a replay reproduces the rules it was played under.

---

## 6. Pipeline: sharing data with Python

- **Replay packs**: extend the existing `viz/export_game.py` into
  `export_pack.py` — dump games (greedy-vs-greedy, `expert`-vs-greedy, random,
  and free-placement / 5-tile variants) to `packs/*.json` + a `manifest.json`.
  Same JSON shape the web viewer already uses (geometry + per-step
  arcs/loops/scores), so Godot's replay mode and the web viewer share one
  format. Include the rule options (§5) in each pack's metadata.
  The offline `expert` preset (§2.2) is ideal here — it's far too slow for live
  play but perfect for generating high-quality showcase games.
- **No weights to ship.** The bot roster is pure search (§2), so nothing needs
  exporting from Python for the game to have a strong opponent. (Only the
  optional distilled net of §2.4 would need a weight-export step.)
- **Recording human/played games**: Godot's `GameState` can serialize a played
  game to the same pack format → shareable / re-loadable in the web viewer or
  as a Godot replay. (Nice symmetry: play a game, then scrub your own game.)

---

## 7. Milestones (each independently shippable)

1. **Rules engine + differential test** (§1) — headless `GameState`, validated
   against Python packs. Build the config hooks (`free_placement`,
   `tile_types`/`deck_counts`) in from the start — they're cheap now and
   invasive later — but validate against the spec's exact configuration. No UI
   yet. *The foundation; everything depends on it.*
2. **Board renderer + replay mode** (§3, §5) — port the web viewer's drawing,
   load a pack, scrub it. Gets a visible artifact fast and reuses milestone 1's
   geometry. *(≈ current web viewer, in-engine.)*
3. **Hot-seat play** (§4) — hand UI, legal-cell highlight, ghost preview,
   rotate, click-to-place, scoring, end screen. **First playable.**
4. **Bots** (§2) — port `lookahead_action` (K/depth), wire the §2.2 presets,
   vs-bot mode with difficulty selection and move timer/animation. **First
   "play against the computer," and it's already a strong one.**
5. **Rule options in the UI** (§1.2, §5) — free-placement toggle and tile-set
   selection in the start menu; record them into saved packs. (The engine
   support landed in milestone 1; this is the surfacing.)
6. **Polish & stretch** — animations (draw-on, loop-seal flash), sound, hidden
   opponent hand toggle, record-played-game-to-pack, `MovieWriter` video
   export of a replay, optional distilled net (§2.4) *only if* profiling says
   the search is too slow, and optionally aligning packs to simulacrum's
   official trajectory schema so `traj.picker` can feed Godot.

Recommended order note: build milestone 1 first no matter what — a wrong rules
engine makes every mode subtly broken, and the differential test is cheap
insurance. Milestones 3 (hot-seat) and 4 (bots) are the fastest path to
"actually playable" — and because the bots are pure search, milestone 4 needs
no ML work at all, which is why it now lands before the rule-options polish
rather than after a neural-net milestone.
