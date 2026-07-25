# Hex Truchet Game Design Search — Handoff Document

**Purpose:** hand off to Claude Code (or a fresh session) with full context on what's done,
what's found, what's broken, and what's next. Read alongside `hex-truchet-research-plan.md`
(the original plan), which this document tracks progress against but does not replace.

**Code location:** `/home/claude/hextruchet/` in the sandbox this was developed in. All
files listed below need to be copied out before starting a new session, or recreated from
the descriptions/code included in this document.

---

## 1. Current status at a glance

| Stage | Status | Result |
|---|---|---|
| 0 — Geometry baseline | **Complete, gate passed (with deck fix)** | Uniform deck fails gate; reweighted deck (tile 0:2 = 1:2) passes |
| 1 — Scoring variants | **Complete** | `area_linear` and `length_plus_area` recommended |
| 2 — Deliberate play (screening) | **Complete — results in, but a real strategic-depth concern** | Perf "hang" was misdiagnosed, not a hang (§7.1); greedy crushes random decisively (gate met); denial gives greedy no measurable edge despite playing genuinely differently (§7.3) — open question, not yet resolved |
| 3 — RL self-play | **Shallow-ceiling question SETTLED: false.** A non-learned lookahead bot beats greedy 50/50, margin +22.6. RL-vs-greedy plateaued far below that (~9%, margin -9.5) — a training-method gap, not a game-ceiling one. | Env validated; self-play plateaued at 0%; direct-vs-greedy RL climbed to ~9% then plateaued; lookahead search (no training at all) crushes greedy outright. See §8.5-8.9. |
| 4 — Godot viewer/game | Re-scoped to a **playable game** (hot-seat, vs-bot, replay, free-placement + 5-tile options), not just a viewer. Bot roster settled: pure search, no ML. | See `viz/GODOT_GAME_PLAN.md`; bot roster §8.11, game-design requests §8.12. Not started in-engine; a web viewer exists (`viz/viewer.html`). |

**The former blocker (§6, old) is resolved and was misdiagnosed** — see §7.1. The
strategic-depth question that replaced it (§7.3) is now **resolved** — see §8.9. There is real,
large exploitable depth beyond greedy; the open work is building an agent that reaches it
efficiently (RL as currently configured falls well short of what's achievable).

---

## 2. Stage 0 results (complete)

### 2.1 Tile enumeration

6 edges, perfect matchings (3 arcs each) → **15 total matchings**, quotiented by rotation
into **5 canonical tiles**:

| Tile | Arc spans | Orbit size |
|---|---|---|
| 0 | (1,1,1) — three tight turns | 2 |
| 1 | (1,2,2) | 6 |
| 2 | (1,1,3) — two tight turns + one straight-through | 3 |
| 3 | (2,2,3) | 3 |
| 4 | (3,3,3) — all straight-through | 1 |

Orbit sizes sum to 15 (verified). This is the full canonical set — there is no way to get
more tile variety without changing the tile's fundamental structure (edge count, arc count).

### 2.2 Uniform deck fails the gate

Random-filling boards with all 5 tiles equally likely:

- Radius 3 (37 cells): mean **0.67 loops/board**, **51% of boards close zero loops**
- Radius 4 (61 cells): mean 1.29 loops/board

Plan's gate required mean ≥5 loops/board. Uniform deck fails badly. A game where half of
all boards score zero is not viable.

### 2.3 Deck reweighting fixes closure rate — but reveals a structural ceiling

Swept many weightings of the 5 tiles (see `stage0b_deck.py`, `stage0c_candidates.py`,
`stage0d_grid.py`). Key findings:

- **Deck composition is the dominant lever.** Loop count spans 0.64 to 9+ loops/board on
  identical board geometry, purely from reweighting.
- **Winning deck: tile 0 : tile 2 = 1 : 2.** Passes the gate at both radii:
  - Radius 3: 4.28 loops/board, P(zero)=0.006, mean area 3.62, area SD 2.39, length SD 3.29
  - Radius 4: 8.04 loops/board, P(zero)=0.000, mean area 4.43, area SD 2.93, length SD 4.28
- **Physical-deck friendly**: only 2 distinct tile shapes needed, in a simple 1:2 ratio.

### 2.4 Minimal-loop dominance: a real, unresolved structural limitation

**This is the single most important finding from Stage 0 and should not be forgotten
or glossed over in later stages.**

No matter how the 5-tile deck is reweighted, **roughly 65–80% of all closed loops are
minimal (length 3)**. This was tested exhaustively:

- Every deck weighting tried (corners, pairs, ratios from 1:10 to 1:1, 3-tile blends)
  trades loop-count against %-minimal along the same Pareto frontier — there is no
  interior sweet spot that escapes it (`stage0d_grid.py`, `stage0e_antiminimal.py`).
- **The %-minimal floor (~57–65%) holds even under a uniform deck across board radii 2
  through 5** — meaning this is not a deck-tuning artifact, it's a property of the tile
  geometry itself. Minimal 3-loops are the "cheapest" coincidence in this tile system and
  will always dominate.
- **Attempted fix: blank/partial "spacer" tiles** (0-arc or 2-arc tiles mixed into the deck,
  hypothesized to physically block tight 3-tile clusters). **This was tested and definitively
  refuted** (`stage0f_spacers.py`): every spacer scenario made things *worse*, not better —
  P(zero) rose sharply and %-minimal rose further. Mechanism: a spacer tile placed anywhere
  along a long loop kills it, but a long loop needs many tiles to all cooperate, so spacers
  hurt long loops far more than short ones, concentrating probability mass further onto the
  3-loop case. **Do not revisit spacer tiles without a new theoretical reason to expect a
  different outcome — this path is closed.**

**Where this leaves the design:** reweighting the existing 5-tile set is maxed out. Two
honest paths remain, per the original plan's §6.2/6.3:

1. **Accept the ~70% minimal-loop floor and design scoring around it** — make minimal loops
   cheap/fast points, put real decision-weight on the rarer long loops and area (which do
   have genuine spread even given the floor: area SD ~2.4–2.9, max area into double digits).
   This is the path currently being pursued (Stage 1 onward).
2. **Change the tile's fundamental geometry** (more edges, different arc count, non-hex
   cells) — a bigger redesign, not attempted, would require returning to first principles.

### 2.5 Deck construction going forward

`make_deck(n_cells, rng)` in `stage2_screen.py` builds the exact-ratio deck:
`n0 = round(n_cells/3)` copies of tile 0, remainder tile 2, shuffled. This is the
canonical deck-construction function; reuse it, don't reinvent.

---

## 3. Stage 1 results (complete)

Tested 9 scoring rules (`stage1_scoring.py`) on the same random games (tile 0:2 = 1:2 deck),
measuring **separation** (does the rule spread player scores apart) via winner-margin,
tie-fraction, and winner-share-std, at radius 3/4 and 2/3 players.

**Key findings:**

- `count_only` (pure loop count, size-blind) is consistently the **weakest** separator —
  confirms that even with 70%+ minimal loops, *who gets the non-minimal ones* still carries
  real signal that a size-blind rule throws away.
- `area_linear` has among the **lowest tie rates** at every board size/player count tested,
  and its separation doesn't depend on rare multi-loop-closure swings (which occur ~1–12%
  of placements depending on deck, per Stage 0's `closures_per_placement` measurement).
- Superlinear/jackpot rules (`length_superlinear_2.0`, `jackpot_len>=6`) separate players
  *more*, but mostly by amplifying the variance of rare lucky placements — not skill. Flagged
  as **high-variance alternatives worth testing for player preference later**, not the lead
  candidates for measuring skill separation in Stage 2.
- **3-player games compress margins and raise tie rates** relative to 2-player, as expected
  (score split three ways). Worth remembering when interpreting Stage 2 3-player results —
  a "weak" separation result at 3p may just be arithmetic, not a bad rule.

**Recommendation carried into Stage 2:** `area_linear` and `length_plus_area` (k=1.0). Stage 2
so far has only exercised `area_linear`; `length_plus_area` has not yet been run through the
Stage 2 engine.

---

## 4. Stage 2 engine (built, correctness-verified, perf-blocked)

### 4.1 What's implemented

**`agents.py`** — core move-search infrastructure:
- `legal_cells_free`, `legal_cells_adjacent` — the two placement rules from the plan
- `score_delta_for_move` — simulates a candidate placement via a **dict-copy clone** (not
  full replay — this was an optimization made after finding the naive replay approach was
  O(cells) per clone, making greedy search cubic overall)
- `random_agent`, `greedy_agent`, `denial_agent` — the plan's three non-RL agent types.
  **`denial_agent` has not yet been exercised in Stage 2 screening** — it was deliberately
  deferred as the slowest option, pending a working screening pass with just random/greedy.
- `GameBoard` — `Board` subclass tagged with `adjacency_required`

**`graph.py`** (extended from Stage 0) — added `Board.try_place_and_get_new_loops(cell,
matching, rotation, enclosed_fn)`, which places a tile, computes newly-closed loops, and
returns an `undo()` closure to revert. This is the method Stage 2's move search uses to
evaluate candidates without permanently mutating the board.

**Important history on this method — read before touching it again:**
1. First attempt used a localized BFS from only the new tile's arcs, to avoid O(cells)
   recomputation per candidate. **This was tested and found broken**: it silently
   under-reported loop closures when a single tile's arcs belonged to two independent loops
   simultaneously (verified failing case: a placement closing loops of length 3 and 18
   simultaneously — the BFS conflated/mishandled this and returned zero closures).
2. Reverted to full `components()` recomputation per call — correct, but back to O(cells)
   per candidate, same asymptotic cost as the original naive approach. **The performance
   problem from the naive approach was never actually solved** — this method exists for
   place/undo ergonomics, not speed.
3. **A second, separate bug** was found and fixed in the same method: `prev_loop_keys` was
   originally computed *after* mutating `self.placed/arcs/port_to_arc` for the new tile,
   making "before" and "after" identical and `new_keys` always empty. Fixed by moving the
   `prev_loop_keys` computation before the mutation.
4. **A third bug**: the `area` field was stored as `enclosed_fn(self, l)` (a raw list of
   cells) instead of `len(enclosed_fn(self, l))` (a count), causing a `TypeError` in
   downstream scorers expecting a number. Fixed.
5. **All three fixes are verified**: full `test_geometry.py`/`test_graph.py` suite still
   passes, and 500/500 randomized trials cross-checking `try_place_and_get_new_loops`
   against the independent `components()`/`loops()` slow path show zero mismatches.

**`stage2_screen.py`** — the screening harness:
- `make_deck(n_cells, rng)` — canonical deck construction (see §2.5)
- Three drafting mechanisms implemented: **`bag`** (draw random tile, place), **`pool`**
  (k=3 face-up, pick one, refill), **`hand`** (hand of h=3, play one, refill). Snake draft
  and offer/choose from the original plan (mechanisms D/E) are **deliberately deferred** —
  they need a separate draft-then-place phase structure, judged not worth building until
  something in the simpler mechanisms looks promising.
- `choose_move`, `do_place` — shared move-selection and execution logic across mechanisms
- `separation(score_lists, n_players)` — same margin/tie/winrate metrics as Stage 1
- `run_screen(n_games, radius, n_players)` — the main sweep entry point: iterates all
  3 mechanisms × 2 placement rules × 2 agent types (random, greedy — **denial not yet
  included**), running `n_games` per config and printing a results table

### 4.2 Correctness is solid; performance/hang behavior is not understood

Per-game timing, tested in isolation with **fresh `Random(seed)` per game** across seeds
0–11, is fast and consistent with no fat tail:

| Config | Median | Max (of 12 seeds) |
|---|---|---|
| bag, greedy | 0.66–0.86s | under 1x median |
| pool, greedy | 1.71–2.51s | under 1.2x median |
| hand, greedy | 1.77–2.49s | under 1.1x median |
| any random-agent config | 0.01s | 0.01s |

This held at both 2-player and 3-player counts (3-player was, if anything, slightly faster).

**But `run_screen()` itself times out**, even at reduced scale (50 games, 2-player only,
900s timeout budget) — despite the isolated-timing extrapolation predicting well under
that budget (~500–600s for the full sweep).

**The discrepancy was traced to one concrete difference, not yet confirmed as the actual
cause:** `run_screen()` seeds **one `Random` object per config** and calls the mechanism
function `n_games` times *reusing that same rng, drawing from one continuing random
stream* — whereas the isolated timing test used a **fresh `Random(seed)` per individual
game** (seeds 0–11 independently). If some point deep in a single continuing stream
triggers a pathological game — most plausibly in the `hand` mechanism's stall-detection
loop (`stall += 1; if stall > n_players * 2: break`), which could interact badly with a
specific sequence of empty-hand states — a single bad draw deep in a 50-long continuing
sequence could hang or loop far longer than any of the 12 independently-seeded test games
happened to hit.

**This was identified but not confirmed or fixed before work was paused.** The next
session should:

1. Reproduce directly: run `run_screen`'s exact seeding pattern (one `Random` object,
   many sequential calls) against a **small** `n_games` (5–10) and watch for the first
   sign of a stall, rather than assuming it'll reproduce at 50.
2. If confirmed, add instrumentation (print game index and elapsed time per game inside
   the `run_screen` loop, not just the aggregate) so a hanging game is caught immediately
   rather than inferred from a timeout.
3. Specifically audit the `hand` mechanism's stall logic in `play_hand()` for a
   correctness/liveness bug — e.g., whether `deck` can be non-empty while all `hands[p]`
   are empty in a way that isn't actually a terminal state, causing the `while any(hands)
   or deck` condition to loop without making progress, only saved from an infinite loop by
   the `stall` counter escaping late.
4. Consider whether `choose_move`'s greedy branch (which is the expensive path — nested
   loop over candidates × legal cells × 6 rotations, each calling
   `try_place_and_get_new_loops`) could, for some rare board states, face a much larger
   `legal` cell count than typical (e.g., early free-placement turns have up to ~37 legal
   cells vs adjacency-required's much smaller frontier) — worth checking if `pool`/`hand`
   greedy configs on `adjacency=False` have a wider variance in per-move cost across a
   single game's own turns, not just across games.

### 4.3 Known stray files / environment note

A file `stage2_results.csv` has appeared twice in the working directory **without being
generated by any script in this session** — both times it was deleted (`rm -f`) rather than
trusted, since its content pattern matches a known bug (a `t=0` falsy-check bug) from an
abandoned early scratch file (`scoring.py`, since deleted) that was never actually run to
completion. **If this file reappears in a new session, do not treat it as valid data** —
regenerate everything from the scripts in this directory instead. It's not currently
understood why it keeps appearing; worth a quick check of whether the sandbox is reusing
state across sessions in an unexpected way.

---

## 5. File inventory

```
hextruchet/
  geometry.py            -- axial coords, tile enumeration/canonicalization (Stage 0 core, stable)
  geometry_ext.py         -- blank/partial tile types (used only for the spacer experiment,
                             §2.4 — that path is closed, this file is not needed going forward
                             unless spacer tiles are revisited)
  graph.py                -- Board class, union-find cycle detection, try_place_and_get_new_loops
                             (Stage 0/2 core, stable — see §4.1 history before editing)
  stage0.py                -- Stage 0 main experiment: random fill + loop/area/run stats,
                             enclosed_cells() (ray-casting area function, used everywhere downstream)
  stage0b_deck.py           -- first deck-weight sweep (corners + hand-picked points)
  stage0c_candidates.py     -- detailed distributions for top Stage 0b candidates
  stage0d_grid.py           -- broader deck-weight sweep with composite balance score
  stage0e_antiminimal.py    -- investigation of the minimal-loop structural ceiling (§2.4)
  stage0f_spacers.py        -- blank/partial spacer tile test (refuted, §2.4 — closed path)
  stage1_scoring.py         -- Stage 1: 9 scoring variants, separation metrics
  agents.py                 -- Stage 2: random/greedy/denial agents, move-search infra
  stage2_screen.py           -- Stage 2: screening sweep harness (BLOCKED, see §4.2)
  test_geometry.py           -- geometry unit tests (all passing)
  test_graph.py              -- cycle-detection unit tests incl. union-find/BFS cross-check (all passing)
  results_stage0.json        -- saved Stage 0 aggregate results
  full_stage0c.txt           -- saved Stage 0c console output (radius 3 + 4 detail tables)
```

**Not yet created:** any Stage 2 results file (blocked), Stage 3 RL code, Godot export code,
Godot viewer project.

---

## 6. Recommended immediate next steps for Claude Code (session 1 — superseded, kept for history)

**All five items below are done.** See §7 for what was actually found and what to do now;
§7.7 supersedes this list. Kept here only as a record of what session 1 thought the plan was.

1. ~~Root-cause the Stage 2 hang per §4.2~~ — done; it wasn't a hang. See §7.1.
2. ~~Re-run the full screening sweep~~ — done. See §7.1, §7.3.
3. ~~Add the `denial` agent into the screening sweep~~ — done. See §7.3.
4. ~~Test `length_plus_area` scorer through the same Stage 2 harness~~ — done. See §7.5.
5. Keep the §2.4 minimal-loop-dominance finding in view when interpreting Stage 2 results —
   **still live advice, restated and sharpened in §7.3 and §7.7.4.**
6. Dev notes stay out of git per earlier instruction; this handoff document itself is a
   working doc, not a public artifact — keep it out of git alongside other dev notes, and
   write the eventual public account from the structured results files once the search
   concludes, not from this document directly. **(Still applies — this repo has no git init
   as of session 2 either.)**

---

## 7. Session 2: Stage 2 unblocked, denial wired in — a strategic-depth concern

### 7.1 The "hang" was misdiagnosed — root cause and fix

The §4.2/§6 hang was investigated by reproducing `run_screen`'s exact seeding pattern (one
`Random` per config, sequential calls) plus 60 independent-seed stress games per config.
**There was no hang, no infinite loop, no pathological game** — worst-case game time was
only ~1.15–1.2× the median at every config, both 2p and 3p. The leading hypothesis in §4.2
(a shared-RNG stream triggering a stall in `play_hand`'s stall-detection loop) does not
reproduce.

The real cause: aggregate runtime. Greedy move search called `Board.components()` (a full
O(all-arcs) rebuild) twice per candidate — once to get the base board's loop set, once after
the trial placement — inside `candidates × legal_cells × 6 rotations`. At ~1.8s/game for
`pool`/`hand`, the full 24-config double sweep (2p+3p) totalled ~850s, just over the ~900s
timeout budget used in session 1.

**Fix** (`graph.py`, `stage2_screen.py`):
- `try_place_and_get_new_loops` now accepts an optional precomputed `prev_loop_keys` — the
  base board's loop set is identical across every candidate in one move, so it's computed
  once per move instead of once per candidate.
- Greedy/denial search now only evaluates frontier cells (adjacent to an already-placed
  tile) under free placement — an isolated placement always scores 0 and can never be the
  best move, so this is behaviour-preserving, not an approximation (see the docstring on
  `_frontier_cells` in `stage2_screen.py`).
- Fixed a real, separate bug: seeding used `hash(tuple)`, which Python randomizes per
  process (`PYTHONHASHSEED`), so sweeps were not reproducible run-to-run. Replaced with
  `config_seed()`, an md5-based deterministic seed.
- Verified: full `test_geometry.py`/`test_graph.py` suites still pass; greedy/random
  iteration order and tie-break logic are unchanged (same nested-loop order, same
  first-strict-greater tie-break), just extracted into a shared `_search_own_moves` helper.

Net effect: ~2× speedup. Full sweep now runs in ~450–990s depending on which agents are
included, comfortably inside a normal timeout.

### 7.2 Mixed-agent matchups added

`choose_move` and all three mechanism functions (`play_bag`/`play_pool`/`play_hand`) now
accept either a single agent-type string (old behaviour, same agent every seat) or a
per-player list (`["greedy", "random"]`) for head-to-head testing. New
`run_matchup(n_games, radius, n_players, skilled_agent, baseline_agent, out_path)` runs
exactly one seat as `skilled_agent` and the rest as `baseline_agent`, rotating which seat
gets the skilled agent evenly across games so seat-order advantage (real, see §7.4) doesn't
get conflated with agent-skill advantage. **This is the tool to use for any future
agent-strength question** — symmetric-population `margin` from `run_screen` is NOT a
skill-vs-skill measurement (see §7.3 for why that distinction mattered here).

### 7.3 Greedy vs random: decisive. Denial vs greedy: not — and this is the important finding.

**Greedy crushes random.** `run_matchup(60, ...)`, all mechanisms × adjacency × {2p,3p}:
greedy wins 100% of games in 11/12 configs, 98.3% in the twelfth (fair baseline 50%/33%).
Gate criterion "greedy meaningfully beats random" (plan §4.5) is unambiguously met.
(`matchup_results.csv`)

**Denial gives no edge over greedy.** Same matchup setup, denial vs greedy: 2p win rates
41.7%–58.3% (fair=50%), 3p 26.7%–33.3% (fair=33.3%) — every config is within sampling noise
of a coin flip. (`matchup_denial_vs_greedy.csv`)

This is *not* the same as "denial reduces to greedy's picks." The full symmetric screen
(`stage2_results_with_denial.csv`, all three agents × 3 mechanisms × 2 adjacency ×
{2p,3p}) shows denial-vs-denial margins consistently *higher* than greedy-vs-greedy margins
(10/12 configs, e.g. 2p pool/False: greedy margin 0.260 → denial margin 0.310). Denial *is*
choosing differently than greedy — mutual denial pressure compounds into more decisive
outcomes when both players do it — but that different behaviour doesn't translate into an
edge against a plain greedy opponent, who isn't reciprocating the denial pressure and so
doesn't get "walked into" whatever setup denial is trying to create.

**Why this matters, and what it doesn't prove:** the two hand-designed heuristics built so
far (greedy: one-ply own-gain max; denial: greedy plus a capped 1-ply opponent-penalty) are
statistically indistinguishable head-to-head. Combined with §2.4's minimal-loop-dominance
finding (65–80% of closed loops are the cheapest/length-3 kind, structurally locked in
regardless of deck weighting), this is a real caution sign that this game's strategic
ceiling may be shallow — i.e. "grab the best available loop each turn" might already be
close to as good as it gets, with not much more to gain from deeper opponent modeling.

**This is NOT dispositive.** Only two heuristics have been tried, and denial's opponent
model is itself a coarse approximation (2 tile types, 3 sampled rotations, 10-cell cap, 0.5
penalty weight, top-8 shortlist — see the `DENIAL_*` constants above `choose_move`'s denial
branch in `stage2_screen.py`). A weak denial implementation losing to greedy is consistent
with either "the game is shallow" or "this particular denial agent is badly tuned/too
narrow." **Stage 3 (RL self-play) is the test that actually resolves this** — if a trained
agent still can't beat greedy by much even with a full training budget, that's strong
evidence of a shallow ceiling; if it finds a real, repeatable edge, greedy/denial were just
weak hand-crafted baselines and the game has more depth than these two heuristics found.

### 7.4 Seat-0 advantage — retracted as an `adjacency_required`-specific concern; real but small and rule-independent

**Correction to an earlier claim in this same session.** The first n=60 sweep showed seat-0
win rate 63–72% under `adjacency_required=True` at 2p (vs the fair 50%), and was written up
as a concern specific to the adjacency rule (amplifying the built-in 19-vs-18 tile-count
edge from splitting 37 cells two ways). **A follow-up dig showed this was mostly a noisy
single-sample read, not a stable structural finding**, and the adjacency-specific framing
was wrong. Re-running the exact same seed reproduced the original numbers exactly (ruling
out a code regression as the explanation), but a *different* seed at the same n=60 gave
final seat-0 win rates of 43–58% for the identical configs — a swing far too large and too
consistent across all 6 configs to be pure noise on a per-config basis, which is itself the
finding: **seat-0 win rate is a much higher-variance metric at n=60 than it looks**, and a
single 60-game sample isn't enough to trust for this particular question.

Re-ran at n=400 (random) / n=150 (greedy) per config, with score snapshotted at the
equal-tile-count point (turn 36 of 37, both players have placed 18) as well as at the final
score, to separate "does going first help beyond the extra tile" from "does the extra tile
alone explain everything":

- **At equal tile counts, seat-0 win rate is 39–55%** across all 12 configs (3 mechanisms ×
  2 adjacency settings × random/greedy) — centered *below* 50% if anything. **No evidence of
  a first-mover/board-control advantage independent of tile count.**
- **The only real, measurable effect is the extra 19th tile itself:** it bumps seat-0's win
  rate by a consistent **+6 to +10 points for random agents** (z = 2.3–4.0 vs a null of zero
  — clearly real) and a smaller, noisier **+0.7 to +3.3 points for greedy** (all 6 configs
  positive, but individually within 1 SE at n=150).
- **This bump is essentially the same size under `adjacency_required=True` and
  `adjacency_required=False`** (e.g. `pool`/random: +8.7pts free vs +8.5pts adjacency-required
  — statistically indistinguishable). Adjacency isn't amplifying the parity effect; it's
  unrelated to it.

**Corrected conclusion:** the 2p seat-0 edge is real but small (~6–10 points for random,
smaller for greedy), is a structurally unavoidable consequence of splitting an odd cell
count (37) between two players, and is **not specific to `adjacency_required` at all** — it
shows up equally under free placement. This is a much weaker and more generic finding than
originally reported, and doesn't block `adjacency_required` specifically. If it's worth
addressing at all, the fix is board-size/parity-level (e.g. an even split, or accepting a
small first/second-seat asymmetry as normal for a 2-player game on an odd-cell board), not
an adjacency-rule-level fix.

**Process note for future sessions:** this is a specific instance of a general lesson —
win-rate-style metrics with a lot of near-ties (see the `frac_tied` columns throughout this
document) are noisier at n=60 than they look, and a single sweep's numbers for this kind of
metric shouldn't be treated as settled without either a same-seed reproducibility check or a
larger-N confirmation run. The margin/tie-rate numbers used for scorer/mechanism comparisons
elsewhere in this document (§7.3, §7.5) are averages over continuous scores rather than
binary win/loss, so they're less exposed to this specific problem, but haven't been
stress-tested for sample-size sensitivity either — worth keeping in mind before leaning hard
on any single-sweep number from this document.

### 7.5 `length_plus_area` scorer: no clear differentiation

Ran the full screen (`stage2_results_length_plus_area.csv`). Margins and tie rates are in
the same range as `area_linear` — no breakage, but no clear improvement either. An earlier
draft of this section noted a "suggestive drop" in 2p adjacency-required seat-0 bias between
scorers — **retract that too, per §7.4's finding that seat-0 win rate at n=60 is simply too
noisy a metric to read a scorer effect from a single sweep.** No real signal either way on
whether the scorer choice affects seat balance; not investigated further since §7.4 found
the underlying "concern" wasn't specific to any rule variant in the first place.

### 7.6 File inventory additions (session 2)

```
  stage2_screen.py                     -- UPDATED: fixed perf bug (§7.1), mixed-agent
                                          matchups (§7.2), denial agent wired in natively
                                          (§7.3; intentionally NOT reusing agents.py's
                                          slower clone-based denial_agent/
                                          best_response_value -- see the docstring above
                                          choose_move's denial branch), on_place callback
                                          hook on all three mechanism functions (§7.4)
  graph.py                             -- UPDATED: try_place_and_get_new_loops takes
                                          optional prev_loop_keys (§7.1)
  stage2_seat_parity_diag.py            -- NEW: equal-tile-count vs final win-rate
                                          diagnostic that produced §7.4's correction;
                                          rerunnable for other board sizes/player counts
  stage2_results.csv                    -- area_linear symmetric screen, random/greedy
                                          only (the original blocked run, now complete)
  stage2_results_with_denial.csv        -- area_linear symmetric screen, random/greedy/
                                          denial
  stage2_results_length_plus_area.csv   -- length_plus_area symmetric screen (§7.5)
  matchup_results.csv                   -- greedy vs random head-to-head (§7.3)
  matchup_denial_vs_greedy.csv          -- denial vs greedy head-to-head (§7.3)
```

### 7.7 Recommended next steps (supersedes §6)

1. **Run Stage 3 (RL self-play) before drawing a firm conclusion on strategic depth.** This
   is the highest-value next step — it's the test that distinguishes "shallow game" from
   "weak hand-crafted heuristics," which §7.3 left open. Nothing else in this document
   blocks starting it.
2. ~~Root-cause the `adjacency_required` 2p seat-0 imbalance~~ — **done, and retracted, see
   §7.4.** It wasn't adjacency-specific and wasn't as large as first reported; no action
   needed here before Stage 3.
3. ~~Same-seed `length_plus_area` vs `area_linear` seat-bias comparison~~ — **moot, see
   §7.5.** The thing it would have settled turned out not to exist.
4. Keep §2.4's minimal-loop-dominance finding and §7.3's denial-vs-greedy null result in
   view together when interpreting any Stage 3 results — if a trained RL agent's advantage
   over greedy also turns out to come mostly from faster/more-consistent minimal-loop
   grabbing rather than genuine long-loop setup play, that's the same "shallow ceiling"
   story showing up a third time, and would be strong grounds to revisit the tile geometry
   itself (Stage 0 §2.4 option 2) rather than continuing to tune agents/scorers on the
   current 5-tile set.
5. **New, from §7.4's process note:** any Stage 3 result reported as a win-rate/margin
   number should get at least a same-seed reproducibility check, and ideally an N-scaling
   check, before being written up as settled — §7.4 showed a single n=60 sweep produced a
   63–72% number that a larger sample walked back to ~50%. RL training curves are less
   exposed to this (they aggregate over many more games implicitly), but any final
   agent-vs-agent evaluation number should not be trusted from one small sweep alone.

---

## 8. Session 3: Stage 3 RL environment — built and validated

The RL self-play environment is implemented as a **`simulacrum`** environment (the
differential-tested batched-tensor RL framework; local checkout at
`/Users/mwburke/projects/simulacrum`, installed editable into the new project venv at
`/Users/mwburke/projects/hextruchet/.venv`). This follows the standard simulacrum workflow:
one `spec.md` as the single source of truth, an independent readable `reference.py` and
batched `fast.py`, validated bit-for-bit against each other.

### 8.1 Scope decisions (locked)

One fixed configuration per env package (simulacrum bakes behavior into constants, no
runtime config). This package, **`hex_truchet`**, specs the **private-hand** variant:

- **2 players**, `hand` mechanism (hold 3, play one, refill), `adjacency_required=True`,
  `area_linear` scorer, radius-3 board (37 cells). Rationale in §8.2.
- **Private hands** (POMDP): you see your own hand truthfully; opponent hand slots show
  only occupied-vs-empty (hand size is public, tile identity hidden) via a fixed sentinel
  in the observation. Matches how the Stage 2 greedy/denial heuristics already behaved
  (they never read the opponent's actual hand).
- **Terminal-only zero-sum reward:** 0 every step except the final placement, where
  `reward = score[last_placer] − score[opponent]` (exact integer arithmetic, no float
  tolerance). A **public-hand sibling** (`hex_truchet_public`) is specced but not yet
  built — identical except observation masking.

### 8.2 Why these choices (for whoever picks this up)

- `hand` mechanism = richest decision space (which of 3 held tiles to play), the best shot
  at a learned policy finding something greedy/denial missed — which is the entire point of
  Stage 3 (§7.3).
- `adjacency_required=True` = much smaller branching factor early (frontier vs all 37
  cells), so faster/more sample-efficient training.
- 2 players = standard self-play footing; the 3p credit-assignment complications aren't
  worth it for a first pass.
- Terminal zero-sum reward directly optimizes "beat the opponent" (the Stage 1/2 margin
  metric), not raw score accumulation. Fixed 37-step horizon makes the sparse terminal
  reward tractable.

### 8.3 Environment structure and the one real subtlety

Package: `/Users/mwburke/projects/hextruchet/hex_truchet/` — `spec.md` (source of truth),
`reference.py`, `fast.py`, `schema.json`, `__init__.py` (constants + `Slots` enum),
`tests/`, plus:

- **`_hexcore.py`** — a **verbatim, script-assembled** vendoring of the proven Stage 0-2
  geometry + `Board`/union-find + `enclosed_cells` code (from this `hextruchet_first_pass/`
  dir). `reference.py` reuses it deliberately — that loop-closure logic is exactly what had
  the historical bugs (§4.1) and is now correct + tested (the original 14 Stage-0 tests pass
  unmodified against the vendored copy). **`fast.py` must NOT reuse `_hexcore`'s
  loop-closure logic** — it reimplements closure as batched tensor ops from `spec.md`, and
  the differential test validates the two independent implementations against each other.
  `fast.py` imports only `_hexcore`'s *pure-geometry lookup* (cell coords, tile arc tables)
  to build static index tables at load — never the union-find/area algorithm.

- **RNG model:** tile draws are **live sequential Bernoulli-without-replacement** from the
  fixed `{12 type-0, 25 type-2}` multiset (slot `INITIAL_DEAL` indices 0..5 at reset;
  `DECK_DRAW` index 0 at steps t=0..30), NOT a pre-shuffled deck. `p =
  remaining_type0/remaining_total` computed in float64 so reference and batched match
  bit-for-bit. Verified: every completed episode ends with exactly the 12/25 multiset.

- **The hard part (`fast.py` loop closure as tensors):** batched connected-components via
  min-label propagation over the 222-port graph, a max-propagated "component has an open
  port" flag to separate loops from runs, then ray-cast crossing-parity for enclosed area
  via a pairwise last-occurrence-parity trick over the fixed ≤7-cell rays. Key simplifying
  theorem: **closed loops are sealed and never change once formed**, so `score =
  total_loop_area(after) − total_loop_area(before)` — no need to diff loop *identities* per
  step, just total enclosed area before vs after placement.

### 8.4 Validation status — PASS, training-eligible

`simulacrum validate hex_truchet`: **overall_pass: True**, `required_tests_not_passed: []`,
**`tolerance_fields_used: {}`** (differential test passed with exact bit-equality, no
float tolerance anywhere). All battery tests pass: spec-contract, differential,
batch-independence, invariant-sweep (all 12 invariants), auto-reset, determinism, replay,
throughput. Two skips are optional (no scripted policies / no separate benchmark factory).
A fresh passing `validation_report.json` exists — the gate before training.

**Known tunable, not a blocker:** batched throughput is ~14.4k steps/s at n=1024, only ~4×
the reference. Bottleneck is the label-propagation convergence loop's per-iteration device
sync. Correctness-first; the battery doesn't enforce a min speedup. If training is too slow,
optimize that loop (fixed iteration count + cache previous-step area) — but re-run the full
battery after, since that loop is on the differential-tested path.

### 8.5 Self-play training results (session 3) — plateaued at zero vs greedy, but read carefully

Built a self-play PPO trainer (`training/train_selfplay.py`): single shared policy plays
both seats, terminal zero-sum margin resolved onto each acting player's own action steps
(the credit-assignment the spec deliberately leaves to the training script), Monte-Carlo
returns (justified by the fixed 37-step horizon + terminal-only reward — no bootstrap
needed), PPO clip + entropy bonus, illegal actions masked out of the policy distribution.

**Result, across 6 eval checkpoints (updates 75 through 450, ~1500-update run stopped early
once the pattern was unambiguous):**

| update | vs random (win) | vs greedy (win) | vs greedy (margin) | entropy |
|---|---|---|---|---|
| 75 | 0.76 | 0.000 | −29.8 | 4.80 |
| 150 | 0.93 | 0.000 | −32.7 | 4.15 |
| 225 | 0.90 | 0.000 | −23.8 | 3.02 |
| 300 | — | — | — | 2.54 |
| 375 | 0.86 | 0.000 | −25.6 | — |
| 450 | 0.86 | **0.004** | −24.8 | 2.29 |

The policy clearly learned *something* (random win rate 44%→86-93%, entropy fell steadily —
it became decisive, not just noisy) but never budged off ~0% vs greedy despite that.
A richer diagnostic (`training/diagnose_vs_greedy.py`) on the update-450 checkpoint showed
this isn't "loses by a bit" — it's a near-shutout: **policy scores 0.46 avg (max ever seen:
5) vs greedy's 28.6**, while the same policy scores 13.6 vs random. Sampling instead of
argmax didn't change the picture.

**The critical control: is greedy actually a crushing strategy, or just crushing weak
opponents?** Ran `training/greedy_baseline.py` (no learned policy at all):

| matchup | seat-0 win / draw | scores |
|---|---|---|
| greedy vs greedy | 44.5% / 6.2% | 12.1 vs 12.3 |
| greedy vs random | 100% / 0% | 28.1 vs 1.0 |
| random vs random | 51.8% / 4.9% | 7.3 vs 6.5 |

**Greedy vs greedy is close to even** (not a runaway) — greedy is not some overwhelming
strategy, it ties itself. This matters for interpreting the self-play result: it means the
self-play policy's near-zero score against greedy is NOT simply "greedy is unbeatable," it's
that the self-play policy apparently never learned even basic greedy-level competence
("reliably take the best available score"), despite outperforming random.

### 8.6 Two competing explanations, and the decisive experiment (in progress)

**H1 (shallow ceiling):** the game's strategic ceiling really is close to greedy — nothing
beats it by much (consistent with §7.3's denial-vs-greedy null result, and with the
greedy-vs-greedy near-tie above).

**H2 (self-play stalled below basic competence, not a ceiling finding at all):** self-play's
two seats are the *same* evolving policy, so it only ever gets pressure from an
equally-mediocre copy of itself — there's no gradient forcing it to become as mechanically
sharp as "always grab the best available score." This is a known self-play failure mode
(the reason e.g. AlphaStar's training mixed in fixed scripted opponents and past-checkpoint
"exploiters" alongside pure self-play rather than relying on self-play alone). The
greedy-vs-greedy near-tie is evidence favoring H2 over H1: if greedy were simply dominant,
it wouldn't tie itself — so the self-play policy's near-shutout looks more like "never
reached greedy's baseline competence" than "hit a real ceiling at greedy's level."

**These two explanations make different, falsifiable predictions for training directly
against a fixed greedy opponent** (not self-play) — which is the experiment now in progress
(`training/train_vs_greedy.py`, not yet run to conclusion as of this writing):

- Converges to **beating greedy** by a real margin → H1 is false, real exploitable depth
  exists that self-play simply never found.
- Converges to **matching greedy** (something near the ~45%/49%/6% split greedy gets against
  itself) → consistent with H1, but a much stronger result than what's in hand now, because
  the policy would have actually reached competence first, ruling out H2 as a confound.
- **Still can't get off zero** even head-to-head against the exact opponent it's scored
  against → the strongest possible version of the shallow-ceiling finding, since it rules
  out H2 (distribution shift / self-play mediocrity) entirely.

**Caveat to check if it succeeds:** if training against greedy finds a large, repeatable
exploit, verify it isn't just abuse of a quirk in greedy's deterministic tie-breaking rule
(`agents.py`/`stage2_screen.py`'s `choose_move`: first-strict-greater-value wins, i.e. a
fixed iteration-order tie-break) rather than a genuine strategic finding — a policy trained
against one fixed deterministic opponent can in principle overfit to exploit its exact
tie-breaking convention specifically, which wouldn't generalize to "beats good play in
general."

### 8.7 What's next in Stage 3 (session 3 — superseded by §8.9, kept for history)

1. ~~Finish the train-vs-greedy experiment~~ — done, see §8.8.
2. ~~Read the result against the three-way split in §8.6~~ — done, see §8.8: it landed in
   between "matching greedy" and "still zero," plateauing at a real-but-modest ~5-9% win rate.
   That result **alone** would still have been ambiguous about H1 vs H2 — §8.9's lookahead-bot
   experiment is what actually resolved it.
3. ~~If shallow ceiling holds up under this stronger test too~~ — it did NOT hold up. See §8.9:
   shallow ceiling is refuted, decisively, by a completely different (non-RL) method.
4. Build the `hex_truchet_public` variant — still open, now lower priority than §8.9's new
   next-steps list (§8.10).

### 8.8 Train-vs-greedy RL result: real improvement, but plateaus far short of what's achievable

`training/train_vs_greedy.py`: the learner plays one seat (re-randomized every episode),
fixed greedy plays the other, only the learner's own action steps get PPO gradients (steps
where greedy acted are real environment transitions but were never sampled from the policy,
so training on them would be invalid). Two runs, ~450 updates total, warm-started from the
self-play checkpoint after a cold-start attempt failed outright (see below).

**Cold start failed a specific, diagnosable way.** Training from scratch directly against
greedy never budged off 0% win rate AND was losing ground against random too (44%→41%→38%
win vs random over 60 updates) — every episode returns a strongly negative margin regardless
of move quality, so PPO's advantage signal had almost nothing to differentiate on. This is
the classic "opponent too strong from a cold start" failure mode, not evidence about the
game. **Fix:** warm-start from the self-play checkpoint (which already reliably beat random)
instead of training from scratch. This is the general lesson, independent of what came next:
when fine-tuning against a much stronger fixed opponent, start from a competent policy, not
random weights.

**Warm-started result: real, monotonic-ish improvement, but it plateaus.**

| updates (cumulative) | win vs greedy | margin vs greedy | win vs random |
|---|---|---|---|
| 20 | 0% | -26.6 | 85% |
| 60 | 0% | -20.4 | 76% |
| 120 | 0.8% | -14.2 | 81% |
| 150 (run 1 end) | 5.5% | -12.8 | 70% |
| 175 | 4.7% | -12.1 | 67% |
| 225 | 8.6% | -9.5 | 66% |
| 300 | 3.9%-7.8%, oscillating | -9.3 to -9.8, flat | 59-65% |

So: real early improvement (0%→~8%), then a plateau in a noisy 4-9% band with margin stuck
around -9.5 for the last ~150 updates. **This alone is genuinely ambiguous** — it's
consistent with either "found a small amount of real depth and hit a hard ceiling near
greedy" or "the RL setup itself (sparse terminal reward, flat-MLP observation, small
network) is the bottleneck, not the game." §8.9 resolves which.

A rich diagnostic on the plateaued checkpoint (`training/diagnose_vs_greedy.py`) showed
something worth keeping in mind for any future RL work on this game: **most of the margin
gain came from suppressing greedy's OWN score (28.6 -> 12.8), not from scoring more itself**
(0.46 -> 3.83). The learner seems to have found something like denial/blocking behavior —
interesting since Stage 2's hand-crafted 1-ply `denial` heuristic (§7.3) found *no* edge over
greedy at all. A flat RL policy stumbled onto a weak version of the thing purpose-built
lookahead does overwhelmingly well (§8.9).

### 8.9 The lookahead-bot experiment: shallow ceiling is FALSE, decisively

**Prompted by a good question:** with RL plateaued at ~5-9%, the user asked whether
hand-designing more bot behaviors would be more productive than continuing to fight the RL
training. It was -- this is the finding that actually settled the strategic-depth question.

**`training/lookahead_bot.py`** -- NOT a learned agent, a deterministic search: at each of
its own turns, take the top-K (K=8) legal candidate moves by immediate area gain (same
candidate generation as greedy), then for EACH candidate simulate the REST of the game with
**both** sides playing plain greedy from that point on, and keep whichever candidate produced
the best actual final margin. This is a "rollout algorithm" (Bertsekas terminology) / one
step of policy improvement over the greedy base policy -- cheap because the env is small and
fast, and it evaluates a TRUE lookahead outcome rather than a hand-tuned 1-ply proxy (unlike
denial, §7.3). No training, no gradients, fully deterministic given a seed.

**Result, n=50 games (seat rotated every game to cancel order effects, per §7.4):**

```
win 1.000  draw 0.000  scores 27.98 / 5.34  margin +22.64
```

**50 out of 50.** At that sample size a 100% win rate against a true win probability below
~90% would be a sub-1%-likelihood event -- this is not noise, it is a real, overwhelming
result. Compare directly to the RL-vs-greedy plateau (§8.8): ~5-9% win rate, margin -9.5,
after ~450 updates of gradient training. A completely non-learned, deterministic search
beats greedy by MORE than RL training could find evidence FOR, let alone achieve.

**This settles the H1/H2 question from §8.6 in H2's favor, unambiguously:**

- **H1 (shallow ceiling) is FALSE.** Greedy is not close to a real ceiling for this game --
  there is large, findable exploitable structure above it (average score margin of nearly 5x:
  28 vs 5).
- **Every prior "greedy looks hard to beat" result was a failure of the METHOD trying, not
  evidence about the GAME:** the 1-ply `denial` heuristic (§7.3) wasn't deep enough to find
  it; self-play (§8.5) got stuck in mutual mediocrity before ever reaching greedy-level
  competence; direct RL-vs-greedy (§8.8) plateaued because sparse terminal reward + a flat,
  spatially-blind MLP observation made credit assignment too hard to find it either. A dead
  simple 8-candidate rollout search found it immediately.

**One important, deliberately-disclosed caveat -- this does NOT mean the question is closed
in every sense:** `lookahead_bot.py` uses **full information** during its rollout simulation
-- it "sees" the opponent's exact hand, which the real private-hand rule (spec.md, this
package's whole premise) would not allow a human or a private-hand-respecting policy to do.
So the +22.6 margin is an **upper bound on exploitability under perfect information**, not
proof that a fair, information-respecting agent can achieve the same. What IS proven,
unconditionally: real exploitable structure exists in this game (less information can only
make exploiting it harder, never easier) -- the shallow-ceiling worry is dead either way, but
"how much of this a real player/policy can access under the private-hand rule" is now the
open question, not "does depth exist at all."

Also worth noting for anyone reusing `lookahead_bot.py`: at K=8 it costs **~29s/game** on
CPU (the K-way branch-and-rollout is expensive) -- far too slow for interactive play (Godot)
or as an in-the-loop RL training opponent without significant optimization (e.g. batching the
search across many training envs at once, the way `greedy_action` was optimized in
`train_selfplay.py`, or reducing K). It's a research tool for settling "does depth exist",
not (yet) a deployable bot.

### 8.10 What's next in Stage 3 (supersedes §8.7)

The question has changed from "does depth exist beyond greedy" (settled: yes) to "how do we
build something that reaches it efficiently and fairly." In rough priority order:

1. **Stop/deprioritize the current train-vs-greedy RL run as configured** -- it plateaued
   ~150 updates ago (§8.8) and there's no reason to expect more of the same setup to close a
   gap this large. Don't just add more updates; change the setup (next items).
2. **Distill the lookahead bot via imitation learning.** It's slow (§8.9) but strong and
   fully deterministic -- generate a dataset of (observation, lookahead-bot-action) pairs and
   train a fast policy net via supervised cross-entropy (behavior cloning) to imitate it, then
   optionally RL-fine-tune from that much stronger starting point. This is the standard
   bootstrap-from-a-strong-teacher pipeline (e.g. AlphaGo's supervised pretraining before
   self-play) and should be far more sample-efficient than continuing pure sparse-reward RL
   from scratch/self-play. Gives a FAST net (needed for Godot, §8.9's speed caveat) that
   approximates a genuinely strong player.
3. **If more direct RL is still wanted:** the concrete, disclosed levers from §8.8/user
   discussion, in expected-impact order: (a) reward shaping -- add a small dense per-step
   term (e.g. area gained that turn) alongside the terminal margin, directly attacking the
   sparse credit-assignment problem; (b) a spatially-aware observation/architecture (the
   current flat 74-dim board encoding + plain MLP has no adjacency/loop-topology structure,
   a lot to ask for a game that's fundamentally about hex-adjacency and loop-closure) --
   likely the more impactful of the two but more implementation work; (c) more
   compute/network capacity, only after (a) and (b), since it's not a design fix.
4. **The private-information question from §8.9 is now the interesting open one:** can a
   private-hand-respecting agent (human, RL policy, or a modified search that reasons under
   hand uncertainty rather than cheating) access a meaningful fraction of the +22.6 margin the
   full-information lookahead bot found, or does most of that edge require knowing the
   opponent's exact hand? Nothing has tested this yet.
5. Build the `hex_truchet_public` variant (§8.1) and compare -- now a more informative
   question given §8.9: does public-hand play let a search/RL agent close MORE of the gap to
   the full-information lookahead bot's +22.6, since less has to be inferred/guessed?
6. Godot (`viz/GODOT_GAME_PLAN.md`) needs a bot roster decision informed by this section --
   likely greedy (fast, weak/fair baseline) + a distilled-from-lookahead net (§2, fast, strong)
   as the two bot difficulty tiers, once §2 exists. The raw `lookahead_bot.py` itself is too
   slow (~29s/move) to wire in directly.

### 8.11 Bot roster settled: pure search, no ML needed (supersedes §8.10 #2 and #6)

**§8.10 #2 recommended distilling the lookahead bot into a fast net, on the premise that the
search itself was too slow to deploy (~29s/move). That premise was wrong** -- it came from
measuring only the most expensive configuration (K=8, rolling out to game end, on CPU while
two other jobs competed for it). Adding a `depth` knob and sweeping the cheap end of the
space shows the search is *already* fast enough for interactive play, so **no distillation,
no neural net, and no ML runtime is needed for a strong opponent.**

**The generalized bot** (`training/lookahead_bot.py::lookahead_action`, presets in
`training/bots.py`) has two knobs:
- `K` -- candidate first-moves tried per turn, ranked by immediate area gain. `K=1` is
  exactly greedy. Dominant strength and cost lever, roughly linear in both.
- `depth` -- plies rolled out (greedy on both sides) before scoring by margin-so-far.
  `depth=0` rolls to game end. Truncating mainly saves *early*-game cost (a full rollout at
  t=4 is ~33 plies, at t=30 only ~7).

**Measured vs plain greedy** (`training/sweep_bots.py`, seats rotated per §7.4; s/move is
Python-on-CPU, a pessimistic proxy for a GDScript port):

| K | depth | win vs greedy | margin | s/move |
|---|---|---|---|---|
| 1 | (n/a) | 0.467 (is greedy) | +0.00 | 0.009 |
| 2 | 4 | 0.900 | +9.10 | 0.051 |
| 3 | 2 | **0.933** | +10.20 | **0.030** |
| 3 | 4 | 0.900 | +9.67 | 0.072 |
| 3 | 8 | 0.900 | +12.43 | 0.140 |
| 4 | 4 | 0.933 | +10.23 | 0.156 |
| 5 | 4 | 0.933 | +10.23 | 0.168 |
| 5 | 8 | 0.900 | **+13.03** | 0.368 |
| 6 | 6 | 0.800 | +8.43 | 0.294 |
| 8 | 0 | 1.000 (n=50) | +22.64 | ~29 |

**K=3/depth=2 wins 93% of games against greedy at 30 ms/move** -- ~1000x cheaper than the
K=8 config while keeping most of the strength. Comfortably interactive; it is the
recommended default opponent. K=8/depth=0 stays useful offline (showcase replay packs) but
must not be wired into live play.

**Read that table for magnitudes, not for ranking.** At n=30 the standard error on a win
rate near 0.9 is ~5.5 points, so every search config from 0.80 to 0.93 is statistically
indistinguishable -- (K=6,depth=6) scoring *lowest* while costing 10x `medium` is almost
certainly noise, not a real inversion. The one solid, enormous difference is greedy (0.467)
vs any-search (0.80-0.93). **Do not re-tune presets on this ordering without a larger
sample** -- §7.4 is the cautionary tale for exactly this metric at exactly this sample size.

**Three findings that do survive the noise:**
- **Depth is not where the strength is.** (K=3,depth=2) matches or beats every deeper config
  at a fraction of the cost. The value comes from *considering several candidates*, not from
  simulating far ahead -- greedy's weakness is myopia about alternatives, not a short
  horizon. Raise K before raising depth. Depth does buy *margin* (the less-noisy metric:
  +12.4 at depth 8 vs +10.2 at depth 2) -- deeper search wins by more, not more often.
- **There is no strength left to buy above `medium` in this range.** Everything from K=2 to
  K=6 saturates; spending 10x the compute changes nothing measurable.
- **The difficulty ladder needs handicapping, not more search.** Since every config saturates
  at ~90%, tiers differ in how badly they beat a human, not whether they do. A genuinely
  *competitive* mid-tier probably needs explicit handicapping (K=2, occasionally taking the
  2nd-best candidate, or some probability of just playing the greedy move) -- **untested, and
  the most important open design question for the vs-bot mode.**

**Consequences:**
- The distillation pipeline is **shelved, not deleted** -- `generate_lookahead_data.py` and
  `lookahead_action(..., return_value=True)` (which returns a free Monte-Carlo value target)
  are written and working if a strong-*and-instant* bot is ever needed (web/mobile export).
  Reach for it only if profiling says the search is too slow; it is a performance
  optimization, not a capability gap.
- The Godot bot roster is now `random` / `easy`(=greedy) / `medium`(K=3,d=2) /
  `hard`(K=3,d=8) / `expert`(K=8,d=0, offline only), all pure search. Nothing to export
  from Python; difficulty is a knob, not a checkpoint. `training/bots.py::PRESETS` is the
  shared definition to keep the GDScript port in sync with.
- The trained RL policies (`training/policy_vs_greedy_warmstart*.pt`, ~8% win rate vs greedy)
  are **not** competitive with even the cheap search configs and should not ship as the
  game's opponent. They remain interesting as a research artifact (§8.8's finding that the
  policy learned denial-ish behavior) and as a warm-start for any future RL work.

### 8.12 Game-design requests to carry into Stage 4 (from session 4)

Two requests came up while settling the bot roster; both are recorded in
`viz/GODOT_GAME_PLAN.md` §1.2 and §5:

1. **Free-placement mode.** Offer placing tiles on *any* empty cell, not just the adjacency
   frontier, as a selectable rule. This is not new logic -- `legal_cells_free` already exists
   in `agents.py` and was screened in Stage 2 (§7). It changes the feel substantially (boards
   develop in disconnected clusters) and widens the early game.
2. **Support all 5 tile types.** The engine already supports all five canonical tiles
   (`_hexcore.canonical_tiles()`); the 2-tile restriction is purely the Stage 0 *deck* choice
   (tile 0 : tile 2 = 1:2), not an engine limit. Making `tile_types`/`deck_counts`
   configurable is cheap and should be built into the GDScript `GameState` from the start.
   **But shipping a 5-tile deck is a design question, not just a config flag** -- Stage 0
   found deck composition is the dominant lever on loop-closure rate, and a *uniform* 5-tile
   deck failed the original gate badly (0.67 loops/board, 51% of boards closing zero loops,
   §2.2). Any 5-tile mode needs its own deck-ratio tuning pass (re-run the Stage 0 sweep for
   the chosen ratio) before it can be called a good game rather than just a supported one.

The bots need no changes for either: the search enumerates whatever legal actions the state
reports, so it is placement-rule and tile-type agnostic. Only the Python caching fast-path
(`train_selfplay._ROTATION_CLASS_REP`, memoizing distinct rotations for tiles 0 and 2) is
2-tile specific; a port should compute that table for all configured types at startup.
