class_name Bots
extends RefCounted

## Non-ML opponents: one tunable rollout search, exposed as named difficulties.
##
## Ported from training/lookahead_bot.py + training/bots.py. No neural net, no
## weights, no inference runtime -- difficulty is a knob, not a checkpoint.
##
## The algorithm (a "rollout algorithm" / one step of policy improvement over
## the greedy base policy):
##   1. Rank legal actions by immediate area gain -- this ranking IS greedy.
##   2. Take the top K, and for each, play BOTH sides forward with plain greedy
##      for `depth` plies (0 = to game end).
##   3. Keep whichever candidate ends with the best margin for the bot.
##
## Measured in Python vs plain greedy (training/bot_sweep.log, n=30/config):
##   K=1        -> 0.467 win (it IS greedy)      0.009 s/move
##   K=3 d=2    -> 0.933 win, +10.20 margin      0.030 s/move   <- best value
##   K=3 d=8    -> 0.900 win, +12.43 margin      0.140 s/move
##   K=8 d=0    -> 1.000 win, +22.64 margin      ~29 s/move     <- offline only
##
## Two findings that shaped the presets, worth not re-deriving:
##   - Depth is NOT where the strength is. (K=3,d=2) matches or beats deeper
##     configs at a fraction of the cost. Raise K before raising depth.
##   - Win rate saturates ~90% for every search config, so difficulty tiers
##     differ in HOW BADLY they win, not whether they do. `easy` exists to be
##     beatable; see HANDICAP_* below for the human-competitive lever.

enum Difficulty { RANDOM, EASY, MEDIUM, HARD, EXPERT }

## Sentinel for "worse than any real margin". Scores are bounded well under
## this (a full board caps out around 40), and GDScript disallows negative
## shift operands, so this is spelled out rather than computed.
const WORST_MARGIN: int = -1000000

## (K, depth, greedy_slip). greedy_slip is the probability of deliberately
## playing the plain-greedy move instead of the searched one -- the handicap
## axis, since more/less search alone barely moves the win rate.
const PRESETS: Dictionary = {
	Difficulty.RANDOM: {"k": 0, "depth": 0, "slip": 0.0},
	Difficulty.EASY: {"k": 1, "depth": 0, "slip": 0.0},
	Difficulty.MEDIUM: {"k": 3, "depth": 2, "slip": 0.35},
	Difficulty.HARD: {"k": 3, "depth": 8, "slip": 0.0},
	Difficulty.EXPERT: {"k": 8, "depth": 0, "slip": 0.0},
}

const DIFFICULTY_NAMES: Dictionary = {
	Difficulty.RANDOM: "Random",
	Difficulty.EASY: "Easy",
	Difficulty.MEDIUM: "Medium",
	Difficulty.HARD: "Hard",
	Difficulty.EXPERT: "Expert",
}

var rng: RandomNumberGenerator


func _init(seed_value: int = 0) -> void:
	rng = RandomNumberGenerator.new()
	if seed_value != 0:
		rng.seed = seed_value
	else:
		rng.randomize()


## Pick a move for the acting player.
func choose(state: GameState, difficulty: Difficulty) -> Vector3i:
	var cfg: Dictionary = PRESETS[difficulty]
	var k: int = int(cfg["k"])
	var depth: int = int(cfg["depth"])
	var slip: float = float(cfg["slip"])

	var actions: Array[Vector3i] = state.legal_actions()
	if actions.is_empty():
		return Vector3i(-1, -1, -1)

	if k <= 0:
		return actions[rng.randi_range(0, actions.size() - 1)]

	var ranked: Array = _rank_by_gain(state, actions)
	if k == 1 or ranked.size() == 1:
		return _best_or_random(ranked, state)

	# Handicap: sometimes just take the greedy move instead of searching.
	if slip > 0.0 and rng.randf() < slip:
		return _best_or_random(ranked, state)

	var top_k: int = mini(k, ranked.size())
	var best_action: Vector3i = ranked[0]["action"]
	var best_margin: int = WORST_MARGIN
	var seat: int = state.current_player

	for i in top_k:
		var action: Vector3i = ranked[i]["action"]
		var sim: GameState = state.clone()
		sim.apply(action.x, action.y, action.z)
		_rollout(sim, depth)
		var margin: int = _margin_for(sim, seat)
		if margin > best_margin:
			best_margin = margin
			best_action = action

	return best_action


## All legal actions scored by immediate area gain, best first.
## Ties keep enumeration order, matching the Python greedy's tie-break.
func _rank_by_gain(state: GameState, actions: Array[Vector3i]) -> Array:
	# The pre-placement area is identical for every candidate at this position,
	# so compute it once rather than once per candidate (halves the work).
	var area_before: int = state.loop_finder.total_loop_area(
			state.board_tile, state.board_rot)

	var scratch_tile: PackedInt32Array = state.board_tile.duplicate()
	var scratch_rot: PackedInt32Array = state.board_rot.duplicate()

	var scored: Array = []
	var hand: PackedInt32Array = state.hands[state.current_player]
	for action: Vector3i in actions:
		var tile: int = hand[action.x]
		scratch_tile[action.y] = tile
		scratch_rot[action.y] = action.z
		var after: int = state.loop_finder.total_loop_area(scratch_tile, scratch_rot)
		scratch_tile[action.y] = -1
		scratch_rot[action.y] = 0
		scored.append({"action": action, "gain": after - area_before})

	scored.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		return int(a["gain"]) > int(b["gain"]))
	return scored


func _best_or_random(ranked: Array, state: GameState) -> Vector3i:
	# No scoring move available -> play a random legal one, matching the
	# Python greedy's fallback (which prevents index-order degeneracy).
	if ranked.is_empty():
		return Vector3i(-1, -1, -1)
	if int(ranked[0]["gain"]) <= 0:
		var actions: Array[Vector3i] = state.legal_actions()
		return actions[rng.randi_range(0, actions.size() - 1)]
	return ranked[0]["action"]


## Play both sides forward with plain greedy. depth <= 0 means to game end.
func _rollout(sim: GameState, depth: int) -> void:
	var plies: int = 0
	while not sim.is_terminal():
		if depth > 0 and plies >= depth:
			break
		var actions: Array[Vector3i] = sim.legal_actions()
		if actions.is_empty():
			break
		var ranked: Array = _rank_by_gain(sim, actions)
		var pick: Vector3i = _best_or_random(ranked, sim)
		if pick.x < 0:
			break
		sim.apply(pick.x, pick.y, pick.z)
		plies += 1


func _margin_for(sim: GameState, seat: int) -> int:
	var mine: int = sim.scores[seat]
	var best_other: int = WORST_MARGIN
	for p in sim.n_players:
		if p != seat:
			best_other = maxi(best_other, sim.scores[p])
	return mine - best_other


static func difficulty_name(d: Difficulty) -> String:
	return DIFFICULTY_NAMES.get(d, "?")
