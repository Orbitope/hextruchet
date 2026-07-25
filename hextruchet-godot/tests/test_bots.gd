extends SceneTree

## Bot strength + cost check, headless.
##
## Two things this must establish before the bots are wired into the UI:
##   1. Search actually beats plain greedy (the Python result should reproduce).
##   2. Per-move cost is small enough for interactive play. The plan's timings
##      are Python-on-desktop and prove nothing about GDScript; this measures
##      the real thing. (WASM is slower still -- see GODOT_GAME_PLAN 0.1 --
##      but this is the first honest number.)
##
##   godot --headless --script res://tests/test_bots.gd

const GAMES_PER_CONFIG := 6


func _init() -> void:
	print("=== Bot strength + cost (vs plain greedy, seats rotated) ===")

	var geo := HexGeometry.new(3, 60.0)
	var finder := LoopFinder.new(geo)

	print("%-10s %6s %6s %8s %10s" % ["preset", "win", "draw", "margin", "ms/move"])
	print("-".repeat(46))

	for d: Bots.Difficulty in [Bots.Difficulty.EASY, Bots.Difficulty.MEDIUM,
			Bots.Difficulty.HARD]:
		_measure(geo, finder, d)

	print("\nNote: EASY is greedy-vs-greedy, so ~0.5 win is correct, not a bug.")
	quit(0)


func _measure(geo: HexGeometry, finder: LoopFinder, d: Bots.Difficulty) -> void:
	var wins: int = 0
	var draws: int = 0
	var margin_total: int = 0
	var move_us: int = 0
	var move_count: int = 0

	for g in GAMES_PER_CONFIG:
		var bot_seat: int = g % 2  # rotate seats to cancel first-mover effects
		var bots := Bots.new(1234 + g)
		var baseline := Bots.new(9876 + g)

		var state := GameState.new()
		state.setup(geo, finder, 500 + g)

		while not state.is_terminal():
			var action: Vector3i
			if state.current_player == bot_seat:
				var t0: int = Time.get_ticks_usec()
				action = bots.choose(state, d)
				move_us += Time.get_ticks_usec() - t0
				move_count += 1
			else:
				action = baseline.choose(state, Bots.Difficulty.EASY)
			if action.x < 0:
				break
			state.apply(action.x, action.y, action.z)

		var bot_score: int = state.scores[bot_seat]
		var opp_score: int = state.scores[1 - bot_seat]
		margin_total += bot_score - opp_score
		if bot_score > opp_score:
			wins += 1
		elif bot_score == opp_score:
			draws += 1

	var n: float = float(GAMES_PER_CONFIG)
	var ms: float = (float(move_us) / float(maxi(move_count, 1))) / 1000.0
	print("%-10s %6.2f %6.2f %+8.2f %10.1f" % [
		Bots.difficulty_name(d),
		float(wins) / n,
		float(draws) / n,
		float(margin_total) / n,
		ms,
	])
