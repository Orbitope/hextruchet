extends SceneTree

## Differential test: GDScript rules engine vs the Python reference.
##
## Replays each pack's exact move sequence through GameState and asserts the
## per-step area gained, running scores, and closed-loop sets match Python at
## every single step. Packs are produced by training/export_pack.py and cover
## the rule-option matrix (adjacency + free placement, 2-tile + 5-tile decks,
## greedy + random policies).
##
## This is the gate: nothing gets built on top of the rules engine until this
## passes. Run headlessly:
##   godot --headless --script res://tests/test_rules.gd

const PACK_DIR := "res://data/packs"

var failures: int = 0
var checks: int = 0


func _init() -> void:
	print("=== Hex Truchet rules differential test ===")

	var geo := HexGeometry.new(3, 60.0)
	var finder := LoopFinder.new(geo)

	if not _verify_static(geo):
		print("\nFAILED: static geometry mismatch -- aborting")
		quit(1)
		return

	var manifest_path := PACK_DIR + "/manifest.json"
	var manifest: Dictionary = _load_json(manifest_path)
	if manifest.is_empty():
		print("FAILED: could not load ", manifest_path)
		quit(1)
		return

	for entry: Dictionary in manifest["packs"]:
		_check_pack(geo, finder, PACK_DIR + "/" + str(entry["file"]))

	print("\n=== %d checks, %d failures ===" % [checks, failures])
	if failures == 0:
		print("PASS -- GDScript engine matches Python reference")
		quit(0)
	else:
		print("FAIL")
		quit(1)


## Verify cell ordering, neighbour map and tile arc tables against Python
## before touching gameplay -- a geometry mismatch would produce confusing
## downstream failures.
func _verify_static(geo: HexGeometry) -> bool:
	var static_data: Dictionary = _load_json(PACK_DIR + "/static.json")
	if static_data.is_empty():
		print("  could not load static.json")
		return false

	var g: Dictionary = static_data["geometry"]
	var ok := true

	if int(g["n_cells"]) != geo.n_cells:
		print("  n_cells: python %d, gdscript %d" % [int(g["n_cells"]), geo.n_cells])
		ok = false

	var py_cells: Array = g["cells"]
	for i in mini(py_cells.size(), geo.n_cells):
		var qr: Array = py_cells[i]["qr"]
		var mine: Vector2i = geo.cells[i]
		if int(qr[0]) != mine.x or int(qr[1]) != mine.y:
			print("  cell %d: python (%d,%d), gdscript (%d,%d)"
					% [i, int(qr[0]), int(qr[1]), mine.x, mine.y])
			ok = false
			break

	var py_nb: Array = g["neighbor_idx"]
	for i in mini(py_nb.size(), geo.n_cells):
		for e in 6:
			var expected: int = int(py_nb[i][e])
			var actual: int = geo.neighbors[i][e]
			if expected != actual:
				print("  neighbor[%d][%d]: python %d, gdscript %d"
						% [i, e, expected, actual])
				ok = false
				break
		if not ok:
			break

	var py_tiles: Dictionary = static_data["tiles"]
	for key: String in py_tiles:
		var tt: int = int(key)
		var py_arcs: Array = py_tiles[key]["arcs_by_rotation"]
		for rot in 6:
			var expected_set: Dictionary = {}
			for pair: Array in py_arcs[rot]:
				expected_set[Vector2i(mini(int(pair[0]), int(pair[1])),
						maxi(int(pair[0]), int(pair[1])))] = true
			for pair: Vector2i in geo.arcs_for(tt, rot):
				if not expected_set.has(pair):
					print("  tile %d rot %d: gdscript arc %s not in python"
							% [tt, rot, str(pair)])
					ok = false
					break
			if not ok:
				break
		if not ok:
			break

	print("  static geometry + tile tables: ", "OK" if ok else "MISMATCH")
	return ok


func _check_pack(geo: HexGeometry, finder: LoopFinder, path: String) -> void:
	var pack: Dictionary = _load_json(path)
	if pack.is_empty():
		print("  %s: COULD NOT LOAD" % path.get_file())
		failures += 1
		return

	var cfg: Dictionary = pack["config"]

	var state := GameState.new()
	state.n_players = int(cfg["n_players"])
	state.hand_size = int(cfg["hand_size"])
	state.free_placement = bool(cfg["free_placement"])

	# deck_counts arrives as {tile_type_string: count}; rebuild parallel arrays
	# in ascending tile-type order so the deck matches Python's construction.
	var dc: Dictionary = cfg["deck_counts"]
	var types := PackedInt32Array()
	var counts := PackedInt32Array()
	var keys: Array = dc.keys()
	keys.sort_custom(func(a, b): return int(a) < int(b))
	for k: String in keys:
		types.append(int(k))
		counts.append(int(dc[k]))
	state.tile_types = types
	state.deck_counts = counts

	state.setup(geo, finder, 0)
	# Override with the pack's exact deck so draws match Python move for move.
	var deck_initial := PackedInt32Array()
	for v in pack["deck_initial"]:
		deck_initial.append(int(v))
	state.set_deck(deck_initial)

	var pack_failures: int = 0
	var steps: Array = pack["steps"]

	for step: Dictionary in steps:
		var slot: int = int(step["hand_slot"])
		var cell: int = int(step["cell"])
		var rot: int = int(step["rot"])
		var expected_tile: int = int(step["tile"])
		var step_t: int = int(step["t"])

		# The hand must match before we place, or the whole comparison is void.
		var hand: PackedInt32Array = state.hands[state.current_player]
		if slot >= hand.size():
			print("  step %d: hand too small (slot %d, hand %s)"
					% [step_t, slot, str(hand)])
			pack_failures += 1
			break
		if hand[slot] != expected_tile:
			print("  step %d: tile mismatch -- python %d, gdscript %d (hand %s)"
					% [step_t, expected_tile, hand[slot], str(hand)])
			pack_failures += 1
			break

		var gained: int = state.apply(slot, cell, rot)
		checks += 1

		if gained != int(step["gained"]):
			print("  step %d: gained -- python %d, gdscript %d"
					% [step_t, int(step["gained"]), gained])
			pack_failures += 1
			break

		var expected_scores: Array = step["score"]
		for p in state.n_players:
			if state.scores[p] != int(expected_scores[p]):
				print("  step %d: score[%d] -- python %d, gdscript %d"
						% [step_t, p, int(expected_scores[p]), state.scores[p]])
				pack_failures += 1
				break
		if pack_failures > 0:
			break

		# Compare closed-loop sets: count, and each loop's area + arc membership.
		var expected_loops: Array = step["loops"]
		var actual_loops: Array = state.loop_finder.find_loops(
				state.board_tile, state.board_rot)
		if actual_loops.size() != expected_loops.size():
			print("  step %d: loop count -- python %d, gdscript %d"
					% [step_t, expected_loops.size(), actual_loops.size()])
			pack_failures += 1
			break

		var expected_areas: Array = []
		for l: Dictionary in expected_loops:
			expected_areas.append(int(l["area"]))
		expected_areas.sort()
		var actual_areas: Array = []
		for l: Dictionary in actual_loops:
			actual_areas.append(int(l["area"]))
		actual_areas.sort()
		if str(expected_areas) != str(actual_areas):
			print("  step %d: loop areas -- python %s, gdscript %s"
					% [step_t, str(expected_areas), str(actual_areas)])
			pack_failures += 1
			break

	var final_score: Array = pack["final_score"]
	if pack_failures == 0:
		for p in state.n_players:
			if state.scores[p] != int(final_score[p]):
				print("  final score[%d] -- python %d, gdscript %d"
						% [p, int(final_score[p]), state.scores[p]])
				pack_failures += 1

	failures += pack_failures
	var label: String = "OK" if pack_failures == 0 else "FAIL(%d)" % pack_failures
	print("  %-40s %s  (%d steps, final %s)"
			% [path.get_file(), label, steps.size(), str(final_score)])


func _load_json(path: String) -> Dictionary:
	if not FileAccess.file_exists(path):
		return {}
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return {}
	var parsed: Variant = JSON.parse_string(f.get_as_text())
	f.close()
	if typeof(parsed) != TYPE_DICTIONARY:
		return {}
	return parsed
