class_name GameState
extends RefCounted

## Headless Hex Truchet rules engine: legality, placement, scoring, deck/hand.
##
## Ported from hex_truchet/spec.md. Contains NO engine or UI dependencies --
## no nodes, no SceneTree -- so it can be unit-tested headlessly and reused
## directly inside the bot search (which clones it thousands of times).
##
## Deliberately MORE general than the RL env spec, which locks one fixed
## configuration because simulacrum bakes one behaviour per package. Run in the
## spec's exact configuration (tile types [0,2] at 12:25, adjacency-required,
## 2 players, hand 3) it must match the Python reference step for step -- that
## is what tests/test_rules.gd checks. The extra options are supersets.

# --- configuration (fixed for a game's lifetime) ---
var tile_types: PackedInt32Array = PackedInt32Array([0, 2])
var deck_counts: PackedInt32Array = PackedInt32Array([12, 25])  # parallel to tile_types
var free_placement: bool = false
var n_players: int = 2
var hand_size: int = 3

# --- state ---
var board_tile: PackedInt32Array = PackedInt32Array()  # -1 empty, else tile type
var board_rot: PackedInt32Array = PackedInt32Array()
var hands: Array[PackedInt32Array] = []
var scores: PackedInt32Array = PackedInt32Array()
var current_player: int = 0
var t: int = 0
var deck: PackedInt32Array = PackedInt32Array()

# --- shared services ---
var geo: HexGeometry
var loop_finder: LoopFinder

## Loop ownership: canonical loop key -> player who first closed it. Purely
## presentational (the renderer tints loops by owner); scoring never reads it.
var loop_owner: Dictionary = {}


func setup(geometry: HexGeometry, finder: LoopFinder, rng_seed: int = 0) -> void:
	geo = geometry
	loop_finder = finder

	board_tile.resize(geo.n_cells)
	board_tile.fill(-1)
	board_rot.resize(geo.n_cells)
	board_rot.fill(0)

	scores.resize(n_players)
	scores.fill(0)

	_build_deck(rng_seed)

	hands.clear()
	for p in n_players:
		var h := PackedInt32Array()
		for i in hand_size:
			if deck.size() > 0:
				h.append(_draw())
		hands.append(h)

	current_player = 0
	t = 0
	loop_owner.clear()


func _build_deck(rng_seed: int) -> void:
	deck = PackedInt32Array()
	for i in tile_types.size():
		for n in deck_counts[i]:
			deck.append(tile_types[i])
	# Seeded RNG instance, never the global randi(), so replays are exact.
	var rng := RandomNumberGenerator.new()
	rng.seed = rng_seed
	# Fisher-Yates
	for i in range(deck.size() - 1, 0, -1):
		var j: int = rng.randi_range(0, i)
		var tmp: int = deck[i]
		deck[i] = deck[j]
		deck[j] = tmp


func _draw() -> int:
	var v: int = deck[0]
	deck.remove_at(0)
	return v


## Load a fixed deck order (for replaying a pack exactly).
func set_deck(order: PackedInt32Array) -> void:
	deck = order.duplicate()
	hands.clear()
	for p in n_players:
		var h := PackedInt32Array()
		for i in hand_size:
			if deck.size() > 0:
				h.append(_draw())
		hands.append(h)


func is_terminal() -> bool:
	return t >= geo.n_cells


## Cells where a tile may legally be placed right now.
## Adjacency-required (spec default): empty cells touching an occupied one,
## or every cell when the board is empty. Free placement: every empty cell.
func legal_cells() -> PackedInt32Array:
	var out := PackedInt32Array()
	var any_placed: bool = t > 0
	for i in geo.n_cells:
		if board_tile[i] != -1:
			continue
		if free_placement or not any_placed:
			out.append(i)
			continue
		for e in HexGeometry.N_EDGES:
			var nb: int = geo.neighbors[i][e]
			if nb >= 0 and board_tile[nb] != -1:
				out.append(i)
				break
	return out


## All legal (hand_slot, cell, rotation) triples for the acting player.
## Rotations are deduplicated to visually-distinct ones per tile type, so the
## bot never wastes search on identical boards and the UI never appears stuck.
func legal_actions() -> Array[Vector3i]:
	var out: Array[Vector3i] = []
	var hand: PackedInt32Array = hands[current_player]
	var cells: PackedInt32Array = legal_cells()
	for slot in hand.size():
		var tile: int = hand[slot]
		if tile < 0:
			continue
		for cell in cells:
			for rot in geo.distinct_rotations[tile]:
				out.append(Vector3i(slot, cell, rot))
	return out


func is_legal(slot: int, cell: int, _rotation: int) -> bool:
	var hand: PackedInt32Array = hands[current_player]
	if slot < 0 or slot >= hand.size():
		return false
	if cell < 0 or cell >= geo.n_cells or board_tile[cell] != -1:
		return false
	if free_placement or t == 0:
		return true
	for e in HexGeometry.N_EDGES:
		var nb: int = geo.neighbors[cell][e]
		if nb >= 0 and board_tile[nb] != -1:
			return true
	return false


## Place a tile and return the area gained.
##
## Score = total enclosed loop area AFTER minus BEFORE. This works because a
## closed loop is sealed and can never change once formed, so the delta is
## exactly the area of loops newly closed by this placement -- no need to diff
## loop identities.
func apply(slot: int, cell: int, rotation: int) -> int:
	var hand: PackedInt32Array = hands[current_player]
	var tile: int = hand[slot]

	var area_before: int = loop_finder.total_loop_area(board_tile, board_rot)
	board_tile[cell] = tile
	board_rot[cell] = rotation
	var area_after: int = loop_finder.total_loop_area(board_tile, board_rot)
	var gained: int = area_after - area_before

	scores[current_player] += gained

	# Remove the played tile, then refill from the deck if any remain.
	var new_hand := PackedInt32Array()
	for i in hand.size():
		if i != slot:
			new_hand.append(hand[i])
	if deck.size() > 0:
		new_hand.append(_draw())
	hands[current_player] = new_hand

	_record_loop_owners(current_player)

	current_player = (current_player + 1) % n_players
	t += 1
	return gained


## Tag any newly-closed loop with the player who closed it (for rendering).
func _record_loop_owners(player: int) -> void:
	for loop: Dictionary in loop_finder.find_loops(board_tile, board_rot):
		var key: String = str(loop["arcs"])
		if not loop_owner.has(key):
			loop_owner[key] = player


func loops() -> Array:
	var out: Array = loop_finder.find_loops(board_tile, board_rot)
	for loop: Dictionary in out:
		loop["owner"] = loop_owner.get(str(loop["arcs"]), -1)
	return out


## Deep copy, for bot search. Shares the immutable geometry/loop-finder
## services but duplicates every mutable array -- a shallow copy here would
## let a rollout corrupt the real game.
func clone() -> GameState:
	var c := GameState.new()
	c.geo = geo
	c.loop_finder = loop_finder
	c.tile_types = tile_types
	c.deck_counts = deck_counts
	c.free_placement = free_placement
	c.n_players = n_players
	c.hand_size = hand_size
	c.board_tile = board_tile.duplicate()
	c.board_rot = board_rot.duplicate()
	c.scores = scores.duplicate()
	c.deck = deck.duplicate()
	c.hands = []
	for h in hands:
		c.hands.append(h.duplicate())
	c.current_player = current_player
	c.t = t
	# loop_owner is presentation-only; rollouts don't need it.
	return c


func winner() -> int:
	## -1 on a draw, else the player index with the highest score.
	var best: int = 0
	var tied: bool = false
	for p in range(1, n_players):
		if scores[p] > scores[best]:
			best = p
			tied = false
		elif scores[p] == scores[best]:
			tied = true
	return -1 if tied else best
