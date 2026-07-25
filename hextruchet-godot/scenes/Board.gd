class_name BoardView
extends Node2D

## Draws the board: hex cells, arcs, closed loops and their enclosed area,
## plus interactive affordances (legal-cell highlight, ghost preview).
##
## Redraws only when something changes -- the board is static between moves,
## so there is no per-frame work.

signal cell_clicked(cell_idx: int)
signal cell_hovered(cell_idx: int)

var geo: HexGeometry
var state: GameState

## Interaction state, set by Main.
var legal_cells: PackedInt32Array = PackedInt32Array()
var hover_cell: int = -1
var ghost_tile: int = -1
var ghost_rotation: int = 0
var show_ghost: bool = false
var last_placed_cell: int = -1
var interactive: bool = true

## Replay override: when non-empty, draw this instead of `state`.
var replay_tile: PackedInt32Array = PackedInt32Array()
var replay_rot: PackedInt32Array = PackedInt32Array()
var replay_loops: Array = []
var use_replay: bool = false

## 0..1 draw-on progress for the most recent placement.
var place_anim: float = 1.0


func configure(geometry: HexGeometry) -> void:
	geo = geometry
	queue_redraw()


func set_state(s: GameState) -> void:
	state = s
	use_replay = false
	queue_redraw()


func set_replay_frame(tiles: PackedInt32Array, rots: PackedInt32Array,
		loops: Array, placed: int) -> void:
	replay_tile = tiles
	replay_rot = rots
	replay_loops = loops
	last_placed_cell = placed
	use_replay = true
	queue_redraw()


func animate_placement() -> void:
	place_anim = 0.0
	var tw := create_tween()
	tw.tween_property(self, "place_anim", 1.0, 0.28) \
		.set_trans(Tween.TRANS_CUBIC).set_ease(Tween.EASE_OUT)
	tw.tween_callback(queue_redraw)
	# Redraw continuously while the tween runs.
	var stepper := create_tween()
	stepper.set_loops(18)
	stepper.tween_callback(queue_redraw).set_delay(0.016)


func _current_tiles() -> PackedInt32Array:
	if use_replay:
		return replay_tile
	return state.board_tile if state != null else PackedInt32Array()


func _current_rots() -> PackedInt32Array:
	if use_replay:
		return replay_rot
	return state.board_rot if state != null else PackedInt32Array()


func _current_loops() -> Array:
	if use_replay:
		return replay_loops
	return state.loops() if state != null else []


func _draw() -> void:
	if geo == null:
		return
	var tiles: PackedInt32Array = _current_tiles()
	var rots: PackedInt32Array = _current_rots()
	if tiles.is_empty():
		return

	var loops: Array = _current_loops()

	# Map cell -> loop colour, and arc key -> loop colour, so loop membership
	# can be looked up per-primitive while drawing.
	var cell_loop: Dictionary = {}
	var arc_loop: Dictionary = {}
	var per_owner: Dictionary = {}
	for loop: Dictionary in loops:
		var loop_owner_id: int = int(loop.get("owner", -1))
		var idx: int = int(per_owner.get(loop_owner_id, 0))
		per_owner[loop_owner_id] = idx + 1
		var col: Color = Palette.loop_color(loop_owner_id, idx)
		for c: int in loop["cells"]:
			cell_loop[c] = col
		for a: Vector3i in loop["arcs"]:
			arc_loop[_arc_key(a.x, a.y, a.z)] = col

	var legal_set: Dictionary = {}
	for c in legal_cells:
		legal_set[c] = true

	# --- hex cells ---
	for i in geo.n_cells:
		var pts: PackedVector2Array = geo.hex_corners(i)
		var fill: Color = Palette.c("cell")
		if tiles[i] != -1:
			# faint owner tint on occupied cells
			var owner_tint: Color = Palette.c("cell")
			fill = owner_tint
		elif interactive and legal_set.has(i):
			fill = Palette.c("cell").lerp(Palette.c("legal"), 0.55)
		draw_colored_polygon(pts, fill)

		if cell_loop.has(i):
			var lc: Color = cell_loop[i]
			lc.a = 0.17
			draw_colored_polygon(pts, lc)

		var outline: PackedVector2Array = pts.duplicate()
		outline.append(pts[0])
		var line_col: Color = Palette.c("cell_line")
		var line_w: float = 1.0
		if i == last_placed_cell:
			line_col = Palette.c("ink")
			line_w = 2.0
		elif interactive and i == hover_cell and legal_set.has(i):
			line_col = Palette.c("ink")
			line_w = 1.5
		draw_polyline(outline, line_col, line_w, true)

	# --- arcs ---
	for i in geo.n_cells:
		var t: int = tiles[i]
		if t < 0:
			continue
		var fade: float = 1.0
		if i == last_placed_cell:
			fade = place_anim
		for pair: Vector2i in geo.arcs_for(t, rots[i]):
			var key: String = _arc_key(i, pair.x, pair.y)
			var pts: PackedVector2Array = geo.arc_points(i, pair.x, pair.y)
			if fade < 1.0:
				pts = _partial(pts, fade)
			if pts.size() < 2:
				continue
			if arc_loop.has(key):
				draw_polyline(pts, arc_loop[key], 4.2, true)
			else:
				var runc: Color = Palette.c("faint")
				runc.a = 0.75
				draw_polyline(pts, runc, 3.2, true)

	# --- ghost preview ---
	if interactive and show_ghost and ghost_tile >= 0 and hover_cell >= 0 \
			and legal_set.has(hover_cell):
		var g: Color = Palette.c("ghost")
		g.a = 0.5
		for pair: Vector2i in geo.arcs_for(ghost_tile, ghost_rotation):
			var pts: PackedVector2Array = geo.arc_points(hover_cell, pair.x, pair.y)
			draw_polyline(pts, g, 3.6, true)


## Truncate a sampled arc to the first `frac` of its length, for the draw-on
## animation.
func _partial(pts: PackedVector2Array, frac: float) -> PackedVector2Array:
	if frac >= 1.0 or pts.size() < 2:
		return pts
	var keep: int = maxi(2, int(ceil(float(pts.size()) * clampf(frac, 0.0, 1.0))))
	var out := PackedVector2Array()
	for i in mini(keep, pts.size()):
		out.append(pts[i])
	return out


static func _arc_key(cell: int, ea: int, eb: int) -> String:
	return "%d-%d-%d" % [cell, mini(ea, eb), maxi(ea, eb)]


func _unhandled_input(event: InputEvent) -> void:
	if not interactive or geo == null:
		return
	if event is InputEventMouseMotion:
		var c: int = cell_at(get_local_mouse_position())
		if c != hover_cell:
			hover_cell = c
			cell_hovered.emit(c)
			queue_redraw()
	elif event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.pressed and mb.button_index == MOUSE_BUTTON_LEFT:
			var c: int = cell_at(get_local_mouse_position())
			if c >= 0:
				cell_clicked.emit(c)


## Nearest cell to a local point, or -1 if the point is outside every hex.
func cell_at(local_pos: Vector2) -> int:
	if geo == null:
		return -1
	var best: int = -1
	var best_d: float = geo.cell_size * 0.62  # inradius-ish cutoff
	for i in geo.n_cells:
		var d: float = geo.centers[i].distance_to(local_pos)
		if d < best_d:
			best_d = d
			best = i
	return best
