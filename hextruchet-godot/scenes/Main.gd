extends Node2D

## Game shell: start menu, hot-seat play, vs-bot play, and pack replay.
##
## Builds its UI in code rather than from a .tscn. The layout is small and
## mostly dynamic (hand tiles come and go, rails rebuild each move), so
## constructing it here keeps the structure in one readable place and avoids
## hand-authoring scene files.

enum Mode { MENU, HOTSEAT, VS_BOT, REPLAY }

const PACK_DIR := "res://data/packs"

var geo: HexGeometry
var finder: LoopFinder
var state: GameState
var bots: Bots

var mode: Mode = Mode.MENU
var bot_difficulty: Bots.Difficulty = Bots.Difficulty.MEDIUM
var bot_seat: int = 1
var free_placement: bool = false
var use_five_tiles: bool = false
var hide_opponent_hand: bool = true
var bot_thinking: bool = false

# selection state for the human's pending move
var selected_slot: int = -1
var selected_rotation: int = 0

# replay state
var replay_pack: Dictionary = {}
var replay_step: int = 0
var replay_playing: bool = false
var replay_timer: float = 0.0
var replay_speed: float = 0.35

# --- nodes ---
var board: BoardView
var ui: CanvasLayer
var menu_panel: PanelContainer
var hud: Control
var score_label: RichTextLabel
var status_label: Label
var hand_bar: HBoxContainer
var loops_label: RichTextLabel
var transport: HBoxContainer
var replay_label: Label
var end_panel: PanelContainer
var end_label: RichTextLabel


func _ready() -> void:
	geo = HexGeometry.new(3, 60.0)
	finder = LoopFinder.new(geo)
	bots = Bots.new()

	board = BoardView.new()
	board.configure(geo)
	board.name = "Board"
	add_child(board)
	board.cell_clicked.connect(_on_cell_clicked)
	board.cell_hovered.connect(_on_cell_hovered)

	_build_ui()
	_show_menu()
	get_viewport().size_changed.connect(_layout_board)


# ---------------------------------------------------------------- UI building

## Opaque panel background. Godot's default PanelContainer style is
## semi-transparent, which lets the board bleed through dialogue text.
func _solid_panel() -> StyleBoxFlat:
	var sb := StyleBoxFlat.new()
	sb.bg_color = Palette.c("panel")
	sb.border_color = Palette.c("hair")
	sb.set_border_width_all(1)
	sb.set_corner_radius_all(10)
	sb.set_content_margin_all(16)
	return sb


func _build_ui() -> void:
	ui = CanvasLayer.new()
	ui.name = "UI"
	add_child(ui)

	# ---- start menu ----
	# A full-rect CenterContainer does the centring; PRESET_CENTER alone only
	# moves the panel's corner to the middle, which pushes it off-screen.
	var menu_center := CenterContainer.new()
	menu_center.set_anchors_preset(Control.PRESET_FULL_RECT)
	ui.add_child(menu_center)

	menu_panel = PanelContainer.new()
	menu_panel.custom_minimum_size = Vector2(430, 0)
	menu_panel.add_theme_stylebox_override("panel", _solid_panel())
	var mv := VBoxContainer.new()
	mv.add_theme_constant_override("separation", 10)
	menu_panel.add_child(mv)

	var title := Label.new()
	title.text = "HEX TRUCHET"
	title.add_theme_font_size_override("font_size", 28)
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	mv.add_child(title)

	var subtitle := Label.new()
	subtitle.text = "Place tiles, close loops, score the area they enclose."
	subtitle.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	subtitle.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	subtitle.add_theme_color_override("font_color", Palette.c("muted"))
	mv.add_child(subtitle)

	mv.add_child(HSeparator.new())

	var b_hot := Button.new()
	b_hot.text = "Hot-seat  (2 players)"
	b_hot.pressed.connect(func() -> void: _start_game(Mode.HOTSEAT))
	mv.add_child(b_hot)

	var b_bot := Button.new()
	b_bot.text = "Play vs Bot"
	b_bot.pressed.connect(func() -> void: _start_game(Mode.VS_BOT))
	mv.add_child(b_bot)

	var diff_row := HBoxContainer.new()
	var diff_lbl := Label.new()
	diff_lbl.text = "Difficulty"
	diff_lbl.custom_minimum_size = Vector2(110, 0)
	diff_row.add_child(diff_lbl)
	var diff_opt := OptionButton.new()
	diff_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	for d: int in [Bots.Difficulty.RANDOM, Bots.Difficulty.EASY,
			Bots.Difficulty.MEDIUM, Bots.Difficulty.HARD]:
		diff_opt.add_item(Bots.difficulty_name(d), d)
	diff_opt.selected = 2
	diff_opt.item_selected.connect(func(i: int) -> void:
		bot_difficulty = diff_opt.get_item_id(i) as Bots.Difficulty)
	diff_row.add_child(diff_opt)
	mv.add_child(diff_row)

	mv.add_child(HSeparator.new())

	var rules_lbl := Label.new()
	rules_lbl.text = "Rules"
	rules_lbl.add_theme_color_override("font_color", Palette.c("muted"))
	mv.add_child(rules_lbl)

	var cb_free := CheckBox.new()
	cb_free.text = "Free placement (place anywhere)"
	cb_free.toggled.connect(func(on: bool) -> void: free_placement = on)
	mv.add_child(cb_free)

	var cb_tiles := CheckBox.new()
	cb_tiles.text = "All 5 tile types (untuned deck)"
	cb_tiles.toggled.connect(func(on: bool) -> void: use_five_tiles = on)
	mv.add_child(cb_tiles)

	var cb_hide := CheckBox.new()
	cb_hide.text = "Hide opponent's hand"
	cb_hide.button_pressed = true
	cb_hide.toggled.connect(func(on: bool) -> void: hide_opponent_hand = on)
	mv.add_child(cb_hide)

	mv.add_child(HSeparator.new())

	var b_replay := Button.new()
	b_replay.text = "Watch a recorded game"
	b_replay.pressed.connect(_start_replay)
	mv.add_child(b_replay)

	menu_center.add_child(menu_panel)

	# ---- in-game HUD ----
	hud = Control.new()
	hud.set_anchors_preset(Control.PRESET_FULL_RECT)
	hud.mouse_filter = Control.MOUSE_FILTER_IGNORE
	ui.add_child(hud)

	var right_panel := PanelContainer.new()
	right_panel.set_anchors_preset(Control.PRESET_TOP_RIGHT)
	right_panel.position = Vector2(-258, 14)
	right_panel.custom_minimum_size = Vector2(244, 0)
	right_panel.add_theme_stylebox_override("panel", _solid_panel())
	hud.add_child(right_panel)

	var right := VBoxContainer.new()
	right.custom_minimum_size = Vector2(212, 0)
	right.add_theme_constant_override("separation", 8)
	right_panel.add_child(right)

	score_label = RichTextLabel.new()
	score_label.bbcode_enabled = true
	score_label.fit_content = true
	score_label.custom_minimum_size = Vector2(212, 0)
	right.add_child(score_label)

	status_label = Label.new()
	status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	status_label.add_theme_color_override("font_color", Palette.c("muted"))
	right.add_child(status_label)

	loops_label = RichTextLabel.new()
	loops_label.bbcode_enabled = true
	loops_label.fit_content = true
	loops_label.custom_minimum_size = Vector2(212, 0)
	right.add_child(loops_label)

	var menu_btn := Button.new()
	menu_btn.text = "Menu"
	menu_btn.pressed.connect(_show_menu)
	right.add_child(menu_btn)

	# hand bar along the bottom
	hand_bar = HBoxContainer.new()
	hand_bar.set_anchors_preset(Control.PRESET_CENTER_BOTTOM)
	hand_bar.position = Vector2(-170, -74)
	hand_bar.custom_minimum_size = Vector2(340, 58)
	hand_bar.add_theme_constant_override("separation", 8)
	hud.add_child(hand_bar)

	# replay transport
	transport = HBoxContainer.new()
	transport.set_anchors_preset(Control.PRESET_CENTER_BOTTOM)
	transport.position = Vector2(-150, -46)
	transport.add_theme_constant_override("separation", 6)
	hud.add_child(transport)

	var b_prev := Button.new()
	b_prev.text = "<"
	b_prev.pressed.connect(func() -> void: _replay_seek(replay_step - 1))
	transport.add_child(b_prev)

	var b_play := Button.new()
	b_play.text = "Play"
	b_play.pressed.connect(func() -> void:
		replay_playing = not replay_playing
		b_play.text = "Pause" if replay_playing else "Play")
	transport.add_child(b_play)

	var b_next := Button.new()
	b_next.text = ">"
	b_next.pressed.connect(func() -> void: _replay_seek(replay_step + 1))
	transport.add_child(b_next)

	replay_label = Label.new()
	replay_label.custom_minimum_size = Vector2(90, 0)
	replay_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	transport.add_child(replay_label)

	# ---- end-of-game panel ----
	var end_center := CenterContainer.new()
	end_center.set_anchors_preset(Control.PRESET_FULL_RECT)
	end_center.mouse_filter = Control.MOUSE_FILTER_IGNORE
	ui.add_child(end_center)

	end_panel = PanelContainer.new()
	end_panel.custom_minimum_size = Vector2(320, 0)
	# Opaque background: the default panel style is translucent, so the board
	# reads straight through the result text.
	end_panel.add_theme_stylebox_override("panel", _solid_panel())
	var ev := VBoxContainer.new()
	ev.add_theme_constant_override("separation", 8)
	end_panel.add_child(ev)
	end_label = RichTextLabel.new()
	end_label.bbcode_enabled = true
	end_label.fit_content = true
	end_label.custom_minimum_size = Vector2(288, 0)
	ev.add_child(end_label)
	var again := Button.new()
	again.text = "Play again"
	again.pressed.connect(func() -> void: _start_game(mode))
	ev.add_child(again)
	var to_menu := Button.new()
	to_menu.text = "Menu"
	to_menu.pressed.connect(_show_menu)
	ev.add_child(to_menu)
	end_center.add_child(end_panel)


# ------------------------------------------------------------- mode switching

func _show_menu() -> void:
	mode = Mode.MENU
	replay_playing = false
	menu_panel.visible = true
	hud.visible = false
	end_panel.visible = false
	board.visible = false


func _start_game(m: Mode) -> void:
	mode = m
	menu_panel.visible = false
	end_panel.visible = false
	hud.visible = true
	board.visible = true
	transport.visible = false
	hand_bar.visible = true

	state = GameState.new()
	state.free_placement = free_placement
	if use_five_tiles:
		# Roughly even split of all five types. Deliberately untuned -- Stage 0
		# showed deck ratio is the dominant lever on loop-closure rate, and a
		# uniform 5-tile deck scores far fewer loops. Offered as a variant, not
		# as the default.
		state.tile_types = PackedInt32Array([0, 1, 2, 3, 4])
		state.deck_counts = PackedInt32Array([8, 7, 8, 7, 7])
	else:
		state.tile_types = PackedInt32Array([0, 2])
		state.deck_counts = PackedInt32Array([12, 25])
	state.setup(geo, finder, randi())

	selected_slot = 0
	selected_rotation = 0
	board.interactive = true
	board.last_placed_cell = -1
	board.set_state(state)
	_layout_board()
	_refresh()

	if mode == Mode.VS_BOT and state.current_player == bot_seat:
		_take_bot_turn()


func _start_replay() -> void:
	var manifest: Dictionary = _load_json(PACK_DIR + "/manifest.json")
	if manifest.is_empty():
		status_label.text = "No replay packs found."
		return
	var packs: Array = manifest["packs"]
	var pick: Dictionary = packs[randi() % packs.size()]
	replay_pack = _load_json(PACK_DIR + "/" + str(pick["file"]))
	if replay_pack.is_empty():
		return

	mode = Mode.REPLAY
	menu_panel.visible = false
	end_panel.visible = false
	hud.visible = true
	board.visible = true
	transport.visible = true
	hand_bar.visible = false
	board.interactive = false

	replay_step = 0
	replay_playing = true
	_layout_board()
	_replay_seek(0)


func _layout_board() -> void:
	var vp: Vector2 = get_viewport_rect().size
	var avail: Vector2 = vp - Vector2(300, 150)
	var s: float = minf(avail.x / geo.board_size.x, avail.y / geo.board_size.y)
	s = clampf(s, 0.35, 2.0)
	board.scale = Vector2(s, s)
	board.position = Vector2(
		(vp.x - 250.0 - geo.board_size.x * s) * 0.5,
		(vp.y - geo.board_size.y * s) * 0.5 - 20.0)


# ------------------------------------------------------------------ gameplay

func _on_cell_hovered(_c: int) -> void:
	if mode == Mode.HOTSEAT or mode == Mode.VS_BOT:
		board.ghost_tile = _selected_tile()
		board.ghost_rotation = selected_rotation
		board.show_ghost = _selected_tile() >= 0 and not bot_thinking


func _on_cell_clicked(cell: int) -> void:
	if mode != Mode.HOTSEAT and mode != Mode.VS_BOT:
		return
	if bot_thinking or state == null or state.is_terminal():
		return
	if mode == Mode.VS_BOT and state.current_player == bot_seat:
		return
	if selected_slot < 0 or not state.is_legal(selected_slot, cell, selected_rotation):
		return

	_commit(selected_slot, cell, selected_rotation)

	if mode == Mode.VS_BOT and not state.is_terminal() \
			and state.current_player == bot_seat:
		_take_bot_turn()


func _commit(slot: int, cell: int, rot: int) -> void:
	state.apply(slot, cell, rot)
	board.last_placed_cell = cell
	board.animate_placement()
	selected_slot = 0
	selected_rotation = 0
	_refresh()
	if state.is_terminal():
		_show_end()


## Run the bot without freezing the frame.
##
## Even Medium costs ~60ms in GDScript on desktop and Hard ~300ms; on a WASM
## export that is a visibly frozen tab. Yielding a frame first lets the UI
## paint "thinking..." and keeps input responsive.
func _take_bot_turn() -> void:
	bot_thinking = true
	board.show_ghost = false
	status_label.text = "%s is thinking..." % Bots.difficulty_name(bot_difficulty)
	await get_tree().process_frame
	await get_tree().process_frame

	var action: Vector3i = bots.choose(state, bot_difficulty)
	bot_thinking = false
	if action.x < 0:
		return
	_commit(action.x, action.y, action.z)


## Play `moves` bot moves immediately (no animation gating). Used by the
## in-engine smoke test and handy for eyeballing a filled board.
func autoplay(moves: int, difficulty: Bots.Difficulty = Bots.Difficulty.EASY) -> Dictionary:
	if state == null:
		return {"error": "no game"}
	var played: int = 0
	for i in moves:
		if state.is_terminal():
			break
		var a: Vector3i = bots.choose(state, difficulty)
		if a.x < 0:
			break
		state.apply(a.x, a.y, a.z)
		board.last_placed_cell = a.y
		played += 1
	selected_slot = 0
	selected_rotation = 0
	_refresh()
	if state.is_terminal():
		_show_end()
	return {"played": played, "t": state.t, "scores": str(state.scores),
			"loops": state.loops().size()}


func _selected_tile() -> int:
	if state == null or selected_slot < 0:
		return -1
	var hand: PackedInt32Array = state.hands[state.current_player]
	if selected_slot >= hand.size():
		return -1
	return hand[selected_slot]


func _refresh() -> void:
	if state == null:
		return
	board.legal_cells = state.legal_cells()
	board.ghost_tile = _selected_tile()
	board.ghost_rotation = selected_rotation
	board.queue_redraw()
	_refresh_scores()
	_refresh_hand()
	_refresh_loops()


func _refresh_scores() -> void:
	var p0: Color = Palette.player_color(0)
	var p1: Color = Palette.player_color(1)
	var n0: String = "You" if mode == Mode.VS_BOT else "Player A"
	var n1: String = Bots.difficulty_name(bot_difficulty) if mode == Mode.VS_BOT \
			else "Player B"
	var turn: int = state.current_player
	var mark0: String = " [b]<[/b]" if turn == 0 and not state.is_terminal() else ""
	var mark1: String = " [b]<[/b]" if turn == 1 and not state.is_terminal() else ""
	score_label.text = (
		"[color=#%s]%s[/color]  [b]%d[/b]%s\n[color=#%s]%s[/color]  [b]%d[/b]%s\n"
		% [p0.to_html(false), n0, state.scores[0], mark0,
		   p1.to_html(false), n1, state.scores[1], mark1])
	if not bot_thinking:
		status_label.text = "Tile %d of %d placed" % [state.t, geo.n_cells]


func _refresh_hand() -> void:
	for child in hand_bar.get_children():
		child.queue_free()
	if state == null or state.is_terminal():
		return

	var hand: PackedInt32Array = state.hands[state.current_player]
	var conceal: bool = mode == Mode.VS_BOT and state.current_player == bot_seat
	for i in hand.size():
		var b := Button.new()
		b.custom_minimum_size = Vector2(64, 52)
		b.toggle_mode = true
		b.button_pressed = (i == selected_slot)
		if conceal:
			b.text = "?"
		else:
			b.text = _tile_label(hand[i])
		var slot: int = i
		b.pressed.connect(func() -> void:
			selected_slot = slot
			selected_rotation = 0
			_refresh())
		hand_bar.add_child(b)

	var rot := Button.new()
	rot.custom_minimum_size = Vector2(64, 52)
	rot.text = "Rotate\n(R)"
	rot.pressed.connect(_cycle_rotation)
	hand_bar.add_child(rot)


## Cycle only through VISUALLY DISTINCT rotations. Tile 4 has exactly one, so
## naive 0..5 cycling would look like a broken control six presses in a row.
func _cycle_rotation() -> void:
	var tile: int = _selected_tile()
	if tile < 0:
		return
	var distinct: PackedInt32Array = geo.distinct_rotations[tile]
	var idx: int = 0
	for i in distinct.size():
		if distinct[i] == selected_rotation:
			idx = i
			break
	selected_rotation = distinct[(idx + 1) % distinct.size()]
	_refresh()


func _tile_label(t: int) -> String:
	match t:
		0: return "Y"     # three tight turns
		1: return "S"
		2: return "T"     # two turns + straight
		3: return "W"
		4: return "|"     # all straight
		_: return "?"


func _refresh_loops() -> void:
	var loops: Array = state.loops() if mode != Mode.REPLAY else board.replay_loops
	if loops.is_empty():
		loops_label.text = "[color=#%s]no loops yet[/color]" \
				% Palette.c("faint").to_html(false)
		return
	var per_owner: Dictionary = {}
	var lines: Array[String] = ["[b]Loops (%d)[/b]" % loops.size()]
	for loop: Dictionary in loops:
		var loop_owner_id: int = int(loop.get("owner", -1))
		var idx: int = int(per_owner.get(loop_owner_id, 0))
		per_owner[loop_owner_id] = idx + 1
		var col: Color = Palette.loop_color(loop_owner_id, idx)
		var who: String = "A" if loop_owner_id == 0 else ("B" if loop_owner_id == 1 else "-")
		lines.append("[color=#%s]%s[/color] %s  area %d" % [
			col.to_html(false), "*", who, int(loop["area"])])
	loops_label.text = "\n".join(lines)


func _show_end() -> void:
	var w: int = state.winner()
	var head: String
	if w < 0:
		head = "[b]Draw[/b]"
	elif mode == Mode.VS_BOT:
		head = "[b]You win[/b]" if w == 1 - bot_seat else "[b]Bot wins[/b]"
	else:
		head = "[b]Player %s wins[/b]" % ("A" if w == 0 else "B")
	end_label.text = "%s\n\nA %d  -  B %d" % [head, state.scores[0], state.scores[1]]
	end_panel.visible = true


# -------------------------------------------------------------------- replay

func _replay_seek(step: int) -> void:
	if replay_pack.is_empty():
		return
	var steps: Array = replay_pack["steps"]
	replay_step = clampi(step, 0, steps.size())

	var tiles := PackedInt32Array()
	tiles.resize(geo.n_cells)
	tiles.fill(-1)
	var rots := PackedInt32Array()
	rots.resize(geo.n_cells)
	rots.fill(0)

	var loops: Array = []
	var last: int = -1
	for i in replay_step:
		var s: Dictionary = steps[i]
		tiles[int(s["cell"])] = int(s["tile"])
		rots[int(s["cell"])] = int(s["rot"])
		last = int(s["cell"])
	if replay_step > 0:
		loops = _loops_from_pack(steps[replay_step - 1])

	board.set_replay_frame(tiles, rots, loops, last)
	replay_label.text = "%d / %d" % [replay_step, steps.size()]

	var sc: Array = [0, 0]
	if replay_step > 0:
		sc = steps[replay_step - 1]["score"]
	var p0: Color = Palette.player_color(0)
	var p1: Color = Palette.player_color(1)
	score_label.text = ("[color=#%s]Player A[/color]  [b]%d[/b]\n[color=#%s]Player B[/color]  [b]%d[/b]"
		% [p0.to_html(false), int(sc[0]), p1.to_html(false), int(sc[1])])
	var cfg: Dictionary = replay_pack.get("config", {})
	status_label.text = "Replay - %s%s" % [
		str(cfg.get("policy", "?")),
		", free placement" if bool(cfg.get("free_placement", false)) else ""]
	_refresh_loops()


func _loops_from_pack(step: Dictionary) -> Array:
	var out: Array = []
	for l: Dictionary in step["loops"]:
		var arcs: Array[Vector3i] = []
		for a: Array in l["arcs"]:
			arcs.append(Vector3i(int(a[0]), int(a[1]), int(a[2])))
		var cells := PackedInt32Array()
		for c in l["cells"]:
			cells.append(int(c))
		out.append({
			"arcs": arcs,
			"cells": cells,
			"area": int(l["area"]),
			"length": int(l["length"]),
			"owner": int(l.get("owner", -1)),
		})
	return out


func _process(delta: float) -> void:
	if mode != Mode.REPLAY or not replay_playing:
		return
	replay_timer += delta
	if replay_timer < replay_speed:
		return
	replay_timer = 0.0
	var steps: Array = replay_pack.get("steps", [])
	if replay_step >= steps.size():
		replay_playing = false
		return
	_replay_seek(replay_step + 1)


func _unhandled_key_input(event: InputEvent) -> void:
	if not (event is InputEventKey) or not event.pressed:
		return
	var k := event as InputEventKey
	match k.keycode:
		KEY_R:
			if mode == Mode.HOTSEAT or mode == Mode.VS_BOT:
				_cycle_rotation()
		KEY_LEFT:
			if mode == Mode.REPLAY:
				_replay_seek(replay_step - 1)
		KEY_RIGHT:
			if mode == Mode.REPLAY:
				_replay_seek(replay_step + 1)
		KEY_SPACE:
			if mode == Mode.REPLAY:
				replay_playing = not replay_playing
		KEY_ESCAPE:
			_show_menu()


func _load_json(path: String) -> Dictionary:
	if not FileAccess.file_exists(path):
		return {}
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return {}
	var parsed: Variant = JSON.parse_string(f.get_as_text())
	f.close()
	return parsed if typeof(parsed) == TYPE_DICTIONARY else {}
