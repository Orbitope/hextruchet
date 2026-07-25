extends Node

## Theme tokens, mirroring the validated web viewer's palette.
##
## Loop colours are keyed by OWNER: warm shades for player 0, cool for player 1,
## with distinct shades inside each family. This was a deliberate fix -- a
## per-loop rainbow looked nice but threw away who scored what, which is the
## actual game information.

var dark_mode: bool = true

const DARK := {
	"bg": Color("0f1116"),
	"panel": Color("171a22"),
	"panel2": Color("1e222c"),
	"hair": Color("242a36"),
	"ink": Color("d7dae3"),
	"muted": Color("828a99"),
	"faint": Color("565e6d"),
	"cell": Color("1a1e28"),
	# Distinctly lighter than the cell fill: at low contrast the 37 hexes read
	# as one dark blob rather than a grid.
	"cell_line": Color("3d4757"),
	"p0": Color("e5a94e"),
	"p1": Color("4ec9c9"),
	"legal": Color("3a4456"),
	"ghost": Color("ffffff"),
}

const LIGHT := {
	"bg": Color("eef0ec"),
	"panel": Color("ffffff"),
	"panel2": Color("f6f7f4"),
	"hair": Color("e2e5e0"),
	"ink": Color("242a30"),
	"muted": Color("5f6771"),
	"faint": Color("9aa0a8"),
	"cell": Color("f7f8f5"),
	"cell_line": Color("e2e6df"),
	"p0": Color("bd7716"),
	"p1": Color("1a8f8f"),
	"legal": Color("cfd6cb"),
	"ghost": Color("333333"),
}

## Warm ramp = player 0, cool ramp = player 1.
const LOOP_RAMP_DARK := [
	[Color("e5a94e"), Color("ef8f5b"), Color("e0587a"), Color("d9c24a")],
	[Color("4ec9c9"), Color("7c9cff"), Color("5ecb8a"), Color("59b6d6")],
]

const LOOP_RAMP_LIGHT := [
	[Color("c67f18"), Color("d1622f"), Color("cf3f63"), Color("b08a12")],
	[Color("159090"), Color("4d6fe0"), Color("2f9e63"), Color("2f83a8")],
]


func c(key: String) -> Color:
	var table: Dictionary = DARK if dark_mode else LIGHT
	return table.get(key, Color.MAGENTA)


func player_color(player: int) -> Color:
	return c("p0") if player == 0 else c("p1")


## Colour for the k-th loop belonging to `owner`. Falls back to a neutral tint
## for loops with no recorded owner (only possible when replaying old packs).
func loop_color(loop_owner_id: int, index: int) -> Color:
	if loop_owner_id < 0:
		return c("muted")
	var ramps: Array = LOOP_RAMP_DARK if dark_mode else LOOP_RAMP_LIGHT
	var ramp: Array = ramps[loop_owner_id % ramps.size()]
	return ramp[index % ramp.size()]
