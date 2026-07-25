class_name HexGeometry
extends RefCounted

## Static hex-board geometry and tile-arc tables.
##
## Ported from hex_truchet/_hexcore.py (geometry.py section). All tables are
## built once at startup and shared. Conventions are locked and MUST match the
## Python reference exactly -- the differential test (tests/test_rules.gd)
## depends on it:
##   - Axial coords (q, r), pointy-top hexes.
##   - Edge indices 0..5; edge i of cell c faces DIRECTIONS[i].
##   - Reciprocity: edge i of A faces B  <=>  edge (i+3)%6 of B faces A.
##   - Rotating a tile by k maps every edge e -> (e + k) % 6.

const DIRECTIONS: Array[Vector2i] = [
	Vector2i(1, 0),
	Vector2i(1, -1),
	Vector2i(0, -1),
	Vector2i(-1, 0),
	Vector2i(-1, 1),
	Vector2i(0, 1),
]

## Canonical tile set: the five rotation-classes of perfect matchings on 6
## edges. Each entry is the rotation-0 arc set as three [edge_a, edge_b] pairs.
## Verified against _hexcore.canonical_tiles().
const BASE_ARCS: Array = [
	[[0, 1], [2, 3], [4, 5]],  # 0: spans (1,1,1) -- three tight turns
	[[0, 1], [2, 4], [3, 5]],  # 1: spans (1,2,2)
	[[0, 1], [2, 5], [3, 4]],  # 2: spans (1,1,3) -- two tight + one straight
	[[0, 2], [1, 4], [3, 5]],  # 3: spans (2,2,3)
	[[0, 3], [1, 4], [2, 5]],  # 4: spans (3,3,3) -- all straight-through
]

const N_TILE_TYPES: int = 5
const N_EDGES: int = 6

var radius: int
var cells: Array[Vector2i] = []              # index -> axial coord
var cell_index: Dictionary = {}              # Vector2i -> index
var n_cells: int
## neighbors[cell_idx][edge] -> neighbor cell index, or -1 if off-board.
var neighbors: Array = []
## arcs_by_rotation[tile_type][rotation] -> Array of 3 Vector2i(edge_a, edge_b).
var arcs_by_rotation: Array = []
## distinct_rotations[tile_type] -> PackedInt32Array of rotations that produce
## visually distinct arc patterns (tile 4 has one; tile 0 has two).
var distinct_rotations: Array[PackedInt32Array] = []

# --- rendering geometry (screen space, y down) ---
var cell_size: float                          # spacing L
var centers: PackedVector2Array = []          # cell_idx -> screen centre
var edge_offset: PackedVector2Array = []      # edge -> centre->edge-midpoint
var board_size: Vector2


func _init(board_radius: int = 3, spacing: float = 60.0) -> void:
	radius = board_radius
	cell_size = spacing
	_build_cells()
	_build_neighbors()
	_build_tile_tables()
	_build_render_geometry()


func _build_cells() -> void:
	# Mirrors _hexcore.hex_board: q ascending, then r ascending. Order is part
	# of the contract -- pack files index cells by position in this list.
	for q in range(-radius, radius + 1):
		var r_lo: int = maxi(-radius, -q - radius)
		var r_hi: int = mini(radius, -q + radius)
		for r in range(r_lo, r_hi + 1):
			var c := Vector2i(q, r)
			cell_index[c] = cells.size()
			cells.append(c)
	n_cells = cells.size()


func _build_neighbors() -> void:
	neighbors.resize(n_cells)
	for i in n_cells:
		var row := PackedInt32Array()
		row.resize(N_EDGES)
		for e in N_EDGES:
			var nb: Vector2i = cells[i] + DIRECTIONS[e]
			row[e] = cell_index.get(nb, -1)
		neighbors[i] = row


func _build_tile_tables() -> void:
	arcs_by_rotation.resize(N_TILE_TYPES)
	distinct_rotations.resize(N_TILE_TYPES)
	for t in N_TILE_TYPES:
		var per_rot: Array = []
		per_rot.resize(N_EDGES)
		var seen: Dictionary = {}
		var distinct := PackedInt32Array()
		for rot in N_EDGES:
			var arcs: Array[Vector2i] = []
			for pair: Array in BASE_ARCS[t]:
				var a: int = (int(pair[0]) + rot) % N_EDGES
				var b: int = (int(pair[1]) + rot) % N_EDGES
				arcs.append(Vector2i(mini(a, b), maxi(a, b)))
			arcs.sort()
			per_rot[rot] = arcs
			# Canonical key so rotations producing an identical arc pattern
			# collapse together (tile 4's six rotations are all the same tile).
			var key: String = str(arcs)
			if not seen.has(key):
				seen[key] = rot
				distinct.append(rot)
		arcs_by_rotation[t] = per_rot
		distinct_rotations[t] = distinct


func _build_render_geometry() -> void:
	# Axial -> pixel for pointy-top hexes, then flipped to Godot's y-down screen
	# space and shifted positive. Matches viz/export_pack.py so packs and the
	# in-engine renderer agree.
	var raw: PackedVector2Array = []
	for c in cells:
		var x: float = cell_size * c.x + 0.5 * cell_size * c.y
		var y: float = -sqrt(3.0) / 2.0 * cell_size * c.y
		raw.append(Vector2(x, -y))  # flip to y-down

	var min_x: float = INF
	var min_y: float = INF
	var max_x: float = -INF
	var max_y: float = -INF
	for p in raw:
		min_x = minf(min_x, p.x)
		min_y = minf(min_y, p.y)
		max_x = maxf(max_x, p.x)
		max_y = maxf(max_y, p.y)

	var pad: float = cell_size
	for p in raw:
		centers.append(Vector2(p.x - min_x + pad, p.y - min_y + pad))
	board_size = Vector2(max_x - min_x + 2.0 * pad, max_y - min_y + 2.0 * pad)

	# Edge midpoint offsets: half the vector to the neighbour across that edge.
	for e in N_EDGES:
		var d: Vector2i = DIRECTIONS[e]
		var x: float = cell_size * d.x + 0.5 * cell_size * d.y
		var y: float = -sqrt(3.0) / 2.0 * cell_size * d.y
		edge_offset.append(Vector2(x, -y) * 0.5)


static func opposite_edge(edge: int) -> int:
	return (edge + 3) % N_EDGES


## Arc span: 1 = adjacent edges, 2 = skip-one, 3 = opposite (straight through).
static func span(ea: int, eb: int) -> int:
	var d: int = absi(ea - eb) % N_EDGES
	return mini(d, N_EDGES - d)


func arcs_for(tile_type: int, rotation: int) -> Array:
	return arcs_by_rotation[tile_type][rotation % N_EDGES]


## Corner points of a cell's hexagon, for drawing.
func hex_corners(cell_idx: int) -> PackedVector2Array:
	var c: Vector2 = centers[cell_idx]
	var r: float = cell_size / sqrt(3.0)
	var pts := PackedVector2Array()
	for k in 6:
		var a: float = deg_to_rad(30.0 + 60.0 * k)
		pts.append(c + Vector2(cos(a), sin(a)) * r)
	return pts


## Tangent-continuous circular arc between two edge-midpoints of one cell,
## SAMPLED INTO POINTS.
##
## Never emit an SVG-style arc command here: it re-derives its own centre from
## the radius+flags and can pick the reflected one, throwing the arc outside
## the cell. That bug cost a debugging round in the web viewer; sampling the
## short way around the true centre is what makes loops read as one continuous
## curve across cell boundaries.
func arc_points(cell_idx: int, ea: int, eb: int, samples: int = 18) -> PackedVector2Array:
	var c: Vector2 = centers[cell_idx]
	var pa: Vector2 = c + edge_offset[ea]
	var pb: Vector2 = c + edge_offset[eb]

	if span(ea, eb) == 3:
		return PackedVector2Array([pa, pb])  # straight through

	# Tangents are radial at each endpoint, so the arc centre is where the two
	# perpendiculars meet.
	var ta: Vector2 = edge_offset[ea].orthogonal()
	var tb: Vector2 = edge_offset[eb].orthogonal()
	var det: float = ta.x * (-tb.y) - (-tb.x) * ta.y
	if absf(det) < 1e-6:
		return PackedVector2Array([pa, pb])

	var rx: float = pb.x - pa.x
	var ry: float = pb.y - pa.y
	var s: float = (rx * (-tb.y) - (-tb.x) * ry) / det
	var oc: Vector2 = pa + ta * s
	var rad: float = oc.distance_to(pa)

	var a0: float = (pa - oc).angle()
	var a1: float = (pb - oc).angle()
	var d: float = wrapf(a1 - a0, -PI, PI)  # short way -- stays inside the cell

	var pts := PackedVector2Array()
	for i in range(samples + 1):
		var a: float = a0 + d * (float(i) / float(samples))
		pts.append(oc + Vector2(cos(a), sin(a)) * rad)
	return pts
