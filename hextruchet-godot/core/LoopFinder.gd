class_name LoopFinder
extends RefCounted

## Loop-closure detection and enclosed-area computation.
##
## Ported from hex_truchet/_hexcore.py (graph.py Board.components/loops +
## stage0.enclosed_cells). This is the one genuinely tricky part of the rules
## engine, and it has a history: an earlier Python attempt at a localized
## BFS-from-new-arcs silently under-reported when one tile's arcs belonged to
## two independent loops at once. The safe formulation, used here, recomputes
## components over all placed arcs.
##
## Definitions:
##   - An ARC lives on a cell and joins two of its edges. Its endpoints are
##     PORTS: (cell, edge).
##   - Two arcs connect when a port pair meets across a cell boundary:
##     (A, i) meets (B, (i+3)%6) where B = neighbour(A, i).
##   - A LOOP is a connected component of arcs in which EVERY arc has BOTH of
##     its ports connected (no open ends). Components with an open port are
##     "runs", not loops, and score nothing.
##   - ENCLOSED AREA is by ray-cast crossing parity: walk from a cell in edge-0
##     direction to the board edge, counting crossings of the loop; odd = inside.
##
## The board is tiny (<=111 arcs at radius 3), so a full recompute per
## placement is instant -- correctness over cleverness.

var geo: HexGeometry


func _init(geometry: HexGeometry) -> void:
	geo = geometry


## Find all closed loops on the board.
##
## board_tile / board_rot are [n_cells] arrays; board_tile[i] == -1 means empty.
## Returns an Array of Dictionaries, each:
##   { "arcs": Array[Vector3i](cell, ea, eb), "cells": PackedInt32Array,
##     "area": int, "length": int }
func find_loops(board_tile: PackedInt32Array, board_rot: PackedInt32Array) -> Array:
	# --- collect arcs -----------------------------------------------------
	# arc_id -> (cell, ea, eb); port_to_arc[cell * 6 + edge] -> arc_id or -1
	var arc_cell := PackedInt32Array()
	var arc_ea := PackedInt32Array()
	var arc_eb := PackedInt32Array()
	var port_to_arc := PackedInt32Array()
	port_to_arc.resize(geo.n_cells * HexGeometry.N_EDGES)
	port_to_arc.fill(-1)

	for cell_idx in geo.n_cells:
		var t: int = board_tile[cell_idx]
		if t < 0:
			continue
		for pair: Vector2i in geo.arcs_for(t, board_rot[cell_idx]):
			var aid: int = arc_cell.size()
			arc_cell.append(cell_idx)
			arc_ea.append(pair.x)
			arc_eb.append(pair.y)
			port_to_arc[cell_idx * HexGeometry.N_EDGES + pair.x] = aid
			port_to_arc[cell_idx * HexGeometry.N_EDGES + pair.y] = aid

	var n_arcs: int = arc_cell.size()
	if n_arcs == 0:
		return []

	# --- union-find over arcs, joined across shared cell boundaries -------
	var parent := PackedInt32Array()
	parent.resize(n_arcs)
	for i in n_arcs:
		parent[i] = i

	# A port is "connected" when the neighbouring cell has an arc meeting it.
	var port_connected := PackedInt32Array()
	port_connected.resize(geo.n_cells * HexGeometry.N_EDGES)
	port_connected.fill(0)

	for aid in n_arcs:
		var cell_idx: int = arc_cell[aid]
		for edge in [arc_ea[aid], arc_eb[aid]]:
			var nb: int = geo.neighbors[cell_idx][edge]
			if nb < 0:
				continue
			var opp: int = HexGeometry.opposite_edge(edge)
			var other: int = port_to_arc[nb * HexGeometry.N_EDGES + opp]
			if other < 0:
				continue
			port_connected[cell_idx * HexGeometry.N_EDGES + edge] = 1
			_union(parent, aid, other)

	# --- group arcs by component; a component is a loop iff no open port ---
	var groups: Dictionary = {}          # root -> Array[int] of arc ids
	for aid in n_arcs:
		var root: int = _find(parent, aid)
		if not groups.has(root):
			groups[root] = []
		groups[root].append(aid)

	var loops: Array = []
	for root: int in groups:
		var members: Array = groups[root]
		var is_loop := true
		for aid: int in members:
			var base: int = arc_cell[aid] * HexGeometry.N_EDGES
			if port_connected[base + arc_ea[aid]] == 0 \
					or port_connected[base + arc_eb[aid]] == 0:
				is_loop = false
				break
		if not is_loop:
			continue

		var arcs: Array[Vector3i] = []
		for aid: int in members:
			arcs.append(Vector3i(arc_cell[aid], arc_ea[aid], arc_eb[aid]))
		arcs.sort()
		var enclosed: PackedInt32Array = _enclosed_cells(arcs)
		loops.append({
			"arcs": arcs,
			"cells": enclosed,
			"area": enclosed.size(),
			"length": members.size(),
		})

	loops.sort_custom(_compare_loops)
	return loops


## Total enclosed area across all closed loops -- the area_linear score basis.
func total_loop_area(board_tile: PackedInt32Array, board_rot: PackedInt32Array) -> int:
	var total: int = 0
	for loop: Dictionary in find_loops(board_tile, board_rot):
		total += int(loop["area"])
	return total


func _compare_loops(a: Dictionary, b: Dictionary) -> bool:
	var aa: Array = a["arcs"]
	var bb: Array = b["arcs"]
	if aa.is_empty() or bb.is_empty():
		return aa.size() < bb.size()
	var x: Vector3i = aa[0]
	var y: Vector3i = bb[0]
	if x != y:
		return x < y
	return int(a["area"]) < int(b["area"])


## Cells enclosed by a loop, via edge-0 ray-cast crossing parity.
##
## Walk from each cell in DIRECTIONS[0] until off-board, counting how many
## times the path crosses this loop's ports. Odd crossings => inside.
func _enclosed_cells(arcs: Array[Vector3i]) -> PackedInt32Array:
	# Ports belonging to this loop, as a set keyed by cell * 6 + edge.
	var loop_ports: Dictionary = {}
	for a: Vector3i in arcs:
		loop_ports[a.x * HexGeometry.N_EDGES + a.y] = true
		loop_ports[a.x * HexGeometry.N_EDGES + a.z] = true

	var inside := PackedInt32Array()
	for start in geo.n_cells:
		var crossings: int = 0
		var cur: int = start
		# Bounded by board size; the walk always exits within `n_cells` steps.
		while cur >= 0:
			if loop_ports.has(cur * HexGeometry.N_EDGES + 0):
				crossings += 1
			cur = geo.neighbors[cur][0]
		if crossings % 2 == 1:
			inside.append(start)
	return inside


func _find(parent: PackedInt32Array, x: int) -> int:
	var root: int = x
	while parent[root] != root:
		root = parent[root]
	# path compression
	var cur: int = x
	while parent[cur] != root:
		var nxt: int = parent[cur]
		parent[cur] = root
		cur = nxt
	return root


func _union(parent: PackedInt32Array, a: int, b: int) -> void:
	var ra: int = _find(parent, a)
	var rb: int = _find(parent, b)
	if ra != rb:
		parent[rb] = ra
