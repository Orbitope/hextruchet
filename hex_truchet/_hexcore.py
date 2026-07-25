"""Vendored hex-truchet geometry + arc-graph + area core.

VERBATIM copy (assembled by script, not hand-transcribed) of the proven,
unit-tested Stage 0-2 modules:
  - geometry.py  (hex axial coords, tile enumeration/canonicalization)
  - graph.py     (Board, union-find loop-closure detection,
                  try_place_and_get_new_loops)
  - enclosed_cells() from stage0.py (ray-cast enclosed area)

Source of truth for these algorithms is
/Users/mwburke/projects/hextruchet/hextruchet_first_pass/. This file is
vendored so the simulacrum package is self-contained. reference.py reuses
this tested code deliberately (see HANDOFF.md 4.1 for the loop-closure bug
history we are avoiding reintroducing). Do NOT edit game logic here; fix it
in the source modules and re-vendor. fast.py must NOT import this -- the
batched implementation reimplements loop closure as tensor ops from spec.md,
and the differential test validates the two against each other.
"""

# ======================================================================
# --- geometry.py ---
# ======================================================================

"""Hex Truchet geometry: axial coords, tile enumeration, canonicalization.

Conventions (locked, per plan section 10.1):
  - Axial coordinates (q, r). Pointy-top hexes.
  - Edge indexing 0-5. Edge i of cell c faces neighbor DIRECTIONS[i].
  - Neighbor reciprocity: edge i of A faces B  <=>  edge (i+3)%6 of B faces A.
  - Rotation by one step maps edge i -> edge (i+1)%6.

A tile is a perfect matching on the 6 edges: a frozenset of 3 frozenset pairs.
"""

from itertools import combinations
from functools import lru_cache

# Axial direction vectors, indexed 0-5.
# Chosen so that DIRECTIONS[(i+3)%6] == -DIRECTIONS[i], which gives reciprocity.
DIRECTIONS = [
    (1, 0),    # 0
    (1, -1),   # 1
    (0, -1),   # 2
    (-1, 0),   # 3
    (-1, 1),   # 4
    (0, 1),    # 5
]


def neighbor(cell, edge):
    """Cell across `edge` from `cell`."""
    dq, dr = DIRECTIONS[edge]
    return (cell[0] + dq, cell[1] + dr)


def opposite_edge(edge):
    """The edge index on the neighbor that faces back."""
    return (edge + 3) % 6


def hex_board(radius):
    """All cells within `radius` of origin, hexagonal board."""
    cells = []
    for q in range(-radius, radius + 1):
        r_lo = max(-radius, -q - radius)
        r_hi = min(radius, -q + radius)
        for r in range(r_lo, r_hi + 1):
            cells.append((q, r))
    return cells


# ---------------------------------------------------------------------------
# Tile enumeration
# ---------------------------------------------------------------------------

def all_matchings():
    """All perfect matchings on 6 labeled edges. Should be 15."""
    def rec(remaining):
        if not remaining:
            yield frozenset()
            return
        first = remaining[0]
        for j in range(1, len(remaining)):
            pair = frozenset((first, remaining[j]))
            rest = remaining[1:j] + remaining[j + 1:]
            for sub in rec(rest):
                yield sub | {pair}
    return list(rec(list(range(6))))


def rotate_matching(m, k):
    """Rotate a matching by k steps."""
    return frozenset(
        frozenset(((e + k) % 6) for e in pair) for pair in m
    )


def span(pair):
    """Arc span: 1 = adjacent edges, 2 = skip-one, 3 = opposite."""
    a, b = tuple(pair)
    d = abs(a - b) % 6
    return min(d, 6 - d)


def span_multiset(m):
    return tuple(sorted(span(p) for p in m))


def canonical_form(m):
    """Lexicographically smallest rotation, as a sortable key."""
    def key(mm):
        return tuple(sorted(tuple(sorted(p)) for p in mm))
    return min((key(rotate_matching(m, k)) for k in range(6)))


def orbit_size(m):
    """Number of distinct rotations of this matching."""
    return len({rotate_matching(m, k) for k in range(6)})


def canonical_tiles():
    """Canonical tile set: one representative per rotation class.

    Returns list of dicts with keys: matching, orbit, spans.
    """
    seen = {}
    for m in all_matchings():
        c = canonical_form(m)
        if c not in seen:
            seen[c] = m
    tiles = []
    for c in sorted(seen.keys()):
        m = seen[c]
        tiles.append({
            "matching": m,
            "orbit": orbit_size(m),
            "spans": span_multiset(m),
        })
    return tiles


def tile_arcs(matching, rotation):
    """Arcs of a tile at a given rotation, as list of (edge_a, edge_b) sorted."""
    rm = rotate_matching(matching, rotation)
    return [tuple(sorted(p)) for p in rm]


# ======================================================================
# --- graph.py (Board, UnionFind) ---
# ======================================================================

"""Arc graph, union-find cycle detection, enclosed area.

An arc lives on a cell and joins two of its edges. Arc endpoints are
"ports": (cell, edge). Two arcs connect when they share a port pair
across a cell boundary: port (A, i) meets port (B, (i+3)%6) where
B = neighbor(A, i).

We model the arc graph as: nodes = arcs, edges = adjacency across shared
cell boundaries. A loop is a cycle in this graph where every arc has both
of its ports matched.
"""



class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        """Returns True if a cycle was created (already same component)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return False


class Board:
    """Board state: cells -> (tile_matching, rotation). Tracks arc graph."""

    def __init__(self, cells):
        self.cells = set(cells)
        self.placed = {}          # cell -> (matching, rotation)
        self.arcs = {}            # arc_id -> (cell, edge_a, edge_b)
        self.port_to_arc = {}     # (cell, edge) -> arc_id
        self._next_arc = 0

    def place(self, cell, matching, rotation):
        assert cell in self.cells, f"{cell} off board"
        assert cell not in self.placed, f"{cell} occupied"
        self.placed[cell] = (matching, rotation)
        new_arcs = []
        for (ea, eb) in tile_arcs(matching, rotation):
            aid = self._next_arc
            self._next_arc += 1
            self.arcs[aid] = (cell, ea, eb)
            self.port_to_arc[(cell, ea)] = aid
            self.port_to_arc[(cell, eb)] = aid
            new_arcs.append(aid)
        return new_arcs

    def new_connections_for(self, cell, new_arcs):
        """Connections introduced by placing new_arcs at `cell`, i.e. only
        checking this tile's 6 boundary ports against already-placed
        neighbors. O(1) instead of O(all placed ports).
        """
        conns = []
        for aid in new_arcs:
            _, ea, eb = self.arcs[aid]
            for edge in (ea, eb):
                nb = neighbor(cell, edge)
                other = self.port_to_arc.get((nb, opposite_edge(edge)))
                if other is not None:
                    conns.append((aid, other))
        return conns

    def arc_connections(self):
        """All connections between arcs, as (arc_a, arc_b) pairs.

        One connection per shared boundary where both sides have an arc.
        """
        conns = []
        seen = set()
        for (cell, edge), aid in self.port_to_arc.items():
            nb = neighbor(cell, edge)
            oe = opposite_edge(edge)
            other = self.port_to_arc.get((nb, oe))
            if other is None:
                continue
            key = tuple(sorted(((cell, edge), (nb, oe))))
            if key in seen:
                continue
            seen.add(key)
            conns.append((aid, other))
        return conns

    def open_ports(self):
        """Ports with no matching arc across the boundary (dead ends)."""
        out = []
        for (cell, edge), aid in self.port_to_arc.items():
            nb = neighbor(cell, edge)
            if self.port_to_arc.get((nb, opposite_edge(edge))) is None:
                out.append((cell, edge))
        return out

    # -----------------------------------------------------------------
    # Component analysis
    # -----------------------------------------------------------------

    def components(self):
        """Partition arcs into connected components.

        Returns list of dicts: {arcs, is_loop, length}.
        A component is a loop iff every arc in it has both ports connected.
        """
        uf = UnionFind()
        for aid in self.arcs:
            uf.add(aid)
        for a, b in self.arc_connections():
            uf.union(a, b)

        # Which ports are connected?
        connected_ports = set()
        for (cell, edge) in self.port_to_arc:
            nb = neighbor(cell, edge)
            if self.port_to_arc.get((nb, opposite_edge(edge))) is not None:
                connected_ports.add((cell, edge))

        groups = {}
        for aid in self.arcs:
            groups.setdefault(uf.find(aid), []).append(aid)

        comps = []
        for root, members in groups.items():
            is_loop = True
            for aid in members:
                cell, ea, eb = self.arcs[aid]
                if (cell, ea) not in connected_ports or (cell, eb) not in connected_ports:
                    is_loop = False
                    break
            comps.append({
                "arcs": sorted(members),
                "is_loop": is_loop,
                "length": len(members),
            })
        return comps

    def loops(self):
        return [c for c in self.components() if c["is_loop"]]

    def runs(self):
        return [c for c in self.components() if not c["is_loop"]]

    # -----------------------------------------------------------------
    # Independent oracle: BFS-based component finding (test cross-check)
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # Fast candidate evaluation: place a tile, get new loops, undo.
    # Maintains union-find + component metadata incrementally so this is
    # O(1) work (6 boundary checks) instead of O(all placed arcs).
    # -----------------------------------------------------------------

    def try_place_and_get_new_loops(self, cell, matching, rotation, enclosed_fn=None,
                                    prev_loop_keys=None):
        """Place a tile, return (new_loop_records, undo_fn). Caller MUST
        call undo_fn() before evaluating another candidate from the same
        base state, or the board is left mutated.

        new_loop_records: list of dicts {arcs(frozenset), length, area}
        area is computed via enclosed_fn(self, loop_dict) if provided,
        else omitted (area=None) for speed when not needed.

        prev_loop_keys: optional precomputed set of frozenset(arc-id) keys for
        the loops of the CURRENT (pre-placement) board. When a caller evaluates
        many candidates against one identical base state, this base loop set is
        the same every time; passing it in skips a redundant full components()
        rebuild per candidate (the dominant cost of greedy move search). MUST
        reflect the board exactly as it is on entry -- pass None to compute it.
        """
        assert cell not in self.placed
        prev_next_arc = self._next_arc
        prev_placed_len = len(self.placed)

        # MUST capture prev loop state BEFORE mutating placed/arcs/ports --
        # a prior version computed this after placement and always saw
        # new_keys as empty as a result. Verified against the manual-clone
        # cross-check in test_graph.py's multi-closure case.
        if prev_loop_keys is None:
            prev_loop_keys = {frozenset(c["arcs"]) for c in self.components() if c["is_loop"]}

        self.placed[cell] = (matching, rotation)
        new_arc_ids = []
        for (ea, eb) in tile_arcs(matching, rotation):
            aid = self._next_arc
            self._next_arc += 1
            self.arcs[aid] = (cell, ea, eb)
            self.port_to_arc[(cell, ea)] = aid
            self.port_to_arc[(cell, eb)] = aid
            new_arc_ids.append(aid)

        # Correctness over speed: recompute full components(). An attempt
        # at a localized BFS-from-new-arcs was tried and verified BROKEN
        # on multi-loop-closure placements (a single tile's arcs can
        # belong to two independent loops simultaneously; a merged BFS
        # from all new arcs together conflates them and can silently
        # under-report). This method is O(all placed arcs) per call, same
        # as the non-incremental path -- it exists for place/undo
        # ergonomics, not asymptotic speedup.
        cur_comps = self.components()
        cur_loops = {frozenset(c["arcs"]): c for c in cur_comps if c["is_loop"]}
        new_keys = set(cur_loops) - prev_loop_keys

        records = []
        for k in new_keys:
            l = cur_loops[k]
            area = len(enclosed_fn(self, l)) if enclosed_fn else None
            records.append({"arcs": k, "length": l["length"], "area": area})

        def undo():
            del self.placed[cell]
            for aid in new_arc_ids:
                cell_, ea, eb = self.arcs[aid]
                del self.arcs[aid]
                del self.port_to_arc[(cell_, ea)]
                del self.port_to_arc[(cell_, eb)]
            self._next_arc = prev_next_arc

        return records, undo

    def components_bfs(self):
        """Slow independent implementation, used only as a test oracle."""
        adj = {aid: set() for aid in self.arcs}
        for a, b in self.arc_connections():
            adj[a].add(b)
            adj[b].add(a)

        connected_ports = set()
        for (cell, edge) in self.port_to_arc:
            nb = neighbor(cell, edge)
            if self.port_to_arc.get((nb, opposite_edge(edge))) is not None:
                connected_ports.add((cell, edge))

        seen = set()
        comps = []
        for start in sorted(self.arcs):
            if start in seen:
                continue
            stack = [start]
            members = []
            seen.add(start)
            while stack:
                cur = stack.pop()
                members.append(cur)
                for nxt in adj[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            is_loop = all(
                (self.arcs[a][0], self.arcs[a][1]) in connected_ports
                and (self.arcs[a][0], self.arcs[a][2]) in connected_ports
                for a in members
            )
            comps.append({
                "arcs": sorted(members),
                "is_loop": is_loop,
                "length": len(members),
            })
        return comps


# ======================================================================
# --- enclosed_cells() from stage0.py ---
# ======================================================================

def enclosed_cells(board, loop):
    """Cells enclosed by a loop, via ray casting in the infinite plane.

    A loop is a closed curve made of arcs. We determine, for each cell on
    the board, whether it is inside. We use the crossing-parity method along
    a path of cells: walk from the cell outward in a fixed direction,
    counting how many times we cross an arc belonging to the loop.
    """
    loop_arcs = set(loop["arcs"])
    # Ports used by this loop
    loop_ports = set()
    for aid in loop_arcs:
        cell, ea, eb = board.arcs[aid]
        loop_ports.add((cell, ea))
        loop_ports.add((cell, eb))

    inside = []
    for cell in board.cells:
        # Walk in direction 0 until off-board, counting crossings.
        crossings = 0
        cur = cell
        steps = 0
        while steps < 200:
            # crossing edge 0 of cur into neighbor
            if (cur, 0) in loop_ports:
                crossings += 1
            nxt = neighbor(cur, 0)
            if nxt not in board.cells:
                break
            cur = nxt
            steps += 1
        if crossings % 2 == 1:
            inside.append(cell)
    return inside
