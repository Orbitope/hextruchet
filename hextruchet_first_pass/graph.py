"""Arc graph, union-find cycle detection, enclosed area.

An arc lives on a cell and joins two of its edges. Arc endpoints are
"ports": (cell, edge). Two arcs connect when they share a port pair
across a cell boundary: port (A, i) meets port (B, (i+3)%6) where
B = neighbor(A, i).

We model the arc graph as: nodes = arcs, edges = adjacency across shared
cell boundaries. A loop is a cycle in this graph where every arc has both
of its ports matched.
"""

from geometry import neighbor, opposite_edge, tile_arcs


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
