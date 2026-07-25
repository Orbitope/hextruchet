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
