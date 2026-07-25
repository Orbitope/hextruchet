"""Extended tile types beyond the standard perfect-matching set.

- Blank tile: 0 arcs. Pure spacer. Blocks the cell for connectivity purposes
  (no arcs pass through) but still occupies a board cell.
- Partial tile: 2 arcs covering 4 edges, 2 edges left as dead stubs (open,
  connect to nothing structurally -- equivalent to two "half-arcs" that
  never join anything). We approximate this as a matching on 4 of 6 edges,
  leaving 2 edges with no arc at all (not even a stub).
"""

from itertools import combinations
from geometry import all_matchings as all_full_matchings


def blank_tile():
    """0 arcs at all. Returns an empty matching."""
    return frozenset()


def partial_matchings():
    """All matchings using exactly 2 of the 3 possible arc-pairs (4 edges
    covered, 2 edges bare). Enumerate: choose 4 of 6 edges, then a perfect
    matching on those 4 (3 matchings each), dedupe by symmetry later if
    needed -- for simulation we don't need to canonicalize, just sample.
    """
    out = []
    edges = list(range(6))
    for four in combinations(edges, 4):
        four = list(four)
        # matchings on these 4 edges: 3 possible pairings
        a, b, c, d = four
        pairings = [
            frozenset((frozenset((a, b)), frozenset((c, d)))),
            frozenset((frozenset((a, c)), frozenset((b, d)))),
            frozenset((frozenset((a, d)), frozenset((b, c)))),
        ]
        out.extend(pairings)
    # dedupe
    return list({p for p in out})
