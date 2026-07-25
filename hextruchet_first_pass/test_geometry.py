"""Geometry tests, per plan sections 10.1 and 10.2."""

from geometry import (
    DIRECTIONS, neighbor, opposite_edge, hex_board,
    all_matchings, rotate_matching, span, span_multiset,
    canonical_form, orbit_size, canonical_tiles,
)


def test_neighbor_reciprocity():
    """Most bug-prone identity in the system. Exhaustive on radius 2."""
    for cell in hex_board(2):
        for e in range(6):
            nb = neighbor(cell, e)
            back = neighbor(nb, opposite_edge(e))
            assert back == cell, f"reciprocity broke: {cell} edge {e}"
    print("PASS neighbor reciprocity (exhaustive, radius 2)")


def test_board_sizes():
    assert len(hex_board(3)) == 37, len(hex_board(3))
    assert len(hex_board(4)) == 61, len(hex_board(4))
    print("PASS board sizes: r3=37, r4=61")


def test_matching_count():
    ms = all_matchings()
    assert len(ms) == 15, f"expected 15 matchings, got {len(ms)}"
    print("PASS matching count = 15")


def test_matchings_wellformed():
    for m in all_matchings():
        assert len(m) == 3
        edges = [e for pair in m for e in pair]
        assert sorted(edges) == list(range(6)), f"bad cover: {m}"
    print("PASS all matchings cover 6 edges exactly once")


def test_rotation_identity():
    for m in all_matchings():
        assert rotate_matching(m, 6) == m
        assert rotate_matching(m, 0) == m
    print("PASS rotating 6 times returns original")


def test_span_rotation_invariant():
    for m in all_matchings():
        base = span_multiset(m)
        for k in range(6):
            assert span_multiset(rotate_matching(m, k)) == base
    print("PASS span multiset is rotation-invariant")


def test_orbit_sum():
    """Strongest available check on canonicalization."""
    tiles = canonical_tiles()
    total = sum(t["orbit"] for t in tiles)
    assert total == 15, f"orbit sizes sum to {total}, expected 15"
    print(f"PASS orbit sizes sum to 15 across {len(tiles)} canonical tiles")


def test_canonical_partition():
    """Every matching maps to exactly one canonical tile."""
    tiles = canonical_tiles()
    canon_keys = {canonical_form(t["matching"]) for t in tiles}
    assert len(canon_keys) == len(tiles)
    for m in all_matchings():
        assert canonical_form(m) in canon_keys
    print("PASS canonical forms partition all 15 matchings")


def report_tiles():
    tiles = canonical_tiles()
    print(f"\n--- Canonical tile set: {len(tiles)} tiles ---")
    for i, t in enumerate(tiles):
        arcs = sorted(tuple(sorted(p)) for p in t["matching"])
        print(f"  tile {i}: arcs={arcs}  spans={t['spans']}  orbit={t['orbit']}")
    print()
    from collections import Counter
    c = Counter(t["spans"] for t in tiles)
    print("  span-multiset frequencies:", dict(c))


if __name__ == "__main__":
    test_neighbor_reciprocity()
    test_board_sizes()
    test_matching_count()
    test_matchings_wellformed()
    test_rotation_identity()
    test_span_rotation_invariant()
    test_orbit_sum()
    test_canonical_partition()
    report_tiles()
