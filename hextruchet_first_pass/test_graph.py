"""Cycle detection tests, per plan section 10.3."""

from geometry import (
    hex_board, neighbor, opposite_edge, canonical_tiles,
    all_matchings, rotate_matching, tile_arcs,
)
from graph import Board

TILES = canonical_tiles()


def test_single_tile():
    b = Board(hex_board(1))
    b.place((0, 0), TILES[0]["matching"], 0)
    comps = b.components()
    loops = [c for c in comps if c["is_loop"]]
    assert len(loops) == 0, "isolated tile should have no loops"
    assert len(comps) == 3, f"expected 3 runs, got {len(comps)}"
    assert len(b.open_ports()) == 6
    print("PASS single isolated tile: 0 loops, 3 runs, 6 open ports")


def test_two_tiles_connect():
    """Two adjacent tiles: arcs meeting at shared edge should join."""
    b = Board(hex_board(2))
    b.place((0, 0), TILES[0]["matching"], 0)
    b.place(neighbor((0, 0), 0), TILES[0]["matching"], 0)
    conns = b.arc_connections()
    assert len(conns) == 1, f"expected 1 connection, got {len(conns)}"
    comps = b.components()
    assert len([c for c in comps if c["is_loop"]]) == 0
    # 6 arcs total, one join -> 5 components
    assert len(comps) == 5, f"expected 5 components, got {len(comps)}"
    print("PASS two adjacent tiles: 1 connection, 0 loops, 5 components")


def find_minimal_loop():
    """Search for the smallest closed loop by brute force over 3 cells
    around a shared vertex. Returns config if found."""
    # Three mutually adjacent cells around a vertex.
    a = (0, 0)
    b = neighbor(a, 0)
    c = neighbor(a, 1)
    # verify mutual adjacency
    assert b in [neighbor(c, e) for e in range(6)], "cells not mutually adjacent"

    ms = all_matchings()
    for ma in ms:
        for mb in ms:
            for mc in ms:
                board = Board(hex_board(2))
                board.place(a, ma, 0)
                board.place(b, mb, 0)
                board.place(c, mc, 0)
                loops = board.loops()
                if loops:
                    shortest = min(l["length"] for l in loops)
                    if shortest == 3:
                        return (a, ma), (b, mb), (c, mc), loops
    return None


def test_minimal_loop():
    """The smallest closure. Most important single test."""
    res = find_minimal_loop()
    assert res is not None, "no 3-tile loop found -- geometry may be wrong"
    (a, ma), (b, mb), (c, mc), loops = res
    three = [l for l in loops if l["length"] == 3]
    assert three, "expected a length-3 loop"
    print(f"PASS minimal 3-tile loop found: length 3, {len(loops)} loop(s) total")
    return res


def test_ring_of_six():
    """Ring of 6 around empty center should close a loop enclosing 1 cell."""
    center = (0, 0)
    ring = [neighbor(center, e) for e in range(6)]
    b = Board(hex_board(2))
    # Each ring cell must connect to its two ring neighbors.
    # For ring cell at direction e, its neighbors in the ring are at
    # directions (e+2)%6 and (e+4)%6 from itself... determine empirically.
    found = False
    ms = all_matchings()
    # Try: for each ring cell, find the arc joining the two edges facing
    # its ring-neighbours.
    for i, rc in enumerate(ring):
        prev_rc = ring[(i - 1) % 6]
        next_rc = ring[(i + 1) % 6]
        e_prev = next(e for e in range(6) if neighbor(rc, e) == prev_rc)
        e_next = next(e for e in range(6) if neighbor(rc, e) == next_rc)
        # need a matching containing pair {e_prev, e_next}
        m = next(m for m in ms if frozenset((e_prev, e_next)) in m)
        b.place(rc, m, 0)
        found = True
    loops = b.loops()
    six = [l for l in loops if l["length"] == 6]
    assert six, f"expected a 6-loop, got lengths {[l['length'] for l in loops]}"
    print(f"PASS ring of six: found 6-length loop, {len(loops)} loop(s) total")


def test_uf_vs_bfs_oracle():
    """Two independent implementations must agree. Strongest check."""
    import random
    rng = random.Random(12345)
    ms = all_matchings()
    cells = hex_board(3)
    for trial in range(300):
        b = Board(cells)
        for cell in cells:
            b.place(cell, rng.choice(ms), rng.randrange(6))
        uf = b.components()
        bfs = b.components_bfs()
        uf_sig = sorted((tuple(c["arcs"]), c["is_loop"]) for c in uf)
        bfs_sig = sorted((tuple(c["arcs"]), c["is_loop"]) for c in bfs)
        assert uf_sig == bfs_sig, f"union-find and BFS disagree on trial {trial}"
    print("PASS union-find matches BFS oracle over 300 random full boards")


def test_arc_conservation():
    """Every arc in exactly one component."""
    import random
    rng = random.Random(999)
    ms = all_matchings()
    cells = hex_board(3)
    for trial in range(200):
        b = Board(cells)
        for cell in cells:
            b.place(cell, rng.choice(ms), rng.randrange(6))
        comps = b.components()
        total = sum(c["length"] for c in comps)
        assert total == 3 * len(cells), f"arc count mismatch: {total}"
        all_arcs = sorted(a for c in comps for a in c["arcs"])
        assert all_arcs == sorted(b.arcs.keys()), "arcs lost or duplicated"
    print("PASS arc conservation: 3*cells arcs, each in exactly one component")


if __name__ == "__main__":
    test_single_tile()
    test_two_tiles_connect()
    test_minimal_loop()
    test_ring_of_six()
    test_arc_conservation()
    test_uf_vs_bfs_oracle()
