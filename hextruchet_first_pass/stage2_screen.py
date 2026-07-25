"""Stage 2 (screening pass): drafting mechanisms x placement rules x agents.

Scoped down per plan contingency: radius 3 only, area_linear scorer only,
small game counts. Purpose is triage -- find what's promising enough to
justify a deeper, slower run with more games/scorers/board sizes.

Mechanisms implemented (subset of plan A-E, cheapest correct first):
  bag  -- draw one tile at random from the fixed deck, place it
  pool -- see k=3 face-up tiles, pick one, place it, pool refills
  hand -- hold a hand of h=3, play one per turn, hand refills

Snake draft and offer/choose deferred -- bigger structural change, not
worth it before something here looks promising.

Placement: adjacency-required vs free.
Agents: random, greedy. (denial deferred -- slowest, and this pass is
triage not final balance numbers.)
"""

import random
import time
import hashlib
import numpy as np

from geometry import hex_board, canonical_tiles, neighbor
from stage0 import enclosed_cells
from agents import legal_cells_free, legal_cells_adjacent, GameBoard, SCORERS

TILES = canonical_tiles()


def config_seed(*parts):
    """Deterministic per-config seed. Python's builtin hash() randomizes
    string/tuple hashes per process (PYTHONHASHSEED), so the old
    hash((...)) % 999999 seeding produced different games every run and was
    not reproducible. md5 of the stringified parts is stable across processes.
    """
    s = "|".join(str(p) for p in parts)
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2 ** 31)


def make_deck(n_cells, rng):
    """Fixed multiset matching Stage 0's winning ratio (tile 0 : tile 2 = 1:2)."""
    n0 = round(n_cells / 3)
    n2 = n_cells - n0
    deck = [0] * n0 + [2] * n2
    rng.shuffle(deck)
    return deck


def legal_fn(board, adjacency_required):
    return legal_cells_adjacent(board, board.cells) if adjacency_required \
        else legal_cells_free(board, board.cells)


def agent_for(agent_types, player):
    """agent_types is either a single string (same agent for every seat,
    the historical behaviour) or a per-player list/tuple for mixed-agent
    matchups (e.g. ["greedy", "random"])."""
    if isinstance(agent_types, (list, tuple)):
        return agent_types[player]
    return agent_types


# Denial agent tuning. Opponent tile pool mirrors make_deck's composition
# (tiles 0 and 2 only) since choose_move doesn't know the actual next
# player's hand/pool contents -- this is "best any plausible next tile could
# do", not a literal lookahead at the real next hand. Rotations and cell
# count are sampled/capped for speed, same tradeoff agents.py's
# best_response_value makes.
DENIAL_TOP_K = 8
DENIAL_OPPONENT_TILES = (0, 2)
DENIAL_OPP_ROTATIONS = (0, 2, 4)
DENIAL_OPP_CELL_CAP = 10


def _frontier_cells(board, legal):
    """Cells in `legal` adjacent to >=1 already-placed tile. A tile placed
    on a cell with no placed neighbor is isolated -- none of its arcs can
    connect, so it always scores 0 and can never be a positive-scoring
    move. Restricting search to the frontier is therefore behaviour-
    preserving, not an approximation (see choose_move's greedy branch,
    unchanged from before this cut)."""
    if not board.placed:
        return []
    placed_set = set(board.placed)
    return [c for c in legal
            if any(neighbor(c, e) in placed_set for e in range(6))]


def _search_own_moves(board, candidates, scorer, eval_cells, base_loop_keys):
    """All (value, candidate_index, cell, rotation) tuples from placing each
    candidate tile at each frontier cell/rotation, scored against the
    board's current (precomputed once) loop set. Iteration order is
    candidates outer, eval_cells, then rotation -- callers that need
    greedy's original first-strict-max tie-break must preserve this order.
    """
    out = []
    for i, t in enumerate(candidates):
        for c in eval_cells:
            for rot in range(6):
                records, undo = board.try_place_and_get_new_loops(
                    c, TILES[t]["matching"], rot, enclosed_cells,
                    prev_loop_keys=base_loop_keys)
                out.append((scorer(records), i, c, rot))
                undo()
    return out


def _best_opponent_reply(board, cell, tile_idx, rotation, scorer, adjacency_required):
    """Place (tile_idx, rotation) at cell, then find the best score any
    opponent could gain on their immediate next placement anywhere legal --
    then undo everything, leaving the board as found. Used by the denial
    agent to penalize moves that hand the next player an easy score.
    """
    _, undo_self = board.try_place_and_get_new_loops(
        cell, TILES[tile_idx]["matching"], rotation, enclosed_cells)
    opp_legal = legal_fn(board, adjacency_required)
    best = 0.0
    if opp_legal:
        opp_cells = opp_legal[:DENIAL_OPP_CELL_CAP]
        opp_base_keys = {frozenset(cc["arcs"])
                         for cc in board.components() if cc["is_loop"]}
        for t2 in DENIAL_OPPONENT_TILES:
            for c2 in opp_cells:
                for r2 in DENIAL_OPP_ROTATIONS:
                    records2, undo2 = board.try_place_and_get_new_loops(
                        c2, TILES[t2]["matching"], r2, enclosed_cells,
                        prev_loop_keys=opp_base_keys)
                    val2 = scorer(records2)
                    undo2()
                    if val2 > best:
                        best = val2
    undo_self()
    return best


def choose_move(board, candidates, player, scorer, agent_type, rng, adjacency_required):
    """candidates: list of tile indices available to place right now.
    Returns (candidate_index_chosen, cell, rotation) or None if no legal cell.
    """
    legal = legal_fn(board, adjacency_required)
    if not legal:
        return None
    if agent_type == "random":
        idx = rng.randrange(len(candidates))
        c = rng.choice(legal)
        rot = rng.randrange(6)
        return idx, c, rot

    # Greedy / denial share the own-gain search below; denial additionally
    # penalizes candidates that set up an easy reply for the next player.
    eval_cells = _frontier_cells(board, legal)
    if eval_cells:
        # The base board is identical across every candidate we try below, so
        # its loop set is too; compute it once instead of once per candidate.
        base_loop_keys = {frozenset(cc["arcs"])
                          for cc in board.components() if cc["is_loop"]}
        scored = _search_own_moves(board, candidates, scorer, eval_cells, base_loop_keys)

        if agent_type == "denial":
            if scored and max(v for v, *_ in scored) > 0:
                top = sorted(scored, key=lambda x: -x[0])[:DENIAL_TOP_K]
                best, best_val = None, -1e9
                for val, i, c, rot in top:
                    opp_best = _best_opponent_reply(
                        board, c, candidates[i], rot, scorer, adjacency_required)
                    net = val - 0.5 * opp_best
                    if net > best_val:
                        best_val = net
                        best = (i, c, rot)
                if best is not None:
                    return best
        else:
            # Greedy: first strictly-greater value wins (preserves the
            # original loop's tie-break exactly).
            best, best_val = None, -1
            for val, i, c, rot in scored:
                if val > best_val:
                    best_val = val
                    best = (i, c, rot)
            if best is not None and best_val > 0:
                return best
    # No positive-scoring move (incl. empty/near-empty board): random legal.
    idx = rng.randrange(len(candidates))
    c = rng.choice(legal)
    rot = rng.randrange(6)
    return idx, c, rot


def do_place(board, cell, tile_idx, rotation, scorer):
    records, undo = board.try_place_and_get_new_loops(
        cell, TILES[tile_idx]["matching"], rotation, enclosed_cells)
    undo()
    board.place(cell, TILES[tile_idx]["matching"], rotation)
    return scorer(records)


def play_bag(cells, n_players, adjacency_required, agent_types, scorer, rng, on_place=None):
    deck = make_deck(len(cells), rng)
    board = GameBoard(cells, adjacency_required)
    scores = [0.0] * n_players
    for i, t in enumerate(deck):
        player = i % n_players
        move = choose_move(board, [t], player, scorer, agent_for(agent_types, player), rng, adjacency_required)
        if move is None:
            break
        _, c, rot = move
        scores[player] += do_place(board, c, t, rot, scorer)
        if on_place:
            on_place(i, player, scores)
    return scores, board


def play_pool(cells, n_players, adjacency_required, agent_types, scorer, rng, k=3, on_place=None):
    deck = make_deck(len(cells), rng)
    pool = deck[:k]
    remaining = deck[k:]
    board = GameBoard(cells, adjacency_required)
    scores = [0.0] * n_players
    turn = 0
    while pool:
        player = turn % n_players
        move = choose_move(board, pool, player, scorer, agent_for(agent_types, player), rng, adjacency_required)
        if move is None:
            break
        idx, c, rot = move
        t = pool.pop(idx)
        if remaining:
            pool.append(remaining.pop(0))
        scores[player] += do_place(board, c, t, rot, scorer)
        if on_place:
            on_place(turn, player, scores)
        turn += 1
    return scores, board


def play_hand(cells, n_players, adjacency_required, agent_types, scorer, rng, h=3, on_place=None):
    deck = make_deck(len(cells), rng)
    hands = [[] for _ in range(n_players)]
    for p in range(n_players):
        for _ in range(h):
            if deck:
                hands[p].append(deck.pop(0))
    board = GameBoard(cells, adjacency_required)
    scores = [0.0] * n_players
    turn = 0
    stall = 0
    while any(hands) or deck:
        player = turn % n_players
        if not hands[player]:
            turn += 1
            stall += 1
            if stall > n_players * 2:
                break
            continue
        stall = 0
        move = choose_move(board, hands[player], player, scorer, agent_for(agent_types, player), rng, adjacency_required)
        if move is None:
            break
        idx, c, rot = move
        t = hands[player].pop(idx)
        if deck:
            hands[player].append(deck.pop(0))
        scores[player] += do_place(board, c, t, rot, scorer)
        if on_place:
            on_place(turn, player, scores)
        turn += 1
    return scores, board


MECHANISMS = {"bag": play_bag, "pool": play_pool, "hand": play_hand}


def separation(score_lists, n_players):
    arr = np.array(score_lists, dtype=float)
    totals = arr.sum(axis=1, keepdims=True)
    nz = (totals[:, 0] > 0)
    if nz.sum() == 0:
        return {"margin": 0.0, "frac_tied": 1.0, "seat0_winrate": 1.0 / n_players}
    normed = np.zeros_like(arr)
    normed[nz] = arr[nz] / totals[nz]
    sorted_n = np.sort(normed[nz], axis=1)[:, ::-1]
    margin = float((sorted_n[:, 0] - sorted_n[:, 1]).mean())
    frac_tied = float(((sorted_n[:, 0] - sorted_n[:, 1]) < 0.02).mean())
    winners = np.argmax(arr, axis=1)
    seat0_winrate = float((winners == 0).mean())
    return {"margin": margin, "frac_tied": frac_tied, "seat0_winrate": seat0_winrate}


def run_screen(n_games, radius, n_players, out_path=None, scorer_name="area_linear",
               agents=("random", "greedy")):
    scorer = SCORERS[scorer_name]
    cells = hex_board(radius)
    print(f"\n{'='*104}")
    print(f"STAGE 2 SCREEN -- radius {radius} ({len(cells)} cells), {n_players} players, "
          f"{n_games} games/config, scorer={scorer_name}")
    print(f"{'='*104}")
    print(f"{'mechanism':<8}{'adjacency':<11}{'agent':<8}{'margin':>9}"
          f"{'frac_tied':>11}{'seat0_wr':>10}{'sec/game':>10}")
    print("-"*104)

    results = {}
    for mech_name, mech_fn in MECHANISMS.items():
        for adj in (False, True):
            for agent in agents:
                rng = random.Random(config_seed(mech_name, adj, agent, radius, n_players, scorer_name))
                scores_all = []
                t0 = time.time()
                for _ in range(n_games):
                    scores, _ = mech_fn(cells, n_players, adj, agent, scorer, rng)
                    scores_all.append(scores)
                elapsed = (time.time() - t0) / n_games
                sep = separation(scores_all, n_players)
                results[(mech_name, adj, agent)] = sep
                line = (f"{mech_name:<8}{str(adj):<11}{agent:<8}{sep['margin']:>9.3f}"
                        f"{sep['frac_tied']:>11.3f}{sep['seat0_winrate']:>10.3f}"
                        f"{elapsed:>9.3f}s")
                print(line, flush=True)
                if out_path:
                    with open(out_path, "a") as f:
                        f.write(f"{radius},{n_players},{scorer_name},{mech_name},{adj},{agent},"
                                f"{sep['margin']:.4f},{sep['frac_tied']:.4f},"
                                f"{sep['seat0_winrate']:.4f},{elapsed:.4f}\n")
    return results


def run_matchup(n_games, radius, n_players, skilled_agent="greedy", baseline_agent="random",
                 out_path=None):
    """Head-to-head: exactly one seat plays `skilled_agent`, the rest play
    `baseline_agent`. The skilled seat is rotated evenly across all n_players
    positions (n_games split as evenly as possible across rotations) so seat-
    order advantage (real, see Stage 2 seat-balance results) doesn't get
    conflated with agent-skill advantage. Reports the skilled agent's win
    rate against the fair baseline of 1/n_players.
    """
    scorer = SCORERS["area_linear"]
    cells = hex_board(radius)
    print(f"\n{'='*104}")
    print(f"STAGE 2 MATCHUP -- radius {radius} ({len(cells)} cells), {n_players} players, "
          f"{skilled_agent} vs {baseline_agent}, {n_games} games/config "
          f"(fair baseline win rate = {1/n_players:.3f})")
    print(f"{'='*104}")
    print(f"{'mechanism':<8}{'adjacency':<11}{skilled_agent+'_wr':>12}"
          f"{'tie_rate':>10}{'sec/game':>10}")
    print("-"*104)

    results = {}
    for mech_name, mech_fn in MECHANISMS.items():
        for adj in (False, True):
            rng = random.Random(config_seed(mech_name, adj, skilled_agent, baseline_agent,
                                             radius, n_players, "matchup"))
            wins, ties, n_run = 0, 0, 0
            t0 = time.time()
            for g in range(n_games):
                skilled_seat = g % n_players
                agent_types = [baseline_agent] * n_players
                agent_types[skilled_seat] = skilled_agent
                scores, _ = mech_fn(cells, n_players, adj, agent_types, scorer, rng)
                best = max(scores)
                winners = [p for p, s in enumerate(scores) if s == best]
                if len(winners) > 1:
                    ties += 1
                elif winners[0] == skilled_seat:
                    wins += 1
                n_run += 1
            elapsed = (time.time() - t0) / n_run
            wr = wins / n_run
            tie_rate = ties / n_run
            results[(mech_name, adj)] = {"win_rate": wr, "tie_rate": tie_rate}
            line = (f"{mech_name:<8}{str(adj):<11}{wr:>12.3f}"
                    f"{tie_rate:>10.3f}{elapsed:>9.3f}s")
            print(line, flush=True)
            if out_path:
                with open(out_path, "a") as f:
                    f.write(f"{radius},{n_players},{mech_name},{adj},"
                            f"{skilled_agent},{baseline_agent},{wr:.4f},{tie_rate:.4f},"
                            f"{elapsed:.4f}\n")
    return results


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_screen(n, radius=3, n_players=2)
    run_screen(n, radius=3, n_players=3)
