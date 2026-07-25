"""Stage 2 diagnostic: does the 2-player seat-0 edge come from board control,
or just from the extra tile an odd cell count gives seat 0?

Session 2 background: a single n=60 sweep showed 63-72% seat-0 win rate under
adjacency_required=True at 2p, written up as an adjacency-specific seat-balance
concern. A same-seed rerun reproduced that exactly (ruling out a code bug), but
a *different* seed at the same n=60 gave 43-58% for the identical configs --
that swing is what this script exists to explain: win-rate-style metrics are
much noisier at n=60 than they look, and single-sweep win-rate numbers for this
question shouldn't be trusted without a larger-N check. See HANDOFF.md 7.4.

Method: snapshot the score at the equal-tile-count point (turn N_CELLS-2, i.e.
both players have placed the same number of tiles) as well as at the final
score. If seat-0's edge is purely "gets one more tile than seat 1 on an odd
board", equal-tile win rate should be ~50% and all the edge should show up
only in the final-vs-equal delta. If seat-0 is also ahead at equal tile
counts, that's evidence of a real board-control advantage beyond parity.

Finding (n=400 random / n=150 greedy per config, radius 3, area_linear):
equal-tile win rate is 39-55% (no board-control edge); the extra tile bumps
seat-0 by a real +6 to +10 points for random (z=2.3-4.0) and a smaller,
noisier +0.7 to +3.3 points for greedy -- and this bump is essentially the
same size under adjacency_required=True and =False. So the edge is real but
small, purely a consequence of splitting an odd cell count two ways, and NOT
specific to adjacency_required.
"""

import random
import sys
import time

from geometry import hex_board
from agents import SCORERS
from stage2_screen import play_bag, play_pool, play_hand, config_seed

MECH = {"bag": play_bag, "pool": play_pool, "hand": play_hand}


def run(mech_name, adj, agent, scorer, cells, equal_turn, n_games, seed_tag):
    fn = MECH[mech_name]
    rng = random.Random(config_seed(mech_name, adj, agent, seed_tag))
    equal_wins0 = equal_ties = final_wins0 = final_ties = n_ok = 0
    for _ in range(n_games):
        snap = {}

        def on_place(turn, player, scores):
            if turn == equal_turn:
                snap["equal"] = list(scores)

        scores, board = fn(cells, 2, adj, agent, scorer, rng, on_place=on_place)
        if len(board.placed) != len(cells) or "equal" not in snap:
            continue  # incomplete game (shouldn't happen on a full board, guard anyway)
        n_ok += 1
        e = snap["equal"]
        if e[0] > e[1]:
            equal_wins0 += 1
        elif e[0] == e[1]:
            equal_ties += 1
        if scores[0] > scores[1]:
            final_wins0 += 1
        elif scores[0] == scores[1]:
            final_ties += 1
    se = (0.25 / n_ok) ** 0.5
    return {
        "n": n_ok,
        "equal_wr0": equal_wins0 / n_ok, "equal_tie": equal_ties / n_ok,
        "final_wr0": final_wins0 / n_ok, "final_tie": final_ties / n_ok,
        "se": se,
    }


def main(radius=3, scorer_name="area_linear", n_random=400, n_greedy=150,
         seed_tag="parity_diag"):
    scorer = SCORERS[scorer_name]
    cells = hex_board(radius)
    equal_turn = len(cells) - 2  # 0-indexed turn after which tile counts are equal
    print(f"{'mech':<6}{'adj':<7}{'agent':<8}{'n':>5}{'equal_wr0':>11}"
          f"{'final_wr0':>11}{'delta':>8}{'~SE':>7}")
    print("-" * 70)
    t0 = time.time()
    for mech in ("bag", "pool", "hand"):
        for adj in (False, True):
            for agent in ("random", "greedy"):
                n = n_random if agent == "random" else n_greedy
                r = run(mech, adj, agent, scorer, cells, equal_turn, n, seed_tag)
                delta = r["final_wr0"] - r["equal_wr0"]
                print(f"{mech:<6}{str(adj):<7}{agent:<8}{r['n']:>5}{r['equal_wr0']:>11.3f}"
                      f"{r['final_wr0']:>11.3f}{delta:>8.3f}{r['se']:>7.3f}", flush=True)
    print(f"\nTOTAL WALL: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
