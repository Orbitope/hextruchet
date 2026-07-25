"""Stage 1: scoring variants, applied to games with tracked ownership.

Ownership model: closure-credit rule. Placements are round-robined among
N players. When a placement closes one or more loops, the placing player
owns those loops. We identify "which loop closed" by diffing the set of
loop-defining arc-sets before and after each placement (a loop's identity
is the frozenset of its arcs, which is stable once closed).

This is still random play (Stage 2 adds deliberate agents) -- the point
here is purely to test whether each scoring RULE produces separation,
using the SAME underlying random games for every rule (fair comparison).
"""

import random
import numpy as np
from collections import Counter, defaultdict

from geometry import hex_board, canonical_tiles
from graph import Board
from stage0 import enclosed_cells

TILES = canonical_tiles()

# Winning deck from Stage 0: tile 0 : tile 2 = 1 : 2
DECK_WEIGHTS = [1, 0, 2, 0, 0]


def play_random_game(cells, n_players, rng, deck_weights=DECK_WEIGHTS):
    """Round-robin random placement. Returns (board, loop_records).

    loop_records: list of dicts {arcs (frozenset), length, area, owner, turn}
    """
    order = list(cells)
    rng.shuffle(order)
    idx = list(range(len(TILES)))

    b = Board(cells)
    prev_loop_keys = set()
    loop_records = []

    for turn, cell in enumerate(order):
        player = turn % n_players
        t = rng.choices(idx, weights=deck_weights)[0]
        rot = rng.randrange(6)
        b.place(cell, TILES[t]["matching"], rot)

        loops = b.loops()
        cur_keys = {frozenset(l["arcs"]): l for l in loops}
        new_keys = set(cur_keys) - prev_loop_keys
        for k in new_keys:
            l = cur_keys[k]
            area = len(enclosed_cells(b, l))
            loop_records.append({
                "arcs": k, "length": l["length"], "area": area,
                "owner": player, "turn": turn,
            })
        prev_loop_keys = set(cur_keys)

    return b, loop_records


# ---------------------------------------------------------------------
# Scoring variants. Each takes loop_records + n_players, returns scores.
# ---------------------------------------------------------------------

def score_length(records, n_players):
    s = [0] * n_players
    for r in records:
        s[r["owner"]] += r["length"]
    return s


def score_area(records, n_players):
    s = [0] * n_players
    for r in records:
        s[r["owner"]] += r["area"]
    return s


def score_length_superlinear(records, n_players, power=1.5):
    s = [0.0] * n_players
    for r in records:
        s[r["owner"]] += r["length"] ** power
    return s


def score_area_superlinear(records, n_players, power=1.5):
    s = [0.0] * n_players
    for r in records:
        s[r["owner"]] += (r["area"] + 1) ** power  # +1 so area=0 still counts
    return s


def score_length_plus_area(records, n_players, k=1.0):
    s = [0.0] * n_players
    for r in records:
        s[r["owner"]] += r["length"] + k * r["area"]
    return s


def score_area_bonus_for_big(records, n_players, threshold=6, bonus=10):
    """Flat points per loop, PLUS a big bonus for loops at/above threshold
    length. Tests whether a discrete 'jackpot' for rare big loops creates
    separation even when most loops are minimal."""
    s = [0.0] * n_players
    for r in records:
        s[r["owner"]] += 1  # participation point per loop closed
        if r["length"] >= threshold:
            s[r["owner"]] += bonus
    return s


def score_count_only(records, n_players):
    """Pure loop count -- ignoring size entirely. Sanity baseline: if this
    performs similarly to length scoring, size doesn't matter much given
    how dominant minimal loops are."""
    s = [0] * n_players
    for r in records:
        s[r["owner"]] += 1
    return s


VARIANTS = {
    "count_only":            score_count_only,
    "length_linear":         score_length,
    "area_linear":           score_area,
    "length_superlinear_1.5": lambda r, n: score_length_superlinear(r, n, 1.5),
    "length_superlinear_2.0": lambda r, n: score_length_superlinear(r, n, 2.0),
    "area_superlinear_1.5":  lambda r, n: score_area_superlinear(r, n, 1.5),
    "length_plus_area":      lambda r, n: score_length_plus_area(r, n, 1.0),
    "jackpot_len>=6":        lambda r, n: score_area_bonus_for_big(r, n, 6, 10),
    "jackpot_len>=9":        lambda r, n: score_area_bonus_for_big(r, n, 9, 15),
}


# ---------------------------------------------------------------------
# Separation metrics
# ---------------------------------------------------------------------

def separation_metrics(all_scores):
    """all_scores: list of per-game score arrays (each length n_players).
    Returns metrics on how much variants distinguish players.
    """
    arr = np.array(all_scores, dtype=float)  # (n_games, n_players)
    # normalize each game's scores to sum=1 (or leave 0 if all zero) so we
    # measure RELATIVE separation, not absolute scale
    totals = arr.sum(axis=1, keepdims=True)
    nonzero = (totals[:, 0] > 0)
    if nonzero.sum() == 0:
        return {"mean_winner_margin": 0.0, "frac_tied": 1.0,
                "cv_of_winner_share": 0.0, "frac_all_zero": 1.0}
    normed = np.zeros_like(arr)
    normed[nonzero] = arr[nonzero] / totals[nonzero]

    sorted_normed = np.sort(normed[nonzero], axis=1)[:, ::-1]
    winner_share = sorted_normed[:, 0]
    margin = sorted_normed[:, 0] - sorted_normed[:, 1]  # winner minus runner-up

    n_players = arr.shape[1]
    tie_thresh = 0.02
    frac_tied = float((margin < tie_thresh).mean())

    return {
        "mean_winner_share": float(winner_share.mean()),
        "mean_margin": float(margin.mean()),
        "frac_tied": frac_tied,
        "winner_share_std": float(winner_share.std()),
        "frac_all_zero": float(1 - nonzero.mean()),
    }


def run_stage1(radius, n_players, n_games, seed=0):
    rng = random.Random(seed)
    cells = hex_board(radius)

    per_variant_scores = defaultdict(list)
    all_records = []

    for g in range(n_games):
        board, records = play_random_game(cells, n_players, rng)
        all_records.append(records)
        for name, fn in VARIANTS.items():
            per_variant_scores[name].append(fn(records, n_players))

    print(f"\n{'='*88}")
    print(f"STAGE 1 -- radius {radius}, {n_players} players, {n_games} games")
    print(f"{'='*88}")
    print(f"{'variant':<26}{'win share':>11}{'margin':>10}{'frac tied':>11}"
          f"{'share SD':>10}{'P(0-0)':>9}")
    print("-"*88)
    for name in VARIANTS:
        m = separation_metrics(per_variant_scores[name])
        print(f"{name:<26}{m['mean_winner_share']:>11.3f}{m['mean_margin']:>10.3f}"
              f"{m['frac_tied']:>11.3f}{m['winner_share_std']:>10.3f}"
              f"{m['frac_all_zero']:>9.3f}")

    return per_variant_scores, all_records


if __name__ == "__main__":
    import sys
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    for radius in (3, 4):
        for n_players in (2, 3):
            run_stage1(radius, n_players, n_games, seed=radius*10+n_players)
