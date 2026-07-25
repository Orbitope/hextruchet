"""Search-based bot: N candidate first-moves + greedy-rollout evaluation,
tested head-to-head against plain greedy. NOT RL -- a deterministic, fully
enumerable lookahead, built to give a confound-free answer to the Stage 3
question (HANDOFF.md 7.3/8.6): is greedy close to a real ceiling, or is there
exploitable depth a shallow search finds easily?

Why this design (a "rollout algorithm", not full minimax): the opponent in
this test IS greedy -- fixed and deterministic -- so there's no need to
search over the opponent's possible replies (that's what makes minimax
expensive). Instead: at my turn, take the top-K legal candidate moves by
immediate area gain (same candidate-generation as greedy_action), then for
EACH candidate simulate the rest of the game with BOTH sides playing plain
greedy from that point on, and keep whichever candidate produced the best
actual final margin. This is a well-known "rollout algorithm" / one step of
policy improvement over the greedy base policy (Bertsekas) -- cheap because
the environment is small and fast, and gives a TRUE lookahead value (not a
hand-tuned 1-ply heuristic like Stage 2's denial agent).

Caveat (disclosed, not hidden): this uses full information -- it "sees" the
opponent's exact hand when rolling out, which the real private-hand rule
would not allow a human/policy to do. This is deliberate: it upper-bounds
how much a lookahead approach could exploit greedy under perfect information.
If even THIS can't beat greedy by much, that's strong evidence for a real
ceiling (less information can only make it harder, not easier). If it CAN,
that shows real exploitable structure exists, but doesn't by itself prove a
private-information-respecting agent could find it.
"""
import torch

from hex_truchet import ACTION_SPACE_SIZE, N_CELLS
from hex_truchet.fast import HexTruchetBatched
from train_selfplay import (
    _ROTATION_CLASS_REP, greedy_action, MASK_DIM, HORIZON,
)

_ACT_CELLROT = N_CELLS * 6


def _candidate_gains(env):
    """[1,666] gain tensor for the single instance in env (n=1), reusing
    greedy_action's type-based gain caching. Returns (gains, mask)."""
    N = env.n
    dev = env.device
    bt, br = env.board_tile, env.board_rotation
    obs = env.observe()
    mask = obs[:, -MASK_DIM:] > 0.5
    base_area = env._total_loop_area(bt, br)
    my_hand = torch.where((env.current_player == 0).view(N, 1), env.hand_p0, env.hand_p1)

    ar = torch.arange(N_CELLS, device=dev)
    gain_by_type = {0: {}, 2: {}}
    for tv in (0, 2):
        for rep_rot in sorted(set(_ROTATION_CLASS_REP[tv].values())):
            bt_rep = bt.unsqueeze(1).expand(N, N_CELLS, N_CELLS).clone()
            br_rep = br.unsqueeze(1).expand(N, N_CELLS, N_CELLS).clone()
            bt_rep[:, ar, ar] = tv
            br_rep[:, ar, ar] = rep_rot
            areas = env._total_loop_area(bt_rep.reshape(N * N_CELLS, N_CELLS),
                                         br_rep.reshape(N * N_CELLS, N_CELLS))
            gain_by_type[tv][rep_rot] = areas.reshape(N, N_CELLS) - base_area.view(N, 1)
        for rot in range(6):
            gain_by_type[tv][rot] = gain_by_type[tv][_ROTATION_CLASS_REP[tv][rot]]

    gains = torch.zeros(N, ACTION_SPACE_SIZE, device=dev, dtype=torch.int64)
    for hs in range(3):
        tile_type = my_hand[:, hs]
        for rot in range(6):
            gain = torch.where(tile_type.view(N, 1) == 0, gain_by_type[0][rot], gain_by_type[2][rot])
            action_ids = hs * _ACT_CELLROT + ar * 6 + rot
            gains[:, action_ids] = gain
    return gains, mask


def top_k_actions(env, K):
    """Top-K distinct legal actions by immediate area gain (ties broken by
    smallest action index, descending gain first). Returns a 1D LongTensor,
    length <= K (fewer if fewer legal actions exist)."""
    gains, mask = _candidate_gains(env)
    gains = gains[0]
    mask = mask[0]
    neg = torch.iinfo(gains.dtype).min
    scored = torch.where(mask, gains, torch.full_like(gains, neg))
    order = torch.argsort(scored, descending=True, stable=True)  # stable -> smallest idx wins ties
    legal_order = order[mask[order]]
    return legal_order[:K]


def _clone_to_batch(env1, K):
    """New HexTruchetBatched(K) with env1's single (n=1) state replicated K
    times, bypassing reset() (which would re-deal hands) -- for branching a
    live mid-game state into K parallel what-if continuations."""
    dev = env1.device
    b = HexTruchetBatched(K, device=dev)
    b.seeds = env1.seeds.repeat(K)
    b.episodes = env1.episodes.repeat(K)
    b.keys = env1.keys.repeat(K)
    b.t = env1.t.repeat(K)
    b.board_tile = env1.board_tile.repeat(K, 1)
    b.board_rotation = env1.board_rotation.repeat(K, 1)
    b.hand_p0 = env1.hand_p0.repeat(K, 1)
    b.hand_p1 = env1.hand_p1.repeat(K, 1)
    b.score_p0 = env1.score_p0.repeat(K)
    b.score_p1 = env1.score_p1.repeat(K)
    b.current_player = env1.current_player.repeat(K)
    return b


@torch.no_grad()
def lookahead_action(env1, K=8, depth=0, return_value=False):
    """The move lookahead_bot plays at env1's current (single-instance) state:
    branch into up to K candidates, roll each out with greedy on both sides,
    return the action (of the K tried) with the best margin for the acting seat.

    K     -- how many candidate first-moves to try (by immediate area gain).
             Dominant strength AND cost knob. K=1 is exactly greedy.
    depth -- plies to roll out after the candidate move. 0 = roll to game end
             (strongest, most expensive). A positive value truncates the
             rollout and scores by margin-so-far, which is the main way to cut
             cost in the early game (a full rollout from t=4 is ~33 plies; from
             t=30 it's only ~7, so late moves are cheap either way).

    Falls back to greedy_action only when there are zero legal actions (should
    not happen for t<37, spec.md Termination); return_value gives None there.

    return_value=True also returns the chosen candidate's rollout margin -- a
    free Monte-Carlo value estimate the search already computed.
    """
    my_seat = int(env1.current_player[0])
    cands = top_k_actions(env1, K)
    if cands.numel() == 0:
        a = greedy_action(env1, env1.observe())
        return (a, None) if return_value else a

    M = cands.numel()
    branch = _clone_to_batch(env1, M)
    obs, rew, term, info = branch.step(cands)  # each branch plays its own candidate
    plies = 1
    while not bool(term.all()) and (depth <= 0 or plies < depth):
        obs, rew, term, info = branch.step(greedy_action(branch, obs))
        plies += 1
    # Score from the live tensors (works for both truncated and terminal
    # rollouts; terminal JSON would only cover the roll-to-end case).
    sp0 = branch.score_p0.to(torch.float32)
    sp1 = branch.score_p1.to(torch.float32)
    margin = sp0 - sp1 if my_seat == 0 else sp1 - sp0
    best = int(torch.argmax(margin))
    action = cands[best].view(1)
    return (action, float(margin[best])) if return_value else action


@torch.no_grad()
def play_one_game(K, seat, seed, device="cpu"):
    """One game: `seat` plays lookahead_bot(K), the other seat plays plain
    greedy. Returns (lookahead_score, greedy_score)."""
    env = HexTruchetBatched(1, device=device)
    env.reset(torch.tensor([seed], dtype=torch.int64, device=device))
    obs = env.observe()
    fs = None
    while True:
        if int(env.current_player[0]) == seat:
            a = lookahead_action(env, K=K)
        else:
            a = greedy_action(env, obs)
        obs, rew, term, info = env.step(a)
        if bool(term.all()):
            fs = info["final_state_json"][0]
            break
    la_score = fs["score_p0"] if seat == 0 else fs["score_p1"]
    gr_score = fs["score_p1"] if seat == 0 else fs["score_p0"]
    return la_score, gr_score


def match(K, n_games, device="cpu", seed_base=20_000_000):
    """lookahead_bot(K) vs greedy, seat rotated every game to cancel
    seat-order effects (per HANDOFF.md 7.4's noise caution)."""
    wins = draws = 0
    la_scores, gr_scores = [], []
    for g in range(n_games):
        seat = g % 2
        la, gr = play_one_game(K, seat, seed_base + g, device=device)
        la_scores.append(la); gr_scores.append(gr)
        if la > gr: wins += 1
        elif la == gr: draws += 1
    n = n_games
    return {
        "win": wins / n, "draw": draws / n,
        "la_score": sum(la_scores) / n, "gr_score": sum(gr_scores) / n,
        "margin": (sum(la_scores) - sum(gr_scores)) / n,
    }


if __name__ == "__main__":
    import sys, time
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    t0 = time.time()
    r = match(K, n_games)
    dt = time.time() - t0
    print(f"lookahead(K={K}) vs greedy, n={n_games} games, {dt:.1f}s ({dt/n_games:.2f}s/game)")
    print(f"  win {r['win']:.3f}  draw {r['draw']:.3f}  "
          f"scores {r['la_score']:.2f} / {r['gr_score']:.2f}  margin {r['margin']:+.2f}")
