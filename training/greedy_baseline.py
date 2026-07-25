"""Baseline matchups with NO learned policy: greedy vs greedy, greedy vs random.

If greedy-vs-greedy is ~50/50 (symmetric, seat-rotated), then 'play greedy' is
the equilibrium and beating greedy is near-impossible => the Stage 3 shallow-
ceiling read is confirmed structurally, independent of any RL training. Also
reports absolute scores to show how much of the board each strategy captures."""
import torch
from hex_truchet.fast import HexTruchetBatched
from hex_truchet import ACTION_SPACE_SIZE
from train_selfplay import greedy_action, MASK_DIM, HORIZON


def random_action(env, obs, device):
    mask = obs[:, -MASK_DIM:] > 0.5
    u = torch.rand(env.n, ACTION_SPACE_SIZE, device=device).masked_fill(~mask, -1)
    return u.argmax(-1)


@torch.no_grad()
def match(seat0_fn, seat1_fn, n_games=384, device="cpu", seed_base=9_000_000):
    env = HexTruchetBatched(n_games, device=device)
    env.reset(torch.arange(n_games, dtype=torch.int64, device=device) + seed_base)
    obs = env.observe()
    sp0 = sp1 = None
    for t in range(HORIZON):
        a0 = seat0_fn(env, obs, device)
        a1 = seat1_fn(env, obs, device)
        action = torch.where(env.current_player == 0, a0, a1)
        obs, rew, term, info = env.step(action)
        if bool(term.all()):
            fs = info["final_state_json"]
            sp0 = torch.tensor([fs[i]["score_p0"] for i in range(env.n)], dtype=torch.float32)
            sp1 = torch.tensor([fs[i]["score_p1"] for i in range(env.n)], dtype=torch.float32)
    margin = sp0 - sp1  # seat-0 perspective
    return {
        "seat0_win": (margin > 0).float().mean().item(),
        "draw": (margin == 0).float().mean().item(),
        "seat0_score": sp0.mean().item(),
        "seat1_score": sp1.mean().item(),
        "margin": margin.mean().item(),
    }


def greedy_fn(env, obs, device):
    return greedy_action(env, obs)


if __name__ == "__main__":
    print("=== baseline matchups (seat-0 perspective) ===")
    r = match(greedy_fn, greedy_fn)
    print(f"greedy vs greedy : seat0_win {r['seat0_win']:.3f} draw {r['draw']:.3f} | "
          f"scores {r['seat0_score']:.2f} / {r['seat1_score']:.2f} | margin {r['margin']:+.2f}")
    r = match(greedy_fn, random_action)
    print(f"greedy vs random : seat0_win {r['seat0_win']:.3f} draw {r['draw']:.3f} | "
          f"scores {r['seat0_score']:.2f} / {r['seat1_score']:.2f} | margin {r['margin']:+.2f}")
    r = match(random_action, random_action)
    print(f"random vs random : seat0_win {r['seat0_win']:.3f} draw {r['draw']:.3f} | "
          f"scores {r['seat0_score']:.2f} / {r['seat1_score']:.2f} | margin {r['margin']:+.2f}")
