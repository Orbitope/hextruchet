"""Richer eval of a saved policy vs greedy: absolute scores + argmax vs sampled,
to distinguish 'scores nothing' (distribution shift / flailing) from
'scores but less' (genuinely outplayed). Cheap one-shot diagnostic."""
import sys
import torch
from hex_truchet.fast import HexTruchetBatched
from hex_truchet import ACTION_SPACE_SIZE, N_CELLS
from train_selfplay import PolicyValue, masked_dist, greedy_action, MASK_DIM, HORIZON


@torch.no_grad()
def rich_eval(net, opponent, n_games=512, device="cpu", sample=False, seed_base=5_000_000):
    env = HexTruchetBatched(n_games, device=device)
    env.reset(torch.arange(n_games, dtype=torch.int64, device=device) + seed_base)
    policy_seat = (torch.arange(n_games, device=device) % 2)
    obs = env.observe()
    sp0 = sp1 = None
    for t in range(HORIZON):
        mask = obs[:, -MASK_DIM:] > 0.5
        logits, _ = net(obs)
        dist = masked_dist(logits, mask)
        pol_act = dist.sample() if sample else dist.probs.argmax(-1)
        if opponent == "random":
            u = torch.rand(env.n, ACTION_SPACE_SIZE, device=device).masked_fill(~mask, -1)
            opp_act = u.argmax(-1)
        else:
            opp_act = greedy_action(env, obs)
        action = torch.where(env.current_player == policy_seat, pol_act, opp_act)
        obs, rew, term, info = env.step(action)
        if bool(term.all()):
            fs = info["final_state_json"]
            sp0 = torch.tensor([fs[i]["score_p0"] for i in range(env.n)], dtype=torch.float32)
            sp1 = torch.tensor([fs[i]["score_p1"] for i in range(env.n)], dtype=torch.float32)
    ps = policy_seat.cpu()
    pol_score = torch.where(ps == 0, sp0, sp1)
    opp_score = torch.where(ps == 0, sp1, sp0)
    margin = pol_score - opp_score
    return {
        "win": (margin > 0).float().mean().item(),
        "draw": (margin == 0).float().mean().item(),
        "pol_score": pol_score.mean().item(),
        "opp_score": opp_score.mean().item(),
        "margin": margin.mean().item(),
        "pol_score_max": pol_score.max().item(),
    }


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "training/policy.pt"
    net = PolicyValue()
    net.load_state_dict(torch.load(ckpt))
    net.eval()
    print(f"checkpoint: {ckpt}")
    for opp in ("random", "greedy"):
        for sample in (False, True):
            r = rich_eval(net, opp, n_games=384, sample=sample)
            tag = "sampled" if sample else "argmax "
            print(f"  vs {opp:6} [{tag}]: win {r['win']:.3f} draw {r['draw']:.3f} | "
                  f"pol_score {r['pol_score']:5.2f} (max {r['pol_score_max']:.0f}) "
                  f"opp_score {r['opp_score']:5.2f} | margin {r['margin']:+.2f}")
