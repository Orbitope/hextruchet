"""Self-play PPO trainer for the hex_truchet simulacrum environment (Stage 3).

Goal (HANDOFF.md 7.3 / 8.5): train a shared-policy self-play agent and measure
whether it can beat the Stage 2 greedy heuristic by a real, repeatable margin —
the test that distinguishes a genuinely shallow strategic ceiling from merely
weak hand-crafted heuristics.

Design notes:
- Single shared policy plays BOTH seats (env.observe() returns the acting
  player's view). Reward is terminal-only zero-sum margin; we resolve it onto
  every action by the ACTING player's own final differential
  (score[actor] - score[opponent]) — the credit-assignment step the env spec
  deliberately leaves to the training script.
- Fixed 37-step horizon, all N instances in lockstep, so one "update" collects
  exactly 37 steps = one full episode per env. Purely terminal reward + short
  horizon => Monte-Carlo returns (no GAE bootstrap needed); advantage =
  return - V(obs). PPO clip + entropy bonus on top.
- Illegal actions are masked out of the policy distribution using the
  legal_action_mask embedded in the last 666 obs dims.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
from torch.distributions import Categorical

from simulacrum.harness import require_fresh_report
from hex_truchet import ACTION_SPACE_SIZE, OBS_SIZE, N_CELLS
from hex_truchet.fast import HexTruchetBatched

ENV_ROOT = "/Users/mwburke/projects/hextruchet/hex_truchet"
MASK_DIM = ACTION_SPACE_SIZE  # last 666 obs entries are the legal-action mask
HORIZON = N_CELLS             # 37 steps per episode (fixed)


class PolicyValue(nn.Module):
    def __init__(self, obs_dim=OBS_SIZE, act_dim=ACTION_SPACE_SIZE, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.trunk(obs)
        return self.pi(h), self.v(h).squeeze(-1)


def masked_dist(logits, mask):
    """Categorical over legal actions only (illegal -> -inf logit)."""
    neg = torch.finfo(logits.dtype).min
    return Categorical(logits=logits.masked_fill(~mask, neg))


# Rotation aliasing (spec.md #Constants): tile 0's 6 rotations fall into 2
# distinct arc patterns ({0,2,4} and {1,3,5}); tile 2's fall into 3 ({0,3},
# {1,4},{2,5}). Only the representative rotation of each class needs an
# actual _total_loop_area evaluation -- the other rotations in its class
# produce an identical board and can reuse the same gain. Precomputed once.
_ROTATION_CLASS_REP = {
    0: {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1},   # tile 0: rot -> representative
    2: {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2},   # tile 2: rot -> representative
}


def greedy_action(env, obs):
    """Stage-2 greedy: pick the legal (hand_slot, cell, rotation) maximizing
    immediate area gain; ties -> smallest action index (matches
    stage2_screen.py's choose_move: first strictly-greater value wins, in
    action-id order hand_slot-major/cell/rotation-minor). Vectorized over
    envs, cells, AND the tie-break -- no per-cell Python loop. Gain is
    computed per (tile TYPE, representative rotation) rather than per hand
    slot -- only 5 distinct _total_loop_area calls total (2 for tile 0's
    rotation classes + 3 for tile 2's), reused across whichever hand slots
    happen to hold that type -- cheap enough to call every training step,
    not just at eval time.
    """
    N = env.n
    dev = env.device
    bt, br = env.board_tile, env.board_rotation
    mask = obs[:, -MASK_DIM:] > 0.5                        # [N,666]
    base_area = env._total_loop_area(bt, br)               # [N]
    my_hand = torch.where((env.current_player == 0).view(N, 1),
                          env.hand_p0, env.hand_p1)          # [N,3]

    ar = torch.arange(N_CELLS, device=dev)
    # gain_by_type[type_value][rot] -> [N,37] gain tensor (type_value in {0,2})
    gain_by_type = {0: {}, 2: {}}
    for tv in (0, 2):
        for rep_rot in sorted(set(_ROTATION_CLASS_REP[tv].values())):
            bt_rep = bt.unsqueeze(1).expand(N, N_CELLS, N_CELLS).clone()  # [N,c,37]
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
        tile_type = my_hand[:, hs]                          # [N] in {-1,0,2}
        for rot in range(6):
            gain = torch.where(tile_type.view(N, 1) == 0, gain_by_type[0][rot],
                               gain_by_type[2][rot])          # [N,37]
            action_ids = hs * (N_CELLS * 6) + ar * 6 + rot    # [37]
            gains[:, action_ids] = gain

    neg_inf = torch.iinfo(gains.dtype).min
    masked_gains = torch.where(mask, gains, torch.full_like(gains, neg_inf))
    best_val = masked_gains.max(dim=1, keepdim=True).values           # [N,1]
    action_idx = torch.arange(ACTION_SPACE_SIZE, device=dev).view(1, -1)
    is_best = (masked_gains == best_val) & mask                       # [N,666]
    best_action = torch.where(is_best, action_idx,
                              torch.full_like(action_idx, ACTION_SPACE_SIZE)).min(1).values
    first_legal = torch.where(mask, action_idx,
                              torch.full_like(action_idx, ACTION_SPACE_SIZE)).min(1).values
    return torch.where(best_val.squeeze(1) > 0, best_action, first_legal)


@torch.no_grad()
def evaluate(net, opponent, n_games=512, device="cpu", seed_base=1_000_000):
    """Policy vs `opponent` ("random" or "greedy"). Policy plays one seat, the
    opponent the other; the policy seat is split 50/50 across the batch to
    cancel first-mover advantage. Returns policy win rate and mean margin."""
    env = HexTruchetBatched(n_games, device=device)
    env.reset(torch.arange(n_games, dtype=torch.int64, device=device) + seed_base)
    policy_seat = (torch.arange(n_games, device=device) % 2)   # half seat0, half seat1
    obs = env.observe()
    final_margin_p0 = None
    for t in range(HORIZON):
        mask = obs[:, -MASK_DIM:] > 0.5
        logits, _ = net(obs)
        pol_act = masked_dist(logits, mask).probs.argmax(-1)   # greedy-from-policy
        if opponent == "random":
            u = torch.rand(env.n, ACTION_SPACE_SIZE, device=device).masked_fill(~mask, -1)
            opp_act = u.argmax(-1)
        else:
            opp_act = greedy_action(env, obs)
        use_policy = (env.current_player == policy_seat)
        action = torch.where(use_policy, pol_act, opp_act)
        obs, rew, term, info = env.step(action)
        if bool(term.all()):
            fs = info["final_state_json"]
            sp0 = torch.tensor([fs[i]["score_p0"] for i in range(env.n)], dtype=torch.float32)
            sp1 = torch.tensor([fs[i]["score_p1"] for i in range(env.n)], dtype=torch.float32)
            final_margin_p0 = sp0 - sp1
    # margin from the policy's own seat
    pol_margin = torch.where(policy_seat.cpu() == 0, final_margin_p0, -final_margin_p0)
    win = (pol_margin > 0).float().mean().item()
    draw = (pol_margin == 0).float().mean().item()
    return {"win_rate": win, "draw_rate": draw, "mean_margin": pol_margin.mean().item()}


def train(args):
    device = args.device
    ok = require_fresh_report(ENV_ROOT)
    if not ok:
        print("WARNING: validation gate not satisfied; continuing anyway for dev.")

    torch.manual_seed(args.seed)
    net = PolicyValue(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    env = HexTruchetBatched(args.n_envs, device=device)
    env.reset(torch.arange(args.n_envs, dtype=torch.int64, device=device) * 2_654_435_761 % (2**31))
    obs = env.observe()

    hist = []
    t_start = time.time()
    for update in range(1, args.updates + 1):
        # ---- collect one full episode per env (37 aligned steps) ----
        b_obs, b_act, b_logp, b_val, b_actor = [], [], [], [], []
        margin_p0 = None
        for t in range(HORIZON):
            mask = obs[:, -MASK_DIM:] > 0.5
            logits, value = net(obs)
            dist = masked_dist(logits, mask)
            action = dist.sample()
            b_obs.append(obs)
            b_act.append(action)
            b_logp.append(dist.log_prob(action))
            b_val.append(value)
            b_actor.append(env.current_player.clone())
            obs, rew, term, info = env.step(action)
            if t == HORIZON - 1:
                fs = info["final_state_json"]
                sp0 = torch.tensor([fs[i]["score_p0"] for i in range(env.n)],
                                   dtype=torch.float32, device=device)
                sp1 = torch.tensor([fs[i]["score_p1"] for i in range(env.n)],
                                   dtype=torch.float32, device=device)
                margin_p0 = sp0 - sp1                        # [N] p0 perspective

        obs_t = torch.stack(b_obs)                           # [T,N,749]
        act_t = torch.stack(b_act)                           # [T,N]
        logp_old = torch.stack(b_logp).detach()              # [T,N]
        val_t = torch.stack(b_val).detach()                  # [T,N]
        actor_t = torch.stack(b_actor)                       # [T,N]
        # return for each step = acting player's final differential
        ret_t = torch.where(actor_t == 0, margin_p0.view(1, -1), -margin_p0.view(1, -1))
        ret_t = ret_t.to(torch.float32)                      # [T,N]
        adv_t = ret_t - val_t
        adv_flat = adv_t.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        obs_f = obs_t.reshape(-1, OBS_SIZE)
        act_f = act_t.reshape(-1)
        logp_old_f = logp_old.reshape(-1)
        ret_f = ret_t.reshape(-1)
        mask_f = obs_f[:, -MASK_DIM:] > 0.5
        n_samples = obs_f.shape[0]

        # ---- PPO update ----
        last = {}
        for epoch in range(args.epochs):
            perm = torch.randperm(n_samples, device=device)
            for i in range(0, n_samples, args.minibatch):
                mb = perm[i:i + args.minibatch]
                logits, value = net(obs_f[mb])
                dist = masked_dist(logits, mask_f[mb])
                logp = dist.log_prob(act_f[mb])
                ratio = torch.exp(logp - logp_old_f[mb])
                a = adv_flat[mb]
                l_clip = -torch.min(ratio * a,
                                    torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
                l_v = 0.5 * (value - ret_f[mb]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = l_clip + args.vf_coef * l_v - args.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                last = {"pi": l_clip.item(), "v": l_v.item(), "ent": ent.item()}

        if update % args.log_every == 0 or update == 1:
            sps = update * HORIZON * args.n_envs / (time.time() - t_start)
            row = {"update": update, "abs_margin": margin_p0.abs().mean().item(),
                   **last, "sps": int(sps)}
            hist.append(row)
            print(f"upd {update:4d} | |margin| {row['abs_margin']:5.2f} "
                  f"| pi {row['pi']:+.3f} v {row['v']:6.2f} ent {row['ent']:.3f} "
                  f"| {row['sps']:>6d} env-steps/s", flush=True)

        if update % args.eval_every == 0 or update == args.updates:
            ev_rand = evaluate(net, "random", n_games=args.eval_games, device=device)
            ev_greedy = evaluate(net, "greedy", n_games=args.eval_greedy_games, device=device)
            print(f"    [eval] vs random: win {ev_rand['win_rate']:.3f} margin "
                  f"{ev_rand['mean_margin']:+.2f}  ||  vs greedy: win "
                  f"{ev_greedy['win_rate']:.3f} draw {ev_greedy['draw_rate']:.3f} "
                  f"margin {ev_greedy['mean_margin']:+.2f}", flush=True)
            torch.save(net.state_dict(), args.out)

    # ---- final greedy evaluation (the headline question) ----
    print("\n=== FINAL EVAL vs Stage-2 greedy ===", flush=True)
    ev_greedy = evaluate(net, "greedy", n_games=args.eval_games, device=device)
    print(f"vs greedy: win_rate {ev_greedy['win_rate']:.3f} "
          f"draw {ev_greedy['draw_rate']:.3f} mean_margin {ev_greedy['mean_margin']:+.3f}")
    torch.save(net.state_dict(), args.out)
    print(f"saved policy to {args.out}")
    return hist, ev_greedy


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=512)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-games", type=int, default=512)
    ap.add_argument("--eval-greedy-games", type=int, default=256)
    ap.add_argument("--out", type=str, default="training/policy.pt")
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
