"""Train directly against a fixed Stage-2 greedy opponent (not self-play).

Why (HANDOFF.md 8.6): self-play plateaued at ~0% win rate vs greedy despite
learning real skill (crushes random). Two explanations are consistent with
that: (H1) the game's strategic ceiling really is near greedy's level, or
(H2) self-play stalled below basic competence because its only opponent was
an equally-mediocre copy of itself -- a known self-play failure mode. The
greedy-vs-greedy baseline (44.5%/49.3%/6.2% win/loss/draw, near-even) favors
H2: greedy isn't overwhelming, it ties itself, so a policy that can't even
score against it plausibly never reached greedy's own baseline competence.

Training directly against greedy removes that confound: every gradient is
computed against the exact opponent we're asking "can you beat this", so
there's no self-play mediocrity to hide behind. Only the learner's OWN action
steps go into the PPO batch -- greedy's steps are real environment
transitions (they still shape the returns) but were not sampled from the
policy, so training on them would be invalid (imitating a move the policy
never actually took). The learner's seat is re-randomized every episode so
it experiences both seats over training and seat-order effects cancel out
in aggregate (not just in eval, as train_selfplay.py's evaluate() does).
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from simulacrum.harness import require_fresh_report
from hex_truchet import ACTION_SPACE_SIZE, OBS_SIZE, N_CELLS
from hex_truchet.fast import HexTruchetBatched
from train_selfplay import PolicyValue, masked_dist, greedy_action, evaluate, MASK_DIM, HORIZON

ENV_ROOT = "/Users/mwburke/projects/hextruchet/hex_truchet"


def train(args):
    device = args.device
    ok = require_fresh_report(ENV_ROOT)
    if not ok:
        print("WARNING: validation gate not satisfied; continuing anyway for dev.")

    torch.manual_seed(args.seed)
    net = PolicyValue(hidden=args.hidden).to(device)
    if args.init_from:
        net.load_state_dict(torch.load(args.init_from, map_location=device))
        print(f"initialized from {args.init_from}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    env = HexTruchetBatched(args.n_envs, device=device)
    env.reset(torch.arange(args.n_envs, dtype=torch.int64, device=device) * 2_654_435_761 % (2**31))
    obs = env.observe()

    hist = []
    t_start = time.time()
    for update in range(1, args.updates + 1):
        # spec: re-randomize which seat the learner plays every episode, so
        # seat-order effects cancel over training, not just at eval time.
        learner_seat = torch.randint(0, 2, (args.n_envs,), device=device)

        b_obs, b_act, b_logp, b_val, b_learner_turn = [], [], [], [], []
        margin_p0 = None
        for t in range(HORIZON):
            mask = obs[:, -MASK_DIM:] > 0.5
            logits, value = net(obs)
            dist = masked_dist(logits, mask)
            pol_act = dist.sample()
            greedy_act = greedy_action(env, obs)
            is_learner_turn = (env.current_player == learner_seat)
            action = torch.where(is_learner_turn, pol_act, greedy_act)

            b_obs.append(obs)
            b_act.append(pol_act)                 # policy's OWN sample (used only where learner acted)
            b_logp.append(dist.log_prob(pol_act))
            b_val.append(value)
            b_learner_turn.append(is_learner_turn.clone())

            obs, rew, term, info = env.step(action)
            if t == HORIZON - 1:
                fs = info["final_state_json"]
                sp0 = torch.tensor([fs[i]["score_p0"] for i in range(env.n)],
                                   dtype=torch.float32, device=device)
                sp1 = torch.tensor([fs[i]["score_p1"] for i in range(env.n)],
                                   dtype=torch.float32, device=device)
                margin_p0 = sp0 - sp1

        obs_t = torch.stack(b_obs)                            # [T,N,749]
        act_t = torch.stack(b_act)                            # [T,N]
        logp_old = torch.stack(b_logp).detach()               # [T,N]
        val_t = torch.stack(b_val).detach()                   # [T,N]
        learner_turn_t = torch.stack(b_learner_turn)          # [T,N] bool

        # return, from the LEARNER's own perspective (learner_seat may be 0 or 1
        # per instance, fixed for this whole episode).
        learner_margin = torch.where(learner_seat == 0, margin_p0, -margin_p0)  # [N]
        ret_t = learner_margin.view(1, -1).expand(HORIZON, -1).to(torch.float32)  # [T,N]
        adv_t = ret_t - val_t

        # keep only the learner's OWN action steps -- these are the only steps
        # actually sampled from the policy (see module docstring).
        keep = learner_turn_t.reshape(-1)
        obs_f = obs_t.reshape(-1, OBS_SIZE)[keep]
        act_f = act_t.reshape(-1)[keep]
        logp_old_f = logp_old.reshape(-1)[keep]
        ret_f = ret_t.reshape(-1)[keep]
        adv_f = adv_t.reshape(-1)[keep]
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        mask_f = obs_f[:, -MASK_DIM:] > 0.5
        n_samples = obs_f.shape[0]

        last = {}
        for epoch in range(args.epochs):
            perm = torch.randperm(n_samples, device=device)
            for i in range(0, n_samples, args.minibatch):
                mb = perm[i:i + args.minibatch]
                logits, value = net(obs_f[mb])
                dist = masked_dist(logits, mask_f[mb])
                logp = dist.log_prob(act_f[mb])
                ratio = torch.exp(logp - logp_old_f[mb])
                a = adv_f[mb]
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
            win_rate = (learner_margin > 0).float().mean().item()
            row = {"update": update, "train_win_vs_greedy": win_rate,
                   "abs_margin": learner_margin.abs().mean().item(),
                   "mean_margin": learner_margin.mean().item(), **last, "sps": int(sps)}
            hist.append(row)
            print(f"upd {update:4d} | train_win {row['train_win_vs_greedy']:.3f} "
                  f"margin {row['mean_margin']:+6.2f} | pi {row['pi']:+.3f} "
                  f"v {row['v']:6.2f} ent {row['ent']:.3f} | {row['sps']:>6d} env-steps/s",
                  flush=True)

        if update % args.eval_every == 0 or update == args.updates:
            ev_greedy = evaluate(net, "greedy", n_games=args.eval_greedy_games, device=device)
            ev_rand = evaluate(net, "random", n_games=args.eval_rand_games, device=device)
            print(f"    [held-out eval] vs greedy: win {ev_greedy['win_rate']:.3f} "
                  f"draw {ev_greedy['draw_rate']:.3f} margin {ev_greedy['mean_margin']:+.2f}  "
                  f"||  vs random: win {ev_rand['win_rate']:.3f} margin "
                  f"{ev_rand['mean_margin']:+.2f}", flush=True)
            torch.save(net.state_dict(), args.out)

    print("\n=== FINAL HELD-OUT EVAL vs Stage-2 greedy ===", flush=True)
    ev_greedy = evaluate(net, "greedy", n_games=args.eval_greedy_games * 2, device=device)
    print(f"vs greedy: win_rate {ev_greedy['win_rate']:.3f} "
          f"draw {ev_greedy['draw_rate']:.3f} mean_margin {ev_greedy['mean_margin']:+.3f}")
    torch.save(net.state_dict(), args.out)
    print(f"saved policy to {args.out}")
    return hist, ev_greedy


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=256)   # greedy opponent cost dominates; smaller batch
    ap.add_argument("--updates", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.02)  # slightly higher: sparse win signal
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-greedy-games", type=int, default=256)
    ap.add_argument("--eval-rand-games", type=int, default=256)
    ap.add_argument("--init-from", type=str, default="",
                    help="optional checkpoint to warm-start from (e.g. the self-play policy)")
    ap.add_argument("--out", type=str, default="training/policy_vs_greedy.pt")
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
