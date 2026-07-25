"""Generate a distillation dataset: (observation, lookahead-bot action, value)
triples, for training a fast student net to imitate lookahead_bot (Stage 3,
HANDOFF.md 8.10 #2).

Design: rather than playing full games with the lookahead bot making every
decision (all of a game's lookahead cost is spent on ONE self-generated
trajectory), generate cheap source games first (greedy/random agents, no
lookahead cost at all) for state DIVERSITY, then query the lookahead bot
ONCE per sampled state for the expert label. Total lookahead compute is
about the same either way (cost scales with turns-remaining-from-the-state,
same range either way) -- the win is state coverage: labeled examples come
from many different opponent pairings and trajectories, not only from
whatever board states the lookahead bot's own play would visit.

The lookahead bot's own rollout search already computes, for its chosen
action, the resulting final margin (a free Monte-Carlo value estimate) --
recorded alongside the action so the student's value head has a target,
useful for RL fine-tuning after distillation (HANDOFF.md 8.10 #3).

Resumable: --out appends to an existing dataset file if present, so this can
be re-run with a different --seed-base to keep growing the dataset.
"""
import argparse
import os
import time

import torch

from hex_truchet import N_CELLS, OBS_SIZE
from hex_truchet.fast import HexTruchetBatched
from train_selfplay import greedy_action, MASK_DIM, HORIZON
from lookahead_bot import lookahead_action

SAMPLE_TS = list(range(2, HORIZON - 1, 2))  # t=2,4,...,34: skip trivial empty-board
                                             # and near-terminal (<=2 tiles left) states


def random_action(env, obs):
    mask = obs[:, -MASK_DIM:] > 0.5
    u = torch.rand(env.n, mask.shape[1], device=env.device).masked_fill(~mask, -1)
    return u.argmax(-1)


AGENTS = {"greedy": greedy_action, "random": random_action}


@torch.no_grad()
def play_and_snapshot(seat0, seat1, n_games, seed_base, device="cpu"):
    """Play n_games games with fixed agents per seat; return a dict of
    [HORIZON, n_games, ...] tensors -- the full state BEFORE each step."""
    fn0, fn1 = AGENTS[seat0], AGENTS[seat1]
    env = HexTruchetBatched(n_games, device=device)
    env.reset(torch.arange(n_games, dtype=torch.int64, device=device) + seed_base)
    obs = env.observe()
    fields = ["board_tile", "board_rotation", "hand_p0", "hand_p1",
              "score_p0", "score_p1", "current_player", "t", "keys"]
    snaps = {f: [] for f in fields}
    for _ in range(HORIZON):
        for f in fields:
            snaps[f].append(getattr(env, f).clone())
        a0 = fn0(env, obs)
        a1 = fn1(env, obs)
        action = torch.where(env.current_player == 0, a0, a1)
        obs, rew, term, info = env.step(action)
    return {f: torch.stack(snaps[f]) for f in fields}  # [HORIZON, n_games, ...]


def _build_single_env(snaps, t_idx, g_idx, device="cpu"):
    b = HexTruchetBatched(1, device=device)
    b.seeds = torch.zeros(1, dtype=torch.int64, device=device)
    b.episodes = torch.zeros(1, dtype=torch.int64, device=device)
    b.keys = snaps["keys"][t_idx, g_idx:g_idx + 1].clone()
    b.t = snaps["t"][t_idx, g_idx:g_idx + 1].clone()
    b.board_tile = snaps["board_tile"][t_idx, g_idx:g_idx + 1].clone()
    b.board_rotation = snaps["board_rotation"][t_idx, g_idx:g_idx + 1].clone()
    b.hand_p0 = snaps["hand_p0"][t_idx, g_idx:g_idx + 1].clone()
    b.hand_p1 = snaps["hand_p1"][t_idx, g_idx:g_idx + 1].clone()
    b.score_p0 = snaps["score_p0"][t_idx, g_idx:g_idx + 1].clone()
    b.score_p1 = snaps["score_p1"][t_idx, g_idx:g_idx + 1].clone()
    b.current_player = snaps["current_player"][t_idx, g_idx:g_idx + 1].clone()
    return b


def label_states(snaps, n_games, K, device="cpu", log_every=25):
    obs_list, act_list, ret_list = [], [], []
    n_total = len(SAMPLE_TS) * n_games
    done = 0
    t0 = time.time()
    for t_idx in SAMPLE_TS:
        for g in range(n_games):
            env1 = _build_single_env(snaps, t_idx, g, device)
            obs = env1.observe()
            action, value = lookahead_action(env1, K=K, return_value=True)
            done += 1
            if value is None:
                continue  # no legal action at this state (shouldn't happen for t<37)
            obs_list.append(obs[0].cpu())
            act_list.append(int(action[0]))
            ret_list.append(value)
            if done % log_every == 0:
                dt = time.time() - t0
                print(f"  {done}/{n_total} states labeled ({len(obs_list)} kept) "
                      f"-- {dt:.0f}s elapsed, {dt/done:.2f}s/state, "
                      f"~{dt/done*(n_total-done):.0f}s remaining", flush=True)
    return (torch.stack(obs_list), torch.tensor(act_list, dtype=torch.int64),
            torch.tensor(ret_list, dtype=torch.float32))


def main(args):
    device = args.device
    pairings = [p.split(":") for p in args.mix.split(",")]
    games_per_pairing = max(1, args.n_games // len(pairings))

    all_obs, all_act, all_ret = [], [], []
    seed = args.seed_base
    for seat0, seat1 in pairings:
        print(f"=== source games: {seat0} vs {seat1}, n={games_per_pairing}, seed_base={seed} ===",
              flush=True)
        snaps = play_and_snapshot(seat0, seat1, games_per_pairing, seed, device=device)
        seed += games_per_pairing * 1000  # keep seed ranges from colliding across calls
        print(f"labeling {len(SAMPLE_TS)} sampled states x {games_per_pairing} games "
              f"via lookahead_bot(K={args.k})...", flush=True)
        obs, act, ret = label_states(snaps, games_per_pairing, args.k, device=device)
        all_obs.append(obs); all_act.append(act); all_ret.append(ret)

    new_obs = torch.cat(all_obs)
    new_act = torch.cat(all_act)
    new_ret = torch.cat(all_ret)

    if os.path.exists(args.out):
        old = torch.load(args.out)
        new_obs = torch.cat([old["obs"], new_obs])
        new_act = torch.cat([old["act"], new_act])
        new_ret = torch.cat([old["ret"], new_ret])
        print(f"appended to existing dataset ({old['obs'].shape[0]} -> {new_obs.shape[0]} examples)")
    torch.save({"obs": new_obs, "act": new_act, "ret": new_ret}, args.out)
    print(f"saved {new_obs.shape[0]} total examples to {args.out}")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=60,
                    help="source games PER pairing in --mix")
    ap.add_argument("--mix", type=str, default="greedy:greedy,greedy:random,random:random",
                    help="comma-separated seat0:seat1 agent pairings for source games")
    ap.add_argument("--k", type=int, default=8, help="lookahead_bot's K")
    ap.add_argument("--seed-base", type=int, default=30_000_000)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="training/lookahead_dataset.pt")
    return ap


if __name__ == "__main__":
    main(build_argparser().parse_args())
