"""Measure strength vs cost across (K, depth) for the tunable search bot, so
the difficulty presets in bots.py are grounded in real numbers rather than
guesses -- and so the Godot game knows which configs are fast enough to be
playable (HANDOFF.md 8.10 #6).

Every config plays vs plain greedy, seat rotated each game to cancel
seat-order effects (per HANDOFF.md 7.4's noise caution). Reports win rate,
mean margin, and seconds-per-move.
"""
import argparse
import time

import torch

from hex_truchet.fast import HexTruchetBatched
from train_selfplay import greedy_action, HORIZON
from lookahead_bot import lookahead_action


@torch.no_grad()
def play_game(K, depth, seat, seed, device="cpu"):
    env = HexTruchetBatched(1, device=device)
    env.reset(torch.tensor([seed], dtype=torch.int64, device=device))
    obs = env.observe()
    n_bot_moves = 0
    bot_time = 0.0
    while True:
        if int(env.current_player[0]) == seat:
            t0 = time.time()
            a = lookahead_action(env, K=K, depth=depth) if K > 1 else greedy_action(env, obs)
            bot_time += time.time() - t0
            n_bot_moves += 1
        else:
            a = greedy_action(env, obs)
        obs, rew, term, info = env.step(a)
        if bool(term.all()):
            fs = info["final_state_json"][0]
            break
    bot_s = fs["score_p0"] if seat == 0 else fs["score_p1"]
    opp_s = fs["score_p1"] if seat == 0 else fs["score_p0"]
    return bot_s, opp_s, bot_time / max(1, n_bot_moves)


def evaluate(K, depth, n_games, device="cpu", seed_base=40_000_000):
    wins = draws = 0
    margins, per_move = [], []
    for g in range(n_games):
        b, o, spm = play_game(K, depth, g % 2, seed_base + g, device)
        margins.append(b - o)
        per_move.append(spm)
        if b > o:
            wins += 1
        elif b == o:
            draws += 1
    n = n_games
    return {"win": wins / n, "draw": draws / n,
            "margin": sum(margins) / n, "s_per_move": sum(per_move) / n}


def main(args):
    configs = []
    for spec in args.configs.split(","):
        K, depth = spec.split(":")
        configs.append((int(K), int(depth)))

    print(f"vs greedy, n={args.n_games} games each, seat rotated\n")
    print(f"{'K':>3} {'depth':>6} {'win':>7} {'draw':>6} {'margin':>8} {'s/move':>8}")
    print("-" * 44)
    for K, depth in configs:
        r = evaluate(K, depth, args.n_games, args.device)
        d = "end" if depth == 0 else str(depth)
        print(f"{K:>3} {d:>6} {r['win']:>7.3f} {r['draw']:>6.3f} "
              f"{r['margin']:>+8.2f} {r['s_per_move']:>8.3f}", flush=True)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str,
                    default="1:0,3:4,3:6,3:0,5:6,5:12,8:8,8:0",
                    help="comma-separated K:depth pairs (depth 0 = roll to game end)")
    ap.add_argument("--n-games", type=int, default=20)
    ap.add_argument("--device", type=str, default="cpu")
    return ap


if __name__ == "__main__":
    main(build_argparser().parse_args())
