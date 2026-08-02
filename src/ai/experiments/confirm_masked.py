"""Confirmation eval for imitation_masked.pt (larger sample to rule out luck).

Run from repo root:
    PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe src/ai/experiments/confirm_masked.py
"""
import torch

from brass_ai.evaluate import evaluate_mcts_vs_baseline
from brass_ai.mcts import ISMCTS, MCTSConfig
from brass_ai.net import PolicyValueNet
from experiments.exp_masked_loss import eval_greedy

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

net = PolicyValueNet()
net.load_state_dict(torch.load("checkpoints/imitation_masked.pt", map_location=device))
net.eval()

wg, mg, hg = eval_greedy(net, n_games=10)
print(f"greedy policy vs heuristic (10 games): win={wg:.0%} vp={mg:.1f}/{hg:.1f}")

mcts = ISMCTS(net, MCTSConfig(c_puct=1.5, max_depth=8), device=device)
for baseline in ("heuristic", "2ply"):
    wr, mvp, bvp = evaluate_mcts_vs_baseline(mcts, n_games=8, sims=60, baseline=baseline)
    print(f"MCTS vs {baseline} (8 games): win={wr:.0%} vp={mvp:.1f}/{bvp:.1f}")
