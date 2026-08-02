"""AlphaZero-style training loop.

Loss (per sample):
  L = -sum_t p_t * log_softmax(logits)_t   (policy cross-entropy over 1316 slots)
    + (value - z)^2                        (MSE on normalized final VP)
    + l2 * ||theta||^2

Iteration: self-play with the current net -> train on the generated samples ->
(repeat). Kept intentionally small for CPU; the same code runs on CUDA by
passing `device="cuda"` (mixed-precision via torch.autocast).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .net import PolicyValueNet
from .selfplay import SelfPlayConfig, Sample, play_batch


@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 5          # passes over the newest batch of samples
    batch_size: int = 256
    lr: float = 1e-3
    l2: float = 1e-4
    amp: bool = True         # fp16 autocast (no-op on CPU)


def compute_loss(batch: dict, net: PolicyValueNet, l2: float, device: str):
    tensors = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
    logits, value = net(tensors)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(tensors["policy"] * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(value, tensors["value"])
    l2_loss = sum(p.pow(2).sum() for p in net.parameters()) * l2
    return policy_loss, value_loss, l2_loss


def train_on_batch(net, batch, cfg: TrainConfig, optimizer) -> dict:
    net.train()
    optimizer.zero_grad(set_to_none=True)
    if cfg.amp and cfg.device != "cpu":
        with torch.autocast(device_type=cfg.device):
            pl, vl, ll = compute_loss(batch, net, cfg.l2, cfg.device)
        total = pl + vl + ll
        total.backward()
    else:
        pl, vl, ll = compute_loss(batch, net, cfg.l2, cfg.device)
        (pl + vl + ll).backward()
    optimizer.step()
    return {
        "policy": pl.detach().item(),
        "value": vl.detach().item(),
        "l2": ll.detach().item(),
    }


def train_on_samples(net: PolicyValueNet, samples: list[Sample], cfg: TrainConfig):
    """One optimization pass (cfg.epochs) over the given samples."""
    if not samples:
        return {}
    net.to(cfg.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    n = len(samples)
    losses = []
    for epoch in range(cfg.epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, cfg.batch_size):
            chunk = [samples[i] for i in idx[start:start + cfg.batch_size]]
            batch = _to_batch(chunk)
            losses.append(train_on_batch(net, batch, cfg, optimizer))
    return _mean_losses(losses)


def _to_batch(samples: list[Sample]) -> dict:
    b = np.stack([s.board for s in samples]).astype(np.float32)
    l = np.stack([s.links for s in samples]).astype(np.float32)
    g = np.stack([s.global_vec for s in samples]).astype(np.float32)
    o = np.stack([s.own_hand for s in samples]).astype(np.float32)
    p = np.stack([s.opp_hands for s in samples]).astype(np.float32)
    pol = np.stack([s.policy for s in samples]).astype(np.float32)
    val = np.asarray([s.value for s in samples], dtype=np.float32)
    return {
        "board": b, "links": l, "global": g,
        "own_hand": o, "opp_hands": p, "policy": pol, "value": val,
    }


def _mean_losses(losses):
    if not losses:
        return {}
    out = {}
    for k in losses[0]:
        out[k] = sum(x[k] for x in losses) / len(losses)
    return out


@dataclass
class LoopConfig:
    iterations: int = 10
    games_per_iter: int = 8
    selfplay: SelfPlayConfig = None  # type: ignore[assignment]
    train: TrainConfig = None  # type: ignore[assignment]
    mcts_sims: int = 80


def run_loop(net: PolicyValueNet, loop: LoopConfig, on_iters=None):
    """Self-play -> train -> repeat. `on_iters(iter, net, stats)` is a callback
    (e.g. for evaluation/logging); return it or None."""
    from .mcts import ISMCTS, MCTSConfig

    mcts = ISMCTS(net, MCTSConfig(c_puct=1.5, max_depth=10))
    sp_cfg = loop.selfplay or SelfPlayConfig(sims=loop.mcts_sims)

    for it in range(loop.iterations):
        t0 = time.time()
        samples, avg_vps, _ = play_batch(mcts, loop.games_per_iter, sp_cfg)
        sp_time = time.time() - t0
        t1 = time.time()
        losses = train_on_samples(net, samples, loop.train or TrainConfig())
        tr_time = time.time() - t1
        stats = {
            "iter": it,
            "samples": len(samples),
            "avg_vps": avg_vps,
            "loss": losses,
            "sp_sec": sp_time,
            "train_sec": tr_time,
        }
        if on_iters:
            stop = on_iters(it, net, stats)
            if stop:
                break
    return net
