"""AlphaZero-style training loop with a persistent optimizer.

Loss (per sample):
  L = -sum_t p_t * log_softmax_masked(logits)_t   (policy CE over LEGAL slots;
      the ~700 always-illegal double-rail slots are masked out of the softmax)
    + (value - z)^2                               (MSE on normalized final VP)
    + l2 * ||theta||^2

The `Trainer` class owns the network, a persistent AdamW optimizer and a
CosineAnnealingLR scheduler, so momentum / LR state survives across self-play
iterations (previously each call rebuilt Adam and lost all state).

Iteration: self-play with the current net -> trainer.train_on_samples(...) ->
(repeat). CPU by default; `device="cuda"` uses mixed-precision autocast.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .net import PolicyValueNet
from .progress import Progress
from .selfplay import SelfPlayConfig, Sample, play_batch


@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 5          # passes over the newest batch of samples per call
    batch_size: int = 256
    lr: float = 1e-3
    l2: float = 1e-4         # explicit L2 on all parameters
    weight_decay: float = 0.0  # AdamW decoupled decay (separate from l2)
    t_max: int = 100         # CosineAnnealingLR period, in total epochs
    min_lr: float = 1e-5
    amp: bool = True         # fp16 autocast (no-op on CPU)


class Trainer:
    """Persistent optimizer + LR scheduler around a PolicyValueNet."""

    def __init__(self, net: PolicyValueNet, cfg: TrainConfig | None = None):
        self.cfg = cfg or TrainConfig()
        self.net = net
        self.net.to(self.cfg.device)
        self.optimizer = torch.optim.AdamW(
            net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.cfg.t_max, 1), eta_min=self.cfg.min_lr
        )
        self.epoch_count = 0

    # ------------------------------------------------------------ training
    def train_on_samples(self, samples: list[Sample]) -> dict:
        """One pass (cfg.epochs) over the given samples, then one LR step."""
        if not samples:
            return {}
        n = len(samples)
        losses = []
        prog = Progress(self.cfg.epochs, "train", every_s=10.0)
        for e in range(self.cfg.epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.cfg.batch_size):
                chunk = [samples[i] for i in idx[start:start + self.cfg.batch_size]]
                batch = _to_batch(chunk)
                losses.append(train_on_batch(self.net, batch, self.cfg, self.optimizer))
            self.scheduler.step()
            self.epoch_count += 1
            prog.update(e + 1)
        return _mean_losses(losses)

    def current_lr(self) -> float:
        return self.scheduler.get_last_lr()[0]

    # ------------------------------------------------------------ replay path
    def train_steps(self, samples: list[Sample], n_steps: int, batch_size: int | None = None) -> dict:
        """`n_steps` gradient steps, each on a random minibatch drawn WITH
        replacement from `samples` — supports a growing replay buffer that
        reuses self-play data across iterations. Does NOT step the LR scheduler
        (call `step_lr` once per iteration so the cosine schedule tracks the
        iteration count, not the step count)."""
        if not samples:
            return {}
        bs = batch_size or self.cfg.batch_size
        losses = []
        n = len(samples)
        prog = Progress(n_steps, "train", every_s=10.0)
        for s in range(n_steps):
            idx = np.random.randint(0, n, size=bs)
            chunk = [samples[i] for i in idx]
            batch = _to_batch(chunk)
            losses.append(train_on_batch(self.net, batch, self.cfg, self.optimizer))
            prog.update(s + 1)
        return _mean_losses(losses)

    def step_lr(self) -> None:
        self.scheduler.step()
        self.epoch_count += 1

    # ------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return {
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch_count,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.net.load_state_dict(sd["model"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self.scheduler.load_state_dict(sd["scheduler"])
        self.epoch_count = sd.get("epoch", 0)


def compute_loss(batch: dict, net: PolicyValueNet, l2: float, device: str):
    tensors = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
    logits, value = net(tensors)
    # Masked policy loss: normalize ONLY over the legal slots. Without this,
    # the ~700 always-illegal double-rail slots pollute the softmax denominator
    # and the net wastes its capacity suppressing them (a large chunk of the
    # initial probability mass sits on phantom slots).
    target = tensors["policy"]
    mask = tensors["legal"].to(torch.bool)
    # Normalize over legal slots only (phantom slots get -inf -> excluded from
    # the softmax denominator), then zero the illegal log-probs so that
    # `target * log_probs` never hits 0 * -inf = NaN.
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=1)
    log_probs = log_probs.masked_fill(~mask, 0.0)
    policy_loss = -(target * log_probs).sum(dim=1).mean()
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


def _to_batch(samples: list[Sample]) -> dict:
    b = np.stack([s.board for s in samples]).astype(np.float32)
    l = np.stack([s.links for s in samples]).astype(np.float32)
    g = np.stack([s.global_vec for s in samples]).astype(np.float32)
    o = np.stack([s.own_hand for s in samples]).astype(np.float32)
    p = np.stack([s.opp_hands for s in samples]).astype(np.float32)
    pol = np.stack([s.policy for s in samples]).astype(np.float32)
    val = np.asarray([s.value for s in samples], dtype=np.float32)
    legal = np.stack([s.legal for s in samples]).astype(np.bool_)
    return {
        "board": b, "links": l, "global": g,
        "own_hand": o, "opp_hands": p, "policy": pol, "value": val, "legal": legal,
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
    """Self-play -> train -> repeat with a persistent Trainer.

    `on_iters(iter, trainer, stats)` is a callback (e.g. evaluation/logging);
    return True from it to stop early."""
    from .mcts import ISMCTS, MCTSConfig

    mcts = ISMCTS(net, MCTSConfig(c_puct=1.5, max_depth=10))
    sp_cfg = loop.selfplay or SelfPlayConfig(sims=loop.mcts_sims)
    trainer = Trainer(net, loop.train or TrainConfig())

    for it in range(loop.iterations):
        t0 = time.time()
        samples, avg_vps, _ = play_batch(mcts, loop.games_per_iter, sp_cfg)
        sp_time = time.time() - t0
        t1 = time.time()
        losses = trainer.train_on_samples(samples)
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
            stop = on_iters(it, trainer, stats)
            if stop:
                break
    return net
