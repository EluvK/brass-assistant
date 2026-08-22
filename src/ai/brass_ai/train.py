"""AlphaZero-style training loop with a persistent optimizer.

Loss (per sample):
  L = -sum_a p_a * log_softmax(score(s, a))_a     (policy CE over concrete
      Engine-generated legal candidates; padding is masked only for batching)
    + ||value_4 - z_4||^2                         (MSE on the 4-player
      normalized final-VP vector, no tanh on the value head)
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
from .hierarchical_policy import (
    ACTION_FEATURE_DIM,
    ACTION_FEATURE_SCHEMA_VERSION,
    pad_candidate_features,
)
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
    econ_lambda: float = 0.2   # weight of the economic-supervision auxiliary loss
    econ_neg_weight: float = 3.0  # extra weight on samples with negative income
    # Bound the largest padded candidate matrix in one GPU batch. Full-legal
    # states can have hundreds of candidates, so a fixed sample batch is unsafe.
    max_candidate_batch: int = 65536


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
            losses.extend(self.train_one_epoch(samples))
            self.scheduler.step()
            self.epoch_count += 1
            prog.update(e + 1)
        stats = _mean_losses(losses)
        stats.update(evaluate_policy(self.net, samples, self.cfg.device))
        return stats

    def train_one_epoch(self, samples: list[Sample], progress_label: str = "train") -> list[dict]:
        """Train one epoch without advancing the LR scheduler.

        This is useful when a dataset is streamed from multiple disk shards:
        the caller can process every shard once, then advance the scheduler
        exactly once for the complete logical epoch.
        """
        if not samples:
            return []
        idx = np.random.permutation(len(samples))
        losses = []
        prog = Progress(len(samples), progress_label, every_s=2.0)
        completed = 0
        for start in range(0, len(samples), self.cfg.batch_size):
            raw = [samples[i] for i in idx[start:start + self.cfg.batch_size]]
            # Padding is determined by the largest candidate row in the batch.
            # Split large-candidate chunks so activation memory scales with the
            # candidate budget rather than the nominal sample batch size.
            raw.sort(key=lambda s: len(s.candidates))
            chunk_start = 0
            while chunk_start < len(raw):
                max_n = max(len(s.candidates) for s in raw[chunk_start:])
                cap = max(1, self.cfg.max_candidate_batch // max_n)
                chunk = raw[chunk_start:chunk_start + min(self.cfg.batch_size, cap)]
                losses.append(train_on_batch(self.net, _to_batch(chunk), self.cfg, self.optimizer))
                chunk_start += len(chunk)
                completed += len(chunk)
                prog.update(completed)
        prog.done()
        return losses

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
            "action_feature_dim": ACTION_FEATURE_DIM,
            "action_feature_schema_version": ACTION_FEATURE_SCHEMA_VERSION,
        }

    def load_state_dict(self, sd: dict) -> None:
        schema_version = sd.get("action_feature_schema_version")
        feature_dim = sd.get("action_feature_dim")
        if schema_version != ACTION_FEATURE_SCHEMA_VERSION or feature_dim != ACTION_FEATURE_DIM:
            raise ValueError(
                "incompatible checkpoint action-feature schema: "
                f"got version={schema_version}, dim={feature_dim}; expected "
                f"version={ACTION_FEATURE_SCHEMA_VERSION}, dim={ACTION_FEATURE_DIM}"
            )
        self.net.load_state_dict(sd["model"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self.scheduler.load_state_dict(sd["scheduler"])
        self.epoch_count = sd.get("epoch", 0)


def compute_loss(batch: dict, net: PolicyValueNet, l2: float, device: str,
                 econ_lambda: float = 0.2, econ_neg_weight: float = 3.0):
    tensors = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
    out = net(tensors, tensors["candidates"], tensors["candidate_mask"])
    target = tensors["policy"]
    policy_loss = -(target * out["candidate_log_probs"].masked_fill(
        ~out["candidate_mask"], 0.0
    )).sum(dim=1).mean()
    value_loss = F.mse_loss(out["value"], tensors["value"])

    # This auxiliary objective is the action-type marginal implied by the
    # candidate policy target. It guides coarse action selection without ever
    # reintroducing a flat slot head.
    type_target = torch.einsum("bn,bnt->bt", target, tensors["candidates"][..., :7])
    type_loss = -(type_target * F.log_softmax(out["type_logits"], dim=1)).sum(dim=1).mean()

    # Economic-supervision auxiliary loss. Targets are raw (income_level -10..30,
    # money); normalize income to ~0..1 and clamp money so the MSE is scale-
    # comparable with the VP loss. Negative-income samples get extra weight
    # (the "negative economy is a real problem" prior); no maximization target
    # is imposed — the net just learns to predict the real economy, so it
    # spontaneously avoids income collapse without chasing income maximization.
    econ_target = tensors["econ"]  # (B,2): (income_level, money)
    inc_t = ((econ_target[:, 0] + 10.0) / 40.0).clamp(0.0, 1.0)
    money_t = (econ_target[:, 1] / 100.0).clamp(0.0, 1.0)
    econ = out["econ"]
    inc_pred = (econ[:, 0] + 10.0) / 40.0
    money_pred = econ[:, 1] / 100.0
    inc_loss = F.mse_loss(inc_pred, inc_t)
    money_loss = F.mse_loss(money_pred, money_t)
    # Extra weight on samples whose target income is negative.
    neg_mask = (econ_target[:, 0] < 0).float()
    inc_loss_neg = F.mse_loss(inc_pred, inc_t, reduction="none")
    w = 1.0 + (econ_neg_weight - 1.0) * neg_mask
    inc_loss = (inc_loss_neg * w).mean()
    econ_loss = econ_lambda * (inc_loss + money_loss)

    l2_loss = sum(p.pow(2).sum() for p in net.parameters()) * l2
    return policy_loss + 0.1 * type_loss, value_loss, econ_loss, l2_loss


def train_on_batch(net, batch, cfg: TrainConfig, optimizer) -> dict:
    net.train()
    optimizer.zero_grad(set_to_none=True)
    if cfg.amp and cfg.device != "cpu":
        with torch.autocast(device_type=cfg.device):
            pl, vl, el, ll = compute_loss(
                batch, net, cfg.l2, cfg.device, cfg.econ_lambda, cfg.econ_neg_weight)
        total = pl + vl + el + ll
        total.backward()
    else:
        pl, vl, el, ll = compute_loss(
            batch, net, cfg.l2, cfg.device, cfg.econ_lambda, cfg.econ_neg_weight)
        (pl + vl + el + ll).backward()
    optimizer.step()
    return {
        "policy": pl.detach().item(),
        "value": vl.detach().item(),
        "econ": el.detach().item(),
        "l2": ll.detach().item(),
    }


def _to_batch(samples: list[Sample]) -> dict:
    b = np.stack([s.board for s in samples]).astype(np.float32)
    l = np.stack([s.links for s in samples]).astype(np.float32)
    g = np.stack([s.global_vec for s in samples]).astype(np.float32)
    o = np.stack([s.own_hand for s in samples]).astype(np.float32)
    p = np.stack([s.opp_hands for s in samples]).astype(np.float32)
    candidate_rows = []
    for sample in samples:
        row = sample.candidates
        if row.dtype == np.uint8:
            row = row.astype(np.float32) / 4.0
        candidate_rows.append(torch.from_numpy(row))
    candidates, candidate_mask = pad_candidate_features(candidate_rows)
    pol = np.zeros(candidate_mask.shape, dtype=np.float32)
    for i, sample in enumerate(samples):
        pol[i, :len(sample.policy)] = sample.policy
    val = np.stack([s.value for s in samples]).astype(np.float32)
    econ = np.stack([s.econ for s in samples]).astype(np.float32)
    era = np.asarray([s.era for s in samples], dtype=np.int64)
    return {
        "board": b, "links": l, "global": g,
        "own_hand": o, "opp_hands": p, "policy": pol, "value": val,
        "candidates": candidates.numpy(), "candidate_mask": candidate_mask.numpy(),
        "econ": econ, "era": era,
    }


def _mean_losses(losses):
    if not losses:
        return {}
    out = {}
    for k in losses[0]:
        out[k] = sum(x[k] for x in losses) / len(losses)
    return out


def evaluate_policy(net: PolicyValueNet, samples: list[Sample], device: str = "cpu",
                    batch_size: int = 256, max_candidate_batch: int = 16384) -> dict:
    """Measure candidate-policy quality on teacher targets.

    Metrics are candidate-level and remain meaningful when every state has a
    different action count. The model is evaluated without changing its
    training/eval state.
    """
    if not samples:
        return {}
    was_training = net.training
    net.eval()
    try:
        totals = {"policy_top1": 0.0, "policy_top3": 0.0, "policy_top5": 0.0,
                  "policy_entropy": 0.0, "action_type_top1": 0.0}
        candidate_counts = []
        seen = 0
        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                raw = samples[start:start + batch_size]
                raw.sort(key=lambda s: len(s.candidates))
                for sub_start in range(0, len(raw), batch_size):
                    sub = raw[sub_start:sub_start + batch_size]
                    max_n = max(len(s.candidates) for s in sub)
                    cap = max(1, max_candidate_batch // max_n)
                    for offset in range(0, len(sub), cap):
                        batch = _to_batch(sub[offset:offset + cap])
                        tensors = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
                        out = net(tensors, tensors["candidates"], tensors["candidate_mask"])
                        mask = tensors["candidate_mask"]
                        target = tensors["policy"]
                        n = target.shape[0]
                        ranking = out["candidate_logits"].masked_fill(~mask, float("-inf")).argsort(
                            dim=1, descending=True
                        )
                        target_idx = target.argmax(dim=1)
                        totals["policy_top1"] += (ranking[:, :1] == target_idx[:, None]).any(dim=1).sum().item()
                        totals["policy_top3"] += (ranking[:, :3] == target_idx[:, None]).any(dim=1).sum().item()
                        totals["policy_top5"] += (ranking[:, :5] == target_idx[:, None]).any(dim=1).sum().item()
                        log_probs = out["candidate_log_probs"].masked_fill(~mask, 0.0)
                        totals["policy_entropy"] += (-(log_probs * log_probs.exp()).sum(dim=1)).sum().item()
                        type_target = torch.einsum("bn,bnt->bt", target, tensors["candidates"][..., :7])
                        totals["action_type_top1"] += (out["type_logits"].argmax(dim=1) == type_target.argmax(dim=1)).sum().item()
                        candidate_counts.extend(mask.sum(dim=1).detach().cpu().tolist())
                        seen += n
        metrics = {key: value / seen for key, value in totals.items()}
        metrics["candidate_count_mean"] = float(np.mean(candidate_counts))
        metrics["candidate_count_p95"] = float(np.percentile(candidate_counts, 95))
        return metrics
    finally:
        net.train(was_training)


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
    from .rust_mcts import RustISMCTS, RustMCTSConfig

    mcts = RustISMCTS(net, RustMCTSConfig(c_puct=2.5, max_depth=10))
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
