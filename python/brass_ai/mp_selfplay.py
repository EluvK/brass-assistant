"""Multi-process self-play worker pool.

Workers (torch.multiprocessing spawn mode) each run their own Python MCTS over
the Rust engine and push packed **numpy** sample arrays back through a Queue.
Only lightweight numpy arrays cross the process boundary — never Rust
`GameState` handles, torch tensors or complex objects. The main process
rebuilds `Sample` objects and trains.

Matchmaking: with `mm_prob > 0`, each game has one rotating "learner" seat
(current net, whose samples are collected) and opponent seats drawn from a pool
of historical nets (plus the current net). This anchors training against a
stable reference and prevents the drift/collapse seen in pure self-play.

Workers default to CPU inference: the network is tiny and per-sim cost is
dominated by Python/Rust bookkeeping, so many workers do not contend for the
GPU, which is reserved for the main process's training.

Windows notes: spawn re-imports the target module in each child; the child
inherits sys.path, so `brass_ai._engine` (PyO3 .pyd) must be importable there.
Each worker pins `torch.set_num_threads(1)` so 8-16 processes really use the
16 CPU cores.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue

import numpy as np
import torch

from . import _engine as be
from .progress import Progress
from .selfplay import Sample, SelfPlayConfig, play_game_with_roles
from .hierarchical_policy import ACTION_FEATURE_DIM, pad_candidate_features

_PACK_TIMEOUT_S = 1800  # per packet; a full game at sims=200 can take minutes


def _worker_fn(worker_id, cmd_queue, result_queue, device, seed_base):
    # Imports happen inside the child (spawn re-imports everything anyway).
    from . import _engine as be  # noqa: F401  (ensure the extension loads here)
    from .net import PolicyValueNet
    from .rust_mcts import RustISMCTS, RustMCTSConfig

    torch.set_num_threads(1)  # one core per worker
    while True:
        cmd = cmd_queue.get()
        if cmd is None:
            break  # shutdown
        weights, pool_weights, games, sims, seed_offset, mcts_cfg, temperature, mm_prob = cmd
        net = PolicyValueNet()
        net.load_state_dict(weights)
        net.eval()
        mcts = RustISMCTS(net, RustMCTSConfig(**mcts_cfg, device=device))

        pool = []
        for pw in pool_weights:
            pn = PolicyValueNet()
            pn.load_state_dict(pw)
            pn.eval()
            pool.append(RustISMCTS(pn, RustMCTSConfig(**mcts_cfg, device=device)))

        cfg = SelfPlayConfig(players=4, sims=sims, temperature=temperature, max_moves=600)
        for gi in range(games):
            try:
                cfg.seed = seed_base + worker_id * 100_000 + seed_offset + gi
                if mm_prob > 0.0 and pool:
                    # Rotating learner seat; opponents = historical pool (with
                    # probability mm_prob) else the current net.
                    learner = gi % 4
                    roles = [mcts.search] * 4
                    for seat in range(4):
                        if seat == learner:
                            continue
                        if np.random.rand() < mm_prob:
                            opp = pool[np.random.randint(len(pool))]
                            roles[seat] = opp.search
                    samples, _ = play_game_with_roles(roles, cfg, collect={learner})
                else:
                    samples, _ = play_game_with_roles([mcts.search] * 4, cfg)
                result_queue.put(("SAMPLES", _pack_samples(samples)))
            except Exception as exc:
                result_queue.put(("ERROR", f"worker={worker_id} game={gi}: {exc!r}"))
                break
        result_queue.put(("DONE", worker_id))


def _pack_samples(samples: list[Sample]) -> dict:
    """Pack samples into numpy arrays only (lightweight, picklable)."""
    n = len(samples)
    if n == 0:
        return {
            "count": 0,
            "pid": np.empty(0, dtype=np.int64),
            "era": np.empty(0, dtype=np.int64),
            "board": np.empty((0, be.BOARD_PLANES, be.BOARD_CELLS), dtype=np.float32),
            "links": np.empty((0, be.LINK_PLANES, be.LINK_CELLS), dtype=np.float32),
            "global": np.empty((0, be.GLOBAL_LEN), dtype=np.float32),
            "own_hand": np.empty((0, be.HAND_LEN), dtype=np.float32),
            "opp_hands": np.empty((0, be.HAND_LEN * 3), dtype=np.float32),
            "candidates": np.empty((0, 0, ACTION_FEATURE_DIM), dtype=np.float32),
            "candidate_mask": np.empty((0, 0), dtype=np.bool_),
            "policy": np.empty((0, 0), dtype=np.float32),
            "value": np.empty((0, 4), dtype=np.float32),
            "econ": np.empty((0, 2), dtype=np.float32),
        }
    candidates, candidate_mask = pad_candidate_features(
        [torch.from_numpy(s.candidates) for s in samples]
    )
    policy = np.zeros(candidate_mask.shape, dtype=np.float32)
    for i, sample in enumerate(samples):
        policy[i, :len(sample.policy)] = sample.policy
    return {
        "pid": np.asarray([s.pid for s in samples], dtype=np.int64),
        "era": np.asarray([s.era for s in samples], dtype=np.int64),
        "board": np.stack([s.board for s in samples]).astype(np.float32),
        "links": np.stack([s.links for s in samples]).astype(np.float32),
        "global": np.stack([s.global_vec for s in samples]).astype(np.float32),
        "own_hand": np.stack([s.own_hand for s in samples]).astype(np.float32),
        "opp_hands": np.stack([s.opp_hands for s in samples]).astype(np.float32),
        "candidates": candidates.numpy(),
        "candidate_mask": candidate_mask.numpy(),
        "policy": policy,
        "value": np.stack([s.value for s in samples]).astype(np.float32),
        "econ": np.stack([s.econ for s in samples]).astype(np.float32),
        "count": n,
    }


def unpack_samples(packed: dict) -> list[Sample]:
    n = packed["count"]
    out = []
    for i in range(n):
        out.append(
            Sample(
                pid=int(packed["pid"][i]),
                era=int(packed["era"][i]),
                board=packed["board"][i],
                links=packed["links"][i],
                global_vec=packed["global"][i],
                own_hand=packed["own_hand"][i],
                opp_hands=packed["opp_hands"][i],
                candidates=packed["candidates"][i, packed["candidate_mask"][i]],
                policy=packed["policy"][i, packed["candidate_mask"][i]],
                value=packed["value"][i].astype(np.float32),
                econ=packed["econ"][i].astype(np.float32),
            )
        )
    return out


class SelfPlayPool:
    """A persistent pool of self-play workers (spawned once, reused)."""

    def __init__(self, n_workers: int = 8, device: str = "cpu"):
        self.n_workers = n_workers
        self.device = device
        self.cmd_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.processes = []
        for wid in range(n_workers):
            p = mp.Process(
                target=_worker_fn,
                args=(wid, self.cmd_queue, self.result_queue, device, 0),
                daemon=True,
            )
            p.start()
            self.processes.append(p)

    def generate(
        self,
        net,
        games_per_worker: int,
        sims: int,
        seed: int = 0,
        mcts_cfg: dict | None = None,
        temperature: float = 1.0,
        mm_pool: list | None = None,
        mm_prob: float = 0.0,
        verbose: bool = True,
    ):
        """Broadcast the current weights (plus matchmaking pool) and collect
        samples from all workers.

        `mm_pool` is a list of state-dicts of historical nets used as opponent
        seats with probability `mm_prob` per game (learner seat = current net).
        Returns (samples, per_worker_sample_counts)."""
        weights = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        pool_weights = []  # empty -> workers build no opponent pool (pure self-play)
        cfg = mcts_cfg or {}
        if mm_pool:
            pool_weights = [{k: v.detach().cpu() for k, v in pw.items()} for pw in mm_pool]
        for _ in range(self.n_workers):
            self.cmd_queue.put(
                (weights, pool_weights, games_per_worker, sims, seed, cfg, temperature, mm_prob)
            )

        # Packets and DONE markers arrive interleaved (workers finish at
        # different times); count DONEs until every worker has reported.
        total_games = self.n_workers * games_per_worker
        prog = Progress(total_games, f"selfplay w={self.n_workers} sims={sims}")
        samples = []
        counts = []
        done = 0
        games_received = 0
        while done < self.n_workers:
            item = self._get_with_timeout()
            tag, payload = item
            if tag == "DONE":
                done += 1
            elif tag == "ERROR":
                raise RuntimeError(f"self-play worker failed: {payload}")
            else:  # "SAMPLES"
                samples.extend(unpack_samples(payload))
                counts.append(payload["count"])
                games_received += 1
                if verbose:
                    prog.update(games_received)
        if verbose:
            prog.done()
        return samples, counts

    def _get_with_timeout(self):
        try:
            return self.result_queue.get(timeout=_PACK_TIMEOUT_S)
        except _queue.Empty as e:
            raise TimeoutError(
                f"self-play worker timed out after {_PACK_TIMEOUT_S}s"
            ) from e

    def close(self):
        for _ in range(self.n_workers):
            self.cmd_queue.put(None)
        for p in self.processes:
            p.join(timeout=30)
        self.processes.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
