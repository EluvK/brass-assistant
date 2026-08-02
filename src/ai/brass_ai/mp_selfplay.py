"""Multi-process self-play worker pool.

Workers (torch.multiprocessing spawn mode) each run their own Python MCTS over
the Rust engine and push packed **numpy** sample arrays back through a Queue.
Only lightweight numpy arrays cross the process boundary — never Rust
`GameState` handles, torch tensors or complex objects. The main process
rebuilds `Sample` objects and trains.

Workers default to CPU inference: the network is tiny and per-sim cost is
dominated by Python/Rust bookkeeping, so many workers do not contend for the
GPU, which is reserved for the main process's training.

Windows notes: spawn re-imports the target module in each child; the child
inherits sys.path, so `brass_engine` (PyO3 .pyd) must be importable there.
Each worker pins `torch.set_num_threads(1)` so 8-16 processes really use the
16 CPU cores.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue

import numpy as np
import torch

from .progress import Progress
from .selfplay import Sample, SelfPlayConfig, play_game

_PACK_TIMEOUT_S = 1800  # per packet; a full game at sims=200 can take minutes


def _worker_fn(worker_id, cmd_queue, result_queue, device, seed_base):
    # Imports happen inside the child (spawn re-imports everything anyway).
    import brass_engine as be  # noqa: F401  (ensure the extension loads here)
    from .net import PolicyValueNet
    from .mcts import ISMCTS, MCTSConfig

    torch.set_num_threads(1)  # one core per worker
    while True:
        cmd = cmd_queue.get()
        if cmd is None:
            break  # shutdown
        weights, games, sims, seed_offset, mcts_cfg, temperature = cmd
        net = PolicyValueNet()
        net.load_state_dict(weights)
        net.eval()
        mcts = ISMCTS(net, MCTSConfig(**mcts_cfg), device=device)
        cfg = SelfPlayConfig(players=4, sims=sims, temperature=temperature, max_moves=600)
        for gi in range(games):
            cfg.seed = seed_base + worker_id * 100_000 + seed_offset + gi
            samples, _ = play_game(mcts, cfg)
            result_queue.put(("SAMPLES", _pack_samples(samples)))
        result_queue.put(("DONE", worker_id))


def _pack_samples(samples: list[Sample]) -> dict:
    """Pack samples into numpy arrays only (lightweight, picklable)."""
    n = len(samples)
    if n == 0:
        return {
            "count": 0,
            "pid": np.empty(0, dtype=np.int64),
            "board": np.empty((0, 17, 49), dtype=np.float32),
            "links": np.empty((0, 6, 39), dtype=np.float32),
            "global": np.empty((0, 50), dtype=np.float32),
            "own_hand": np.empty((0, 35), dtype=np.float32),
            "opp_hands": np.empty((0, 105), dtype=np.float32),
            "policy": np.empty((0, 1316), dtype=np.float32),
            "value": np.empty(0, dtype=np.float32),
            "legal": np.empty((0, 1316), dtype=np.bool_),
        }
    return {
        "pid": np.asarray([s.pid for s in samples], dtype=np.int64),
        "board": np.stack([s.board for s in samples]).astype(np.float32),
        "links": np.stack([s.links for s in samples]).astype(np.float32),
        "global": np.stack([s.global_vec for s in samples]).astype(np.float32),
        "own_hand": np.stack([s.own_hand for s in samples]).astype(np.float32),
        "opp_hands": np.stack([s.opp_hands for s in samples]).astype(np.float32),
        "policy": np.stack([s.policy for s in samples]).astype(np.float32),
        "value": np.asarray([s.value for s in samples], dtype=np.float32),
        "legal": np.stack([s.legal for s in samples]).astype(np.bool_),
        "count": n,
    }


def unpack_samples(packed: dict) -> list[Sample]:
    n = packed["count"]
    out = []
    for i in range(n):
        out.append(
            Sample(
                pid=int(packed["pid"][i]),
                board=packed["board"][i],
                links=packed["links"][i],
                global_vec=packed["global"][i],
                own_hand=packed["own_hand"][i],
                opp_hands=packed["opp_hands"][i],
                policy=packed["policy"][i],
                value=float(packed["value"][i]),
                legal=packed["legal"][i],
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
        verbose: bool = True,
    ):
        """Broadcast the current weights and collect samples from all workers.

        Returns (samples, per_worker_sample_counts)."""
        weights = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        cfg = mcts_cfg or {}
        for _ in range(self.n_workers):
            self.cmd_queue.put((weights, games_per_worker, sims, seed, cfg, temperature))

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
