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
from .rust_mcts import RustISMCTS, RustMCTSConfig, heuristic_search
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
        (weights, pool_weights, games, sims, seed_offset, mcts_cfg, temperature,
         mm_prob, heuristic_prob, selfplay_opts, task_id) = cmd
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

        options = dict(selfplay_opts or {})
        options.setdefault("max_moves", 600)
        cfg = SelfPlayConfig(players=4, sims=sims, temperature=temperature, **options)
        for gi in range(games):
            stats: dict = {}
            try:
                game_id = task_id * games + gi
                cfg.seed = seed_base + seed_offset + game_id
                rng = np.random.default_rng(cfg.seed)
                np.random.seed(cfg.seed)
                learner = game_id % 4
                roles = [mcts.search] * 4
                collect = {0, 1, 2, 3}

                for seat in range(4):
                    if seat == learner:
                        continue
                    r = rng.random()
                    if heuristic_prob > 0.0 and r < heuristic_prob:
                        roles[seat] = heuristic_search
                        collect.discard(seat)
                    elif mm_prob > 0.0 and pool and r < (heuristic_prob + mm_prob):
                        opp = pool[rng.integers(len(pool))]
                        roles[seat] = opp.search
                        collect.discard(seat)

                samples, vps = play_game_with_roles(
                    roles, cfg, collect=collect, stats=stats
                )
                stats["vps"] = [float(x) for x in vps]
                result_queue.put(("SAMPLES", _pack_samples(samples, stats)))
            except RuntimeError as exc:
                if "samples discarded" not in str(exc):
                    result_queue.put(("ERROR", f"worker={worker_id} game={gi}: {exc!r}"))
                    break
                # A game that ran past max_moves has no valid terminal target:
                # report the partial diagnostics and move on to the next game.
                result_queue.put(("SAMPLES", _pack_samples([], stats)))
            except Exception as exc:
                result_queue.put(("ERROR", f"worker={worker_id} game={gi}: {exc!r}"))
                break
        result_queue.put(("DONE", worker_id))


def _pack_samples(samples: list[Sample], stats: dict | None = None) -> dict:
    """Pack samples into numpy arrays only (lightweight, picklable)."""
    diagnostics = {
        "failed_applies": int((stats or {}).get("failed_applies", 0)),
        "rewritten_applies": int((stats or {}).get("rewritten_applies", 0)),
        "moves": int((stats or {}).get("moves", 0)),
        "vps": [float(x) for x in (stats or {}).get("vps", [])],
        "collected_vps": (stats or {}).get("collected_vps", []),
        "filtered_games": int((stats or {}).get("filtered_games", 0)),
    }
    n = len(samples)
    if n and samples[0].policy_by_canonical is not None:
        # Snapshot form: a few KB per sample instead of a dense candidate matrix.
        return {
            **diagnostics,
            "mode": "snapshot",
            "count": n,
            "pid": np.asarray([s.pid for s in samples], dtype=np.int64),
            "era": np.asarray([s.era for s in samples], dtype=np.int64),
            "snapshot": [s.snapshot for s in samples],
            "policy_moves": [list(s.policy_by_canonical) for s in samples],
            "policy_probs": [
                np.asarray(list(s.policy_by_canonical.values()), dtype=np.float32)
                for s in samples
            ],
            "played": [s.played_canonical for s in samples],
            "value": np.stack([s.value for s in samples]).astype(np.float32),
            "winner": np.stack([s.winner for s in samples]).astype(np.float32),
            "econ": np.stack([s.econ for s in samples]).astype(np.float32),
        }
    if n == 0:
        return {
            "count": 0,
            **diagnostics,
            "mode": "dense",
            "pid": np.empty(0, dtype=np.int64),
            "era": np.empty(0, dtype=np.int64),
            "cells": np.empty((0, be.BOARD_CELLS, be.F_CELL), dtype=np.float32),
            "links": np.empty((0, be.LINK_CELLS, be.F_LINK), dtype=np.float32),
            "merchants": np.empty((0, be.MERCHANT_COUNT, be.F_MERCHANT), dtype=np.float32),
            "seats": np.empty((0, be.SEAT_COUNT, be.F_SEAT), dtype=np.float32),
            "global": np.empty((0, be.F_GLOBAL), dtype=np.float32),
            "candidates": np.empty((0, 0, ACTION_FEATURE_DIM), dtype=np.float32),
            "candidate_mask": np.empty((0, 0), dtype=np.bool_),
            "policy": np.empty((0, 0), dtype=np.float32),
            "value": np.empty((0, 4), dtype=np.float32),
            "winner": np.empty((0, 4), dtype=np.float32),
            "econ": np.empty((0, 2), dtype=np.float32),
        }
    candidates, candidate_mask = pad_candidate_features(
        [torch.from_numpy(s.candidates) for s in samples]
    )
    policy = np.zeros(candidate_mask.shape, dtype=np.float32)
    for i, sample in enumerate(samples):
        policy[i, :len(sample.policy)] = sample.policy
    return {
        **diagnostics,
        "pid": np.asarray([s.pid for s in samples], dtype=np.int64),
        "era": np.asarray([s.era for s in samples], dtype=np.int64),
        "cells": np.stack([s.cells for s in samples]).astype(np.float32),
        "links": np.stack([s.links for s in samples]).astype(np.float32),
        "merchants": np.stack([s.merchants for s in samples]).astype(np.float32),
        "seats": np.stack([s.seats for s in samples]).astype(np.float32),
        "global": np.stack([s.global_vec for s in samples]).astype(np.float32),
        "candidates": candidates.numpy(),
        "candidate_mask": candidate_mask.numpy(),
        "policy": policy,
        "played": [s.played_canonical for s in samples],
        "value": np.stack([s.value for s in samples]).astype(np.float32),
        "winner": np.stack([s.winner for s in samples]).astype(np.float32),
        "econ": np.stack([s.econ for s in samples]).astype(np.float32),
        "count": n,
    }


def unpack_samples(packed: dict) -> list[Sample]:
    n = packed["count"]
    if packed.get("mode") == "snapshot":
        return [
            Sample(
                pid=int(packed["pid"][i]),
                era=int(packed["era"][i]),
                snapshot=packed["snapshot"][i],
                played_canonical=packed["played"][i],
                policy_by_canonical=dict(
                    zip(packed["policy_moves"][i], packed["policy_probs"][i])
                ),
                value=packed["value"][i].astype(np.float32),
                winner=packed["winner"][i].astype(np.float32),
                econ=packed["econ"][i].astype(np.float32),
            )
            for i in range(n)
        ]
    out = []
    for i in range(n):
        out.append(
            Sample(
                pid=int(packed["pid"][i]),
                era=int(packed["era"][i]),
                cells=packed["cells"][i],
                links=packed["links"][i],
                merchants=packed["merchants"][i],
                seats=packed["seats"][i],
                global_vec=packed["global"][i],
                candidates=packed["candidates"][i, packed["candidate_mask"][i]],
                policy=packed["policy"][i, packed["candidate_mask"][i]],
                played_canonical=packed["played"][i],
                value=packed["value"][i].astype(np.float32),
                winner=packed["winner"][i].astype(np.float32),
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
        # Filled by `generate`: aggregate self-play diagnostics of the last call.
        self.last_diagnostics: dict = {}
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
        heuristic_prob: float = 0.0,
        selfplay_opts: dict | None = None,
        verbose: bool = True,
    ):
        """Broadcast the current weights (plus matchmaking pool) and collect
        samples from all workers.

        `mm_pool` is a list of state-dicts of historical nets used as opponent
        seats with probability `mm_prob` per game (learner seat = current net).
        `heuristic_prob` mixes in the fast Rust heuristic teacher as opponent seats.
        `selfplay_opts` overrides `SelfPlayConfig` fields in the worker
        (`store_snapshots`, `determinize_observation`, temperature schedule, ...).
        Returns (samples, per_worker_sample_counts)."""
        weights = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        pool_weights = []  # empty -> workers build no opponent pool (pure self-play)
        cfg = mcts_cfg or {}
        if mm_pool:
            pool_weights = [{k: v.detach().cpu() for k, v in pw.items()} for pw in mm_pool]
        for task_id in range(self.n_workers):
            self.cmd_queue.put(
                (weights, pool_weights, games_per_worker, sims, seed, cfg, temperature,
                 mm_prob, heuristic_prob, selfplay_opts, task_id)
            )

        # Packets and DONE markers arrive interleaved (workers finish at
        # different times); count DONEs until every worker has reported.
        total_games = self.n_workers * games_per_worker
        prog = Progress(total_games, f"selfplay w={self.n_workers} sims={sims}")
        samples = []
        counts = []
        game_vps = []
        collected_vps = []
        filtered_games = 0
        failed_applies = 0
        rewritten_applies = 0
        move_total = 0
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
                failed_applies += int(payload.get("failed_applies", 0))
                rewritten_applies += int(payload.get("rewritten_applies", 0))
                move_total += int(payload.get("moves", 0))
                if payload.get("vps"):
                    game_vps.append(payload["vps"])
                collected_vps.extend(payload.get("collected_vps", []))
                filtered_games += payload.get("filtered_games", 0)
                games_received += 1
                if verbose:
                    prog.update(games_received)
        if verbose:
            prog.done()
        self.last_diagnostics = {
            "failed_applies": failed_applies,
            "rewritten_applies": rewritten_applies,
            "moves": move_total,
            "games": games_received,
            "game_vps": game_vps,
            "collected_vps": collected_vps,
            "filtered_games": filtered_games,
        }
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
