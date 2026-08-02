"""Minimal elapsed/ETA progress reporter for long-running steps.

Prints a line every `every_s` seconds so long runs (self-play, evaluation,
training) show live feedback and an estimated time-to-completion instead of
staying silent for minutes.
"""

from __future__ import annotations

import time


class Progress:
    def __init__(self, total: int, label: str, every_s: float = 15.0):
        self.total = max(total, 1)
        self.label = label
        self.every_s = every_s
        self.t0 = time.time()
        self._last = -1e9
        self._last_printed = None

    def update(self, done: int, extra: str = "") -> None:
        now = time.time()
        if now - self._last < self.every_s:
            return
        self._last = now
        elapsed = now - self.t0
        done = min(done, self.total)
        rate = done / max(elapsed, 1e-6)
        eta = (self.total - done) / max(rate, 1e-9)
        line = f"[{self.label}] {done}/{self.total}  elapsed {elapsed:5.0f}s  ETA {eta:5.0f}s"
        if extra:
            line += f"  {extra}"
        print(line, flush=True)
        self._last_printed = line

    def done(self) -> None:
        self.update(self.total)
