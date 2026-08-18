"""Minimal elapsed/ETA progress reporter for long-running steps.

Prints a line every `every_s` seconds so long runs (self-play, evaluation,
training) show live feedback and an estimated time-to-completion instead of
staying silent for minutes.
"""

from __future__ import annotations

import time


class Progress:
    def __init__(self, total: int, label: str, every_s: float = 0.1):
        self.total = max(total, 1)
        self.label = label
        self.every_s = every_s
        self.t0 = time.time()
        self._last = -1e9

    def update(self, done: int, extra: str = "") -> None:
        now = time.time()
        if now - self._last < self.every_s and done < self.total:
            return
        self._last = now
        elapsed = now - self.t0
        done = min(done, self.total)
        rate = done / max(elapsed, 1e-6)
        eta = (self.total - done) / max(rate, 1e-9)
        line = f"[{self.label}] {done}/{self.total} elapsed:{elapsed:5.0f}s  ETA:{eta:5.0f}s"
        if extra:
            line += f" | {extra}"
        
        # use carriage return to overwrite the previous line, and flush to ensure it appears immediately
        print(f"\r{line}".ljust(100), end="", flush=True)

    def done(self) -> None:
        self.update(self.total)
        print()
