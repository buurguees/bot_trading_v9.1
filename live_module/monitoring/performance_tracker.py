from __future__ import annotations
import time

class Heartbeat:
    def __init__(self, interval_s: float = 0.0):
        self.interval_s = interval_s
        self._last = time.time()

    def ping(self):
        now = time.time()
        if self.interval_s and (now - self._last) >= self.interval_s:
            self._last = now
