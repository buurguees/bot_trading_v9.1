# train_env/callbacks/periodic_checkpoint.py
from __future__ import annotations
import os
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path

class PeriodicCheckpoint(BaseCallback):
    def __init__(self, save_every_steps: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_every_steps = int(save_every_steps)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_every_steps == 0:
            if self.verbose: print(f"[CHECKPOINT] {self.num_timesteps:,} steps")
        return True
