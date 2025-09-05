# train_env/callbacks/strategy_keeper.py
from __future__ import annotations
from stable_baselines3.common.callbacks import BaseCallback
from ..utilities.strategy_aggregator import aggregate_top_k

class StrategyKeeper(BaseCallback):
    def __init__(self, provisional_file: str, best_json_file: str, top_k: int, every_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.prov = provisional_file
        self.best = best_json_file
        self.top_k = int(top_k)
        self.every = int(every_steps)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0:
            if self.verbose: print("[STRAT] Aggregating provisional -> best.json ...")
            aggregate_top_k(self.prov, self.best, self.top_k)
        return True
