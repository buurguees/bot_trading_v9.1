# train_env/vec_factory.py
# Crea N entornos vectoriales (SubprocVecEnv) con seeds distintos.

from __future__ import annotations
from typing import List, Callable
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from .gym_wrapper import TradingGymWrapper
from .dataset import make_base_env

def make_vec_envs(n_envs: int, seed: int, data_cfg: dict, env_cfg: dict) -> SubprocVecEnv:
    def make_one(rank: int) -> Callable:
        def _thunk():
            np.random.seed(seed + rank)
            base = make_base_env(
                data_root=data_cfg["root"],
                symbol=data_cfg["symbols"][0],
                market=data_cfg["market"],
                tfs=data_cfg["tfs"],
                base_tf=data_cfg["tfs"][0],
                stage=data_cfg["stage"],
                episode_bars=env_cfg["episode_bars"],
                warmup_bars=env_cfg["warmup_bars"],
            )
            return TradingGymWrapper(base_env=base, reward_yaml=env_cfg["reward_yaml"], tfs=data_cfg["tfs"])
        return _thunk
    return SubprocVecEnv([make_one(i) for i in range(n_envs)])
