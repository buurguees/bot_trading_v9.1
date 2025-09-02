# train_env/gym_wrapper.py
from __future__ import annotations
import gymnasium as gym
import numpy as np
import random
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from base_env.base_env import BaseTradingEnv
from .reward_shaper import RewardShaper
from .strategy_curriculum import StrategyCurriculum

class TradingGymWrapper(gym.Env):
    """
    En SPOT:
      action_space = Discrete(5)  -> 0=policy, 1=close, 2=block, 3=force_long, 4=force_short
    En FUTURES:
      action_space = MultiDiscrete([5, Nlevers])
        - a[0] = acción trading anterior
        - a[1] = índice de leverage (map a valor por [min,max,step] desde YAML)
    """
    metadata = {"render_modes": []}

    def __init__(self, base_env: BaseTradingEnv, reward_yaml: str, tfs: List[str],
                 leverage_spec: Optional[dict] = None, strategy_log_path: Optional[str] = None):
        super().__init__()
        self.env = base_env
        self.tfs = tfs
        self.shaper = RewardShaper(reward_yaml)
        self.strategy_log_path = strategy_log_path
        
        # ← NUEVO: Curriculum learning basado en estrategias existentes
        self.curriculum = None
        if strategy_log_path:
            # Intentar cargar estrategias existentes para curriculum
            strategies_file = strategy_log_path.replace("_provisional.jsonl", "_strategies.json")
            try:
                self.curriculum = StrategyCurriculum(strategies_file, verbose=False)
                print(f"[CURRICULUM] Integrado en {self.__class__.__name__}")
            except Exception as e:
                print(f"[CURRICULUM] No se pudo cargar: {e}")

        # espacios
        self._lev_spec = None
        if leverage_spec:
            mn, mx, st = float(leverage_spec["min"]), float(leverage_spec["max"]), float(leverage_spec.get("step", 1.0))
            n_levels = int(round((mx - mn) / st)) + 1
            self._lev_spec = (mn, mx, st, n_levels)
            self.action_space = gym.spaces.MultiDiscrete([5, n_levels])
        else:
            self.action_space = gym.spaces.Discrete(5)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim(),), dtype=np.float32)

        # logger de estrategias (igual que antes, si lo usabas)
        from .strategy_logger import StrategyLogger
        self.strategy_log = StrategyLogger(strategy_log_path or "models/tmp/tmp_provisional.jsonl")

    def _obs_dim(self) -> int:
        per_tf = 7; pos = 4; ana = 2
        return len(self.tfs)*per_tf + pos + ana

    def _flatten_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        vec: List[float] = []
        for tf in self.tfs:
            bar = obs["tfs"].get(tf, {}); feats = obs["features"].get(tf, {})
            vec.extend([
                float(bar.get("close", 0.0)),
                float(feats.get("ema20", 0.0) or 0.0),
                float(feats.get("ema50", 0.0) or 0.0),
                float(feats.get("rsi14", 50.0) or 50.0),
                float(feats.get("atr14", 0.0) or 0.0),
                float(feats.get("macd_hist", 0.0) or 0.0),
                float(feats.get("bb_p", 0.5) or 0.5),
            ])
        pos = obs.get("position", {})
        vec.extend([float(pos.get("side", 0)), float(pos.get("qty", 0.0)),
                    float(pos.get("entry_price", 0.0)), float(pos.get("unrealized_pnl", 0.0))])
        ana = obs.get("analysis", {})
        vec.extend([float(ana.get("confidence", 0.0)), float(ana.get("side_hint", 0))])
        return np.asarray(vec, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return self._flatten_obs(obs), {}

    def _lev_from_idx(self, idx: int) -> float:
        mn, mx, st, n = self._lev_spec
        return float(mn + idx * st)

    def step(self, action):
        leverage = None
        trade_action = action
        if self._lev_spec is not None:
            # MultiDiscrete
            trade_action = int(action[0])
            lev_idx = int(action[1])
            leverage = self._lev_from_idx(lev_idx)

        # ← NUEVO: Curriculum learning - sugerir modificaciones basadas en estrategias exitosas
        if self.curriculum and random.random() < 0.05:  # 5% de las veces
            # Intentar cargar estrategias malas para evitarlas
            bad_strategies = []
            try:
                bad_strat_file = self.strategy_log_path.replace("_provisional.jsonl", "_bad_strategies.json")
                if Path(bad_strat_file).exists():
                    with open(bad_strat_file, 'r') as f:
                        bad_strategies = json.load(f)
            except:
                pass
                
            suggested_action = self.curriculum.suggest_action_modification(trade_action, {}, bad_strategies)
            if suggested_action is not None:
                trade_action = suggested_action
                print(f"[CURRICULUM] Acción modificada: {action} → {trade_action}")

        # inyecta la acción y el leverage (si aplica)
        self.env.set_action_override(int(trade_action), leverage_override=leverage, leverage_index=lev_idx if self._lev_spec else None)

        obs, base_r, done, info = self.env.step()
        evs = info.get("events", [])
        if evs:
            self.strategy_log.append_many(evs)
        
        # ← NUEVO: Obtener información de milestones y runs vacíos
        balance_milestones = info.get("balance_milestones", 0)
        empty_run = self.env._empty_runs_count > 0 and not evs and not done
        
        shaped, parts = self.shaper.compute(obs, base_r, evs, empty_run, balance_milestones)
        return self._flatten_obs(obs), float(shaped), bool(done), False, {"r_parts": parts, **info}

    def needs_learning_rate_reset(self) -> bool:
        """← NUEVO: Expone el método del entorno base para SubprocVecEnv"""
        return self.env.needs_learning_rate_reset()

    def reset_learning_rate_flag(self):
        """← NUEVO: Expone el método del entorno base para SubprocVecEnv"""
        self.env.reset_learning_rate_flag()
