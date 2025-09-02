# train_env/callbacks.py
from __future__ import annotations
import os
from stable_baselines3.common.callbacks import BaseCallback
from .strategy_aggregator import aggregate_top_k
from pathlib import Path
import json
import random
from typing import Dict, Any, List

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

class StrategyConsultant(BaseCallback):
    """
    ← NUEVO: Callback que consulta estrategias existentes para mejorar el aprendizaje
    - Carga las mejores estrategias de {symbol}_strategies.json
    - Las usa para guiar el entrenamiento del agente
    - Implementa "curriculum learning" basado en estrategias exitosas
    """
    def __init__(self, strategies_file: str, consult_every_steps: int = 100000, verbose: int = 0):
        super().__init__(verbose)
        self.strategies_file = Path(strategies_file)
        self.consult_every_steps = int(consult_every_steps)
        self.strategies = []
        self.last_consultation = 0
        
    def _on_training_start(self) -> None:
        """Cargar estrategias al inicio del entrenamiento"""
        self._load_strategies()
        if self.verbose:
            print(f"[STRAT-CONSULT] Cargadas {len(self.strategies)} estrategias existentes")
    
    def _on_step(self) -> bool:
        """Consultar estrategias periódicamente"""
        if self.num_timesteps - self.last_consultation >= self.consult_every_steps:
            self._load_strategies()
            self.last_consultation = self.num_timesteps
            if self.verbose:
                print(f"[STRAT-CONSULT] Estrategias actualizadas: {len(self.strategies)} disponibles")
        return True
    
    def _load_strategies(self) -> None:
        """Carga las mejores estrategias del archivo JSON"""
        if not self.strategies_file.exists():
            self.strategies = []
            return
            
        try:
            with self.strategies_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.strategies = data
                else:
                    self.strategies = []
        except Exception as e:
            if self.verbose:
                print(f"[STRAT-CONSULT] Error cargando estrategias: {e}")
            self.strategies = []
    
    def get_best_strategies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Devuelve las mejores estrategias para consulta"""
        return self.strategies[:limit] if self.strategies else []
    
    def get_random_strategy(self) -> Dict[str, Any]:
        """Devuelve una estrategia aleatoria de las mejores"""
        if not self.strategies:
            return {}
        return random.choice(self.strategies[:min(20, len(self.strategies))])

class MainModelSaver(BaseCallback):
    """
    Guarda el modelo principal en la ruta fija especificada
    """
    def __init__(self, save_every_steps: int, fixed_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_every_steps = int(save_every_steps)
        self.fixed_path = Path(fixed_path)
        self.fixed_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_every_steps == 0:
            if self.verbose: print(f"[MODEL] Saving to {self.fixed_path}")
            self.model.save(str(self.fixed_path))
        return True
