# train_env/callbacks/strategy_consultant.py
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, Any, List
from stable_baselines3.common.callbacks import BaseCallback

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
            # Manejar archivos vacíos o corruptos
            if self.strategies_file.stat().st_size == 0:
                self.strategies = []
                return
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
