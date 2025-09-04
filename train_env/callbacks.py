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

class AntiBadStrategy(BaseCallback):
    """
    ← NUEVO: Callback que identifica y evita las PEORES estrategias
    - Carga las 500 peores estrategias de {symbol}_strategies.json
    - Las usa para NO repetir patrones de pérdida
    - Implementa "negative learning" o "avoidance learning"
    """
    def __init__(self, strategies_file: str, bad_strategies_file: str, consult_every_steps: int = 100000, verbose: int = 0):
        super().__init__(verbose)
        self.strategies_file = Path(strategies_file)
        self.bad_strategies_file = Path(bad_strategies_file)
        self.consult_every_steps = int(consult_every_steps)
        self.bad_strategies = []
        self.last_consultation = 0
        
    def _on_training_start(self) -> None:
        """Cargar estrategias malas al inicio del entrenamiento"""
        self._load_bad_strategies()
        if self.verbose:
            print(f"[ANTI-BAD] Cargadas {len(self.bad_strategies)} estrategias MALAS para evitar")
    
    def _on_step(self) -> bool:
        """Consultar estrategias malas periódicamente"""
        if self.num_timesteps - self.last_consultation >= self.consult_every_steps:
            self._load_bad_strategies()
            self.last_consultation = self.num_timesteps
            if self.verbose:
                print(f"[ANTI-BAD] Estrategias malas actualizadas: {len(self.bad_strategies)} para evitar")
        return True
    
    def _load_bad_strategies(self) -> None:
        """Carga las PEORES estrategias del archivo JSON"""
        if not self.strategies_file.exists():
            self.bad_strategies = []
            return
            
        try:
            # Manejar archivos vacíos o corruptos
            if self.strategies_file.stat().st_size == 0:
                self.bad_strategies = []
                return
            with self.strategies_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Ordenar por ROI (peor primero) y tomar las 500 peores
                    sorted_strategies = sorted(data, key=lambda x: x.get("roi_pct", 0))
                    self.bad_strategies = sorted_strategies[:500]  # Las 500 peores
                    
                    # Guardar en archivo separado para consulta rápida
                    self.bad_strategies_file.parent.mkdir(parents=True, exist_ok=True)
                    with self.bad_strategies_file.open("w", encoding="utf-8") as f:
                        json.dump(self.bad_strategies, f, ensure_ascii=False, indent=2)
                        
                else:
                    self.bad_strategies = []
        except Exception as e:
            if self.verbose:
                print(f"[ANTI-BAD] Error cargando estrategias malas: {e}")
            self.bad_strategies = []
    
    def get_bad_strategies(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Devuelve las estrategias más malas para evitar"""
        return self.bad_strategies[:limit] if self.bad_strategies else []
    
    def is_similar_to_bad_strategy(self, current_action: int, obs: Dict[str, Any]) -> bool:
        """
        Verifica si la acción actual es similar a una estrategia mala
        Retorna True si debe EVITARSE
        """
        if not self.bad_strategies:
            return False
            
        # Por ahora, implementación simple
        # En el futuro, esto podría ser más sofisticado (análisis de patrones)
        
        # Buscar estrategias muy malas (ROI < -20%)
        very_bad = [s for s in self.bad_strategies if s.get("roi_pct", 0) < -20.0]
        if not very_bad:
            return False
            
        # Si la acción actual es similar a una estrategia muy mala, evitarla
        for bad_strat in very_bad[:10]:  # Solo revisar las 10 peores
            if self._is_action_similar_to_bad_strategy(current_action, bad_strat):
                if self.verbose:
                    print(f"[ANTI-BAD] ⚠️ Acción {current_action} similar a estrategia mala (ROI: {bad_strat.get('roi_pct', 0):.1f}%)")
                return True
                
        return False
    
    def _is_action_similar_to_bad_strategy(self, action: int, bad_strategy: Dict[str, Any]) -> bool:
        """
        Verifica si una acción es similar a una estrategia mala
        Por ahora, implementación simple basada en el tipo de acción
        """
        # Si la estrategia mala fue LONG y la acción actual es LONG, podría ser similar
        if action == 1 and bad_strategy.get("side", 0) == 1:  # LONG
            return True
        elif action == 2 and bad_strategy.get("side", 0) == -1:  # SHORT
            return True
            
        return False
    
    def get_avoidance_stats(self) -> Dict[str, Any]:
        """Devuelve estadísticas de evitación para monitoreo"""
        if not self.bad_strategies:
            return {"bad_strategies_loaded": 0}
            
        return {
            "bad_strategies_loaded": len(self.bad_strategies),
            "worst_roi": min(s.get("roi_pct", 0) for s in self.bad_strategies),
            "avg_bad_roi": sum(s.get("roi_pct", 0) for s in self.bad_strategies) / len(self.bad_strategies),
            "very_bad_count": len([s for s in self.bad_strategies if s.get("roi_pct", 0) < -20.0])
        }

class MainModelSaver(BaseCallback):
    """
    Guarda el modelo principal usando ModelManager para gestión segura
    """
    def __init__(self, save_every_steps: int, fixed_path: str, model_manager=None, verbose: int = 0):
        super().__init__(verbose)
        self.save_every_steps = int(save_every_steps)
        self.fixed_path = Path(fixed_path)
        self.fixed_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_manager = model_manager

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_every_steps == 0:
            if self.verbose: print(f"[MODEL] Saving to {self.fixed_path}")
            
            if self.model_manager:
                # Usar ModelManager para guardado seguro
                self.model_manager.ensure_safe_save(self.model)
            else:
                # Fallback al método original
                self.model.save(str(self.fixed_path))
        return True
