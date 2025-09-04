# train_env/callbacks/anti_bad_strategy.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
from stable_baselines3.common.callbacks import BaseCallback

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
