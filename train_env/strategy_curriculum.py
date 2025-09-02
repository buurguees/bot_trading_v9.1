# train_env/strategy_curriculum.py
# Sistema de curriculum learning basado en estrategias exitosas existentes
from __future__ import annotations
from pathlib import Path
import json
import random
from typing import Dict, Any, List, Optional
import numpy as np

class StrategyCurriculum:
    """
    Sistema de curriculum learning que usa estrategias existentes para guiar el entrenamiento
    - Analiza patrones de las mejores estrategias
    - Sugiere parámetros de trading basados en éxito histórico
    - Implementa "imitation learning" suave
    """
    
    def __init__(self, strategies_file: str, verbose: bool = True):
        self.strategies_file = Path(strategies_file)
        self.verbose = verbose
        self.strategies = []
        self.patterns = {}
        self._load_strategies()
        self._analyze_patterns()
    
    def _load_strategies(self) -> None:
        """Carga las estrategias del archivo JSON"""
        if not self.strategies_file.exists():
            if self.verbose:
                print(f"[CURRICULUM] No se encontró archivo de estrategias: {self.strategies_file}")
            return
            
        try:
            with self.strategies_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.strategies = data
                    if self.verbose:
                        print(f"[CURRICULUM] Cargadas {len(self.strategies)} estrategias")
                else:
                    self.strategies = []
        except Exception as e:
            if self.verbose:
                print(f"[CURRICULUM] Error cargando estrategias: {e}")
            self.strategies = []
    
    def _analyze_patterns(self) -> None:
        """Analiza patrones comunes en las mejores estrategias"""
        if not self.strategies:
            return
            
        # Filtrar solo estrategias exitosas (ROI > 0)
        successful = [s for s in self.strategies if s.get("roi_pct", 0) > 0]
        if not successful:
            return
            
        # Análisis de patrones
        self.patterns = {
            "timeframes": {},
            "duration_ranges": {},
            "entry_conditions": {},
            "exit_conditions": {},
            "risk_levels": {},
            "avg_roi": np.mean([s.get("roi_pct", 0) for s in successful]),
            "avg_r_multiple": np.mean([s.get("r_multiple", 0) for s in successful]),
            "success_rate": len(successful) / len(self.strategies)
        }
        
        # Análisis de timeframes preferidos
        for strat in successful:
            tf = strat.get("exec_tf", "unknown")
            self.patterns["timeframes"][tf] = self.patterns["timeframes"].get(tf, 0) + 1
            
            # Duración de trades
            bars = strat.get("bars_held", 0)
            if bars <= 10:
                self.patterns["duration_ranges"]["short"] = self.patterns["duration_ranges"].get("short", 0) + 1
            elif bars <= 50:
                self.patterns["duration_ranges"]["medium"] = self.patterns["duration_ranges"].get("medium", 0) + 1
            else:
                self.patterns["duration_ranges"]["long"] = self.patterns["duration_ranges"].get("long", 0) + 1
        
        if self.verbose:
            print(f"[CURRICULUM] Patrones analizados:")
            print(f"   - Timeframes preferidos: {dict(sorted(self.patterns['timeframes'].items(), key=lambda x: x[1], reverse=True)[:3])}")
            print(f"   - Duración preferida: {dict(sorted(self.patterns['duration_ranges'].items(), key=lambda x: x[1], reverse=True))}")
            print(f"   - ROI promedio: {self.patterns['avg_roi']:.2f}%")
            print(f"   - R-multiple promedio: {self.patterns['avg_r_multiple']:.2f}")
    
    def get_trading_hints(self) -> Dict[str, Any]:
        """Devuelve sugerencias de trading basadas en estrategias exitosas"""
        if not self.patterns:
            return {}
            
        hints = {
            "preferred_timeframes": [],
            "optimal_duration": "medium",
            "target_roi": self.patterns.get("avg_roi", 10.0),
            "target_r_multiple": self.patterns.get("avg_r_multiple", 2.0),
            "confidence": self.patterns.get("success_rate", 0.5)
        }
        
        # Timeframes preferidos
        if self.patterns.get("timeframes"):
            sorted_tfs = sorted(self.patterns["timeframes"].items(), key=lambda x: x[1], reverse=True)
            hints["preferred_timeframes"] = [tf for tf, _ in sorted_tfs[:3]]
        
        # Duración óptima
        if self.patterns.get("duration_ranges"):
            optimal = max(self.patterns["duration_ranges"].items(), key=lambda x: x[1])[0]
            hints["optimal_duration"] = optimal
        
        return hints
    
    def suggest_action_modification(self, current_action: int, obs: Dict[str, Any]) -> Optional[int]:
        """
        Sugiere modificaciones a la acción actual basándose en estrategias exitosas
        Implementa "imitation learning" suave
        """
        if not self.strategies or random.random() > 0.1:  # Solo 10% de las veces
            return None
            
        # Buscar estrategia similar en el contexto actual
        similar_strategy = self._find_similar_strategy(obs)
        if not similar_strategy:
            return None
            
        # Sugerir acción basada en la estrategia exitosa
        suggested_action = self._action_from_strategy(similar_strategy, obs)
        if suggested_action is not None and suggested_action != current_action:
            if self.verbose:
                print(f"[CURRICULUM] Sugerencia: {current_action} → {suggested_action} (basado en estrategia exitosa)")
            return suggested_action
            
        return None
    
    def _find_similar_strategy(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Encuentra una estrategia similar al contexto actual"""
        if not self.strategies:
            return None
            
        # Filtrar estrategias exitosas
        successful = [s for s in self.strategies if s.get("roi_pct", 0) > 5.0]
        if not successful:
            return None
            
        # Seleccionar aleatoriamente una estrategia exitosa
        return random.choice(successful)
    
    def _action_from_strategy(self, strategy: Dict[str, Any], obs: Dict[str, Any]) -> Optional[int]:
        """Convierte una estrategia en una acción sugerida"""
        # Por ahora, implementación simple
        # En el futuro, esto podría ser más sofisticado
        
        # Basarse en el R-multiple de la estrategia
        r_multiple = strategy.get("r_multiple", 0)
        if r_multiple > 2.0:  # Estrategia muy exitosa
            return 1  # Sugerir LONG
        elif r_multiple < -2.0:  # Estrategia muy exitosa en SHORT
            return 2  # Sugerir SHORT
        else:
            return 0  # Sugerir HOLD
            
        return None
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Devuelve estadísticas del curriculum para monitoreo"""
        return {
            "strategies_loaded": len(self.strategies),
            "patterns_analyzed": bool(self.patterns),
            "success_rate": self.patterns.get("success_rate", 0.0),
            "avg_roi": self.patterns.get("avg_roi", 0.0),
            "confidence": self.patterns.get("success_rate", 0.0)
        }
