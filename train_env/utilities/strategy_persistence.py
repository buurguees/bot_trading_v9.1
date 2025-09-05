# train_env/strategy_persistence.py
"""
Módulo para persistencia de estrategias con soporte completo para leverage.
Maneja el guardado y carga de estrategias incluyendo el campo leverage.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


class StrategyPersistence:
    """
    Gestor de persistencia de estrategias con soporte completo para leverage.
    """
    
    def __init__(self, strategies_file: str):
        self.strategies_file = Path(strategies_file)
        self.strategies_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _convert_numpy_types(self, obj):
        """Convierte tipos NumPy a tipos nativos de Python para serialización JSON"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def save_strategy(self, strategy_data: Dict[str, Any]) -> None:
        """
        Guarda una estrategia individual con campo leverage incluido.
        
        Args:
            strategy_data: Diccionario con los datos de la estrategia
        """
        # Asegurar que el campo leverage esté presente
        if "leverage" not in strategy_data:
            strategy_data["leverage"] = 1.0  # Default para SPOT
        
        # Convertir tipos NumPy
        strategy_data = self._convert_numpy_types(strategy_data)
        
        # Guardar como JSONL (una línea por estrategia)
        with self.strategies_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(strategy_data, ensure_ascii=False) + "\n")
    
    def load_strategies(self) -> List[Dict[str, Any]]:
        """
        Carga todas las estrategias desde el archivo.
        
        Returns:
            Lista de estrategias cargadas
        """
        strategies = []
        if not self.strategies_file.exists():
            return strategies
        
        with self.strategies_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    strategy = json.loads(line.strip())
                    # Asegurar que el campo leverage esté presente
                    if "leverage" not in strategy:
                        strategy["leverage"] = 1.0
                    strategies.append(strategy)
                except json.JSONDecodeError:
                    continue
        
        return strategies
    
    def update_strategy(self, strategy_id: str, updates: Dict[str, Any]) -> bool:
        """
        Actualiza una estrategia existente.
        
        Args:
            strategy_id: ID de la estrategia a actualizar
            updates: Diccionario con los campos a actualizar
            
        Returns:
            True si se encontró y actualizó la estrategia, False en caso contrario
        """
        strategies = self.load_strategies()
        updated = False
        
        for i, strategy in enumerate(strategies):
            if strategy.get("strategy_id") == strategy_id:
                # Actualizar campos
                strategy.update(updates)
                # Asegurar que leverage esté presente
                if "leverage" not in strategy:
                    strategy["leverage"] = 1.0
                updated = True
                break
        
        if updated:
            # Guardar estrategias actualizadas
            self._save_all_strategies(strategies)
        
        return updated
    
    def _save_all_strategies(self, strategies: List[Dict[str, Any]]) -> None:
        """
        Guarda todas las estrategias al archivo.
        
        Args:
            strategies: Lista de estrategias a guardar
        """
        # Convertir tipos NumPy
        strategies = self._convert_numpy_types(strategies)
        
        with self.strategies_file.open("w", encoding="utf-8") as f:
            for strategy in strategies:
                f.write(json.dumps(strategy, ensure_ascii=False) + "\n")
    
    def get_strategy_by_id(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene una estrategia específica por ID.
        
        Args:
            strategy_id: ID de la estrategia
            
        Returns:
            Estrategia encontrada o None
        """
        strategies = self.load_strategies()
        for strategy in strategies:
            if strategy.get("strategy_id") == strategy_id:
                return strategy
        return None
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """
        Elimina una estrategia por ID.
        
        Args:
            strategy_id: ID de la estrategia a eliminar
            
        Returns:
            True si se encontró y eliminó la estrategia, False en caso contrario
        """
        strategies = self.load_strategies()
        original_count = len(strategies)
        
        # Filtrar estrategias, excluyendo la que se quiere eliminar
        strategies = [s for s in strategies if s.get("strategy_id") != strategy_id]
        
        if len(strategies) < original_count:
            self._save_all_strategies(strategies)
            return True
        
        return False
    
    def get_strategies_by_leverage(self, leverage_range: tuple = None) -> List[Dict[str, Any]]:
        """
        Obtiene estrategias filtradas por rango de leverage.
        
        Args:
            leverage_range: Tupla (min_leverage, max_leverage) o None para todas
            
        Returns:
            Lista de estrategias filtradas
        """
        strategies = self.load_strategies()
        
        if leverage_range is None:
            return strategies
        
        min_lev, max_lev = leverage_range
        filtered = []
        
        for strategy in strategies:
            leverage = strategy.get("leverage", 1.0)
            if min_lev <= leverage <= max_lev:
                filtered.append(strategy)
        
        return filtered
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de las estrategias guardadas.
        
        Returns:
            Diccionario con estadísticas
        """
        strategies = self.load_strategies()
        
        if not strategies:
            return {
                "total_strategies": 0,
                "leverage_stats": {},
                "avg_roi": 0.0,
                "avg_r_multiple": 0.0
            }
        
        # Estadísticas de leverage
        leverages = [s.get("leverage", 1.0) for s in strategies]
        leverage_stats = {
            "min": min(leverages),
            "max": max(leverages),
            "avg": sum(leverages) / len(leverages),
            "count_by_range": {
                "spot (1.0x)": len([l for l in leverages if l == 1.0]),
                "low (1.1-3.0x)": len([l for l in leverages if 1.1 <= l <= 3.0]),
                "medium (3.1-10.0x)": len([l for l in leverages if 3.1 <= l <= 10.0]),
                "high (10.1x+)": len([l for l in leverages if l > 10.0])
            }
        }
        
        # Estadísticas de rendimiento
        rois = [s.get("roi_pct", 0.0) for s in strategies if s.get("roi_pct") is not None]
        r_multiples = [s.get("r_multiple", 0.0) for s in strategies if s.get("r_multiple") is not None]
        
        return {
            "total_strategies": len(strategies),
            "leverage_stats": leverage_stats,
            "avg_roi": sum(rois) / len(rois) if rois else 0.0,
            "avg_r_multiple": sum(r_multiples) / len(r_multiples) if r_multiples else 0.0,
            "profitable_strategies": len([s for s in strategies if s.get("roi_pct", 0.0) > 0]),
            "futures_strategies": len([s for s in strategies if s.get("leverage", 1.0) > 1.0])
        }
    
    def clear_all_strategies(self) -> None:
        """Elimina todas las estrategias del archivo."""
        if self.strategies_file.exists():
            self.strategies_file.unlink()
    
    def backup_strategies(self, backup_path: str) -> None:
        """
        Crea una copia de seguridad de las estrategias.
        
        Args:
            backup_path: Ruta donde guardar el backup
        """
        if self.strategies_file.exists():
            import shutil
            shutil.copy2(self.strategies_file, backup_path)
