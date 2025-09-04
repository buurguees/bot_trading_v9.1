"""
Gestor de artefactos de modelo por símbolo.
Maneja modelos PPO, checkpoints, backups y estrategias de forma segura.
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from stable_baselines3 import PPO


class ModelManager:
    """
    Gestor centralizado de artefactos de modelo por símbolo.
    
    Estructura de directorios:
    models/{symbol}/
    ├── {symbol}_PPO.zip              # Modelo principal
    ├── {symbol}_PPO.zip.backup       # Backup del modelo
    ├── {symbol}_strategies.json      # Mejores estrategias (TOP-1000)
    ├── {symbol}_strategies_provisional.jsonl  # Estrategias provisionales
    ├── {symbol}_bad_strategies.json  # Estrategias malas para evitar
    ├── {symbol}_progress.json        # Progreso del entrenamiento
    ├── {symbol}_runs.jsonl           # Historial de runs
    └── checkpoints/                  # Checkpoints periódicos
        ├── checkpoint_1000000.zip
        ├── checkpoint_2000000.zip
        └── ...
    """
    
    def __init__(self, symbol: str, models_root: str = "models", overwrite: bool = False):
        self.symbol = symbol
        self.models_root = Path(models_root)
        self.symbol_dir = self.models_root / symbol
        self.overwrite = overwrite
        
        # Crear directorio del símbolo
        self.symbol_dir.mkdir(parents=True, exist_ok=True)
        (self.symbol_dir / "checkpoints").mkdir(exist_ok=True)
        
        # Rutas de archivos
        self.model_path = self.symbol_dir / f"{symbol}_PPO.zip"
        self.backup_path = self.symbol_dir / f"{symbol}_PPO.zip.backup"
        self.strategies_path = self.symbol_dir / f"{symbol}_strategies.json"
        self.provisional_path = self.symbol_dir / f"{symbol}_strategies_provisional.jsonl"
        self.bad_strategies_path = self.symbol_dir / f"{symbol}_bad_strategies.json"
        self.progress_path = self.symbol_dir / f"{symbol}_progress.json"
        self.runs_path = self.symbol_dir / f"{symbol}_runs.jsonl"
    
    def load_model(self, env=None, **ppo_kwargs) -> Optional[PPO]:
        """
        Carga el modelo PPO existente o crea uno nuevo.
        
        Args:
            env: Entorno para crear nuevo modelo si no existe
            **ppo_kwargs: Parámetros para PPO si se crea nuevo modelo
            
        Returns:
            Modelo PPO cargado o nuevo
        """
        if self.model_path.exists() and not self.overwrite:
            try:
                print(f"[MODEL-MANAGER] Cargando modelo existente: {self.model_path}")
                return PPO.load(str(self.model_path), env=env)
            except Exception as e:
                print(f"[MODEL-MANAGER] Error cargando modelo: {e}")
                if self.backup_path.exists():
                    print(f"[MODEL-MANAGER] Intentando cargar backup: {self.backup_path}")
                    try:
                        return PPO.load(str(self.backup_path), env=env)
                    except Exception as e2:
                        print(f"[MODEL-MANAGER] Error cargando backup: {e2}")
        
        # Crear nuevo modelo
        if env is None:
            raise ValueError("Se requiere 'env' para crear nuevo modelo")
        
        print(f"[MODEL-MANAGER] Creando nuevo modelo para {self.symbol}")
        model = PPO("MlpPolicy", env=env, **ppo_kwargs)
        
        # Guardar inmediatamente
        self.save_model(model)
        
        return model
    
    def save_model(self, model: PPO, create_backup: bool = True) -> None:
        """
        Guarda el modelo PPO.
        
        Args:
            model: Modelo PPO a guardar
            create_backup: Si crear backup del modelo anterior
        """
        # Crear backup si existe modelo anterior
        if create_backup and self.model_path.exists():
            shutil.copy2(self.model_path, self.backup_path)
            print(f"[MODEL-MANAGER] Backup creado: {self.backup_path}")
        
        # Guardar modelo
        model.save(str(self.model_path))
        print(f"[MODEL-MANAGER] Modelo guardado: {self.model_path}")
    
    def save_checkpoint(self, model: PPO, timesteps: int) -> None:
        """
        Guarda un checkpoint del modelo.
        
        Args:
            model: Modelo PPO
            timesteps: Número de timesteps para el nombre del archivo
        """
        checkpoint_path = self.symbol_dir / "checkpoints" / f"checkpoint_{timesteps}.zip"
        model.save(str(checkpoint_path))
        print(f"[MODEL-MANAGER] Checkpoint guardado: {checkpoint_path}")
    
    def load_best_checkpoint(self, env=None) -> Optional[PPO]:
        """
        Carga el mejor checkpoint disponible.
        
        Args:
            env: Entorno para cargar el modelo
            
        Returns:
            Mejor modelo PPO o None si no hay checkpoints
        """
        checkpoints_dir = self.symbol_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return None
        
        # Buscar el checkpoint con más timesteps
        checkpoint_files = list(checkpoints_dir.glob("checkpoint_*.zip"))
        if not checkpoint_files:
            return None
        
        # Ordenar por timesteps (mayor primero)
        checkpoint_files.sort(key=lambda x: int(x.stem.split("_")[1]), reverse=True)
        best_checkpoint = checkpoint_files[0]
        
        try:
            print(f"[MODEL-MANAGER] Cargando mejor checkpoint: {best_checkpoint}")
            return PPO.load(str(best_checkpoint), env=env)
        except Exception as e:
            print(f"[MODEL-MANAGER] Error cargando checkpoint: {e}")
            return None
    
    def cleanup_old_checkpoints(self, keep_last: int = 5) -> None:
        """
        Limpia checkpoints antiguos, manteniendo solo los últimos N.
        
        Args:
            keep_last: Número de checkpoints a mantener
        """
        checkpoints_dir = self.symbol_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return
        
        checkpoint_files = list(checkpoints_dir.glob("checkpoint_*.zip"))
        if len(checkpoint_files) <= keep_last:
            return
        
        # Ordenar por timesteps (mayor primero)
        checkpoint_files.sort(key=lambda x: int(x.stem.split("_")[1]), reverse=True)
        
        # Eliminar checkpoints antiguos
        for old_checkpoint in checkpoint_files[keep_last:]:
            old_checkpoint.unlink()
            print(f"[MODEL-MANAGER] Checkpoint eliminado: {old_checkpoint}")
    
    def prune_strategies(self, top_k: int = 1000) -> None:
        """
        ← NUEVO: Sistema de pruning de estrategias mejorado.
        Mantiene Top-K estrategias por Profit Factor, Win Rate y PnL acumulado.
        Elimina duplicados y malas estrategias automáticamente.
        
        Args:
            top_k: Número de mejores estrategias a mantener
        """
        if not self.strategies_path.exists():
            print(f"[MODEL-MANAGER] No hay estrategias para podar en {self.strategies_path}")
            return
        
        try:
            # Cargar estrategias existentes
            with self.strategies_path.open("r", encoding="utf-8") as f:
                strategies = json.load(f)
            
            if not strategies:
                print("[MODEL-MANAGER] No hay estrategias para podar")
                return
            
            print(f"[MODEL-MANAGER] Podando {len(strategies)} estrategias...")
            
            # 1. Eliminar duplicados basados en características clave
            unique_strategies = self._remove_duplicates(strategies)
            print(f"[MODEL-MANAGER] Estrategias únicas después de deduplicación: {len(unique_strategies)}")
            
            # 2. Calcular scores compuestos
            scored_strategies = self._calculate_strategy_scores(unique_strategies)
            
            # 3. Ordenar por score compuesto
            scored_strategies.sort(key=lambda x: x["_composite_score"], reverse=True)
            
            # 4. Mantener solo Top-K
            top_strategies = scored_strategies[:top_k]
            
            # 5. Remover scores temporales
            for strategy in top_strategies:
                strategy.pop("_composite_score", None)
                strategy.pop("_profit_factor", None)
                strategy.pop("_win_rate", None)
                strategy.pop("_pnl_score", None)
            
            # 6. Guardar estrategias podadas
            self._save_strategies_safe(top_strategies)
            
            print(f"[MODEL-MANAGER] ✅ Pruning completado: {len(top_strategies)}/{top_k} estrategias mantenidas")
            
            # 7. Mostrar estadísticas
            self._print_pruning_stats(top_strategies)
            
        except Exception as e:
            print(f"[MODEL-MANAGER] ❌ Error durante pruning: {e}")
    
    def _remove_duplicates(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Elimina estrategias duplicadas basándose en características clave.
        """
        seen = set()
        unique_strategies = []
        
        for strategy in strategies:
            # Crear clave única basada en características principales
            key = (
                strategy.get("entry_price", 0.0),
                strategy.get("exit_price", 0.0),
                strategy.get("sl", 0.0),
                strategy.get("tp", 0.0),
                strategy.get("leverage", 1.0),
                strategy.get("exec_tf", ""),
                strategy.get("bars_held", 0)
            )
            
            if key not in seen:
                seen.add(key)
                unique_strategies.append(strategy)
        
        return unique_strategies
    
    def _calculate_strategy_scores(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calcula scores compuestos para cada estrategia.
        """
        for strategy in strategies:
            # Profit Factor
            profit_factor = self._calculate_profit_factor(strategy)
            strategy["_profit_factor"] = profit_factor
            
            # Win Rate
            win_rate = self._calculate_win_rate(strategy)
            strategy["_win_rate"] = win_rate
            
            # PnL Score
            pnl_score = self._calculate_pnl_score(strategy)
            strategy["_pnl_score"] = pnl_score
            
            # Score compuesto (pesos: 40% PF, 30% WR, 30% PnL)
            composite_score = (
                0.4 * profit_factor +
                0.3 * win_rate +
                0.3 * pnl_score
            )
            strategy["_composite_score"] = composite_score
        
        return strategies
    
    def _calculate_profit_factor(self, strategy: Dict[str, Any]) -> float:
        """
        Calcula el Profit Factor de una estrategia.
        """
        roi_pct = strategy.get("roi_pct", 0.0)
        realized_pnl = strategy.get("realized_pnl", 0.0)
        
        if realized_pnl <= 0:
            return 0.0
        
        # Profit Factor simplificado basado en ROI
        if roi_pct > 0:
            return min(10.0, roi_pct / 10.0)  # Normalizar a [0, 10]
        else:
            return max(-5.0, roi_pct / 10.0)  # Penalizar pérdidas
    
    def _calculate_win_rate(self, strategy: Dict[str, Any]) -> float:
        """
        Calcula el Win Rate de una estrategia.
        """
        roi_pct = strategy.get("roi_pct", 0.0)
        
        # Win Rate basado en ROI positivo
        if roi_pct > 0:
            return min(10.0, roi_pct / 5.0)  # Normalizar a [0, 10]
        else:
            return 0.0
    
    def _calculate_pnl_score(self, strategy: Dict[str, Any]) -> float:
        """
        Calcula el score de PnL de una estrategia.
        """
        realized_pnl = strategy.get("realized_pnl", 0.0)
        
        # Score basado en PnL absoluto
        if realized_pnl > 0:
            return min(10.0, realized_pnl / 100.0)  # Normalizar a [0, 10]
        else:
            return max(-5.0, realized_pnl / 100.0)  # Penalizar pérdidas
    
    def _save_strategies_safe(self, strategies: List[Dict[str, Any]]) -> None:
        """
        Guarda estrategias de forma segura con backup.
        """
        # Crear backup si existe archivo anterior
        if self.strategies_path.exists():
            backup_path = self.strategies_path.with_suffix(".json.backup")
            shutil.copy2(self.strategies_path, backup_path)
        
        # Guardar nuevas estrategias
        with self.strategies_path.open("w", encoding="utf-8") as f:
            json.dump(strategies, f, ensure_ascii=False, indent=2)
    
    def _print_pruning_stats(self, top_strategies: List[Dict[str, Any]]) -> None:
        """
        Imprime estadísticas del pruning.
        """
        if not top_strategies:
            return
        
        print(f"\n🏆 TOP ESTRATEGIAS DESPUÉS DEL PRUNING:")
        print("=" * 80)
        
        for i, strategy in enumerate(top_strategies[:10], 1):  # Top 10
            roi_pct = strategy.get("roi_pct", 0.0)
            realized_pnl = strategy.get("realized_pnl", 0.0)
            exec_tf = strategy.get("exec_tf", "N/A")
            leverage = strategy.get("leverage", 1.0)
            bars_held = strategy.get("bars_held", 0)
            
            print(f"{i:2d}. ROI: {roi_pct:6.1f}% | PnL: {realized_pnl:8.1f} | TF: {exec_tf} | Lev: {leverage:4.1f}x | Bars: {bars_held:3d}")
        
        print("=" * 80)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el modelo y sus artefactos.
        
        Returns:
            Diccionario con información del modelo
        """
        info = {
            "symbol": self.symbol,
            "model_exists": self.model_path.exists(),
            "backup_exists": self.backup_path.exists(),
            "strategies_count": 0,
            "provisional_count": 0,
            "bad_strategies_count": 0,
            "checkpoints_count": 0,
            "runs_count": 0
        }
        
        # Contar estrategias
        if self.strategies_path.exists():
            try:
                with self.strategies_path.open("r") as f:
                    strategies = json.load(f)
                    info["strategies_count"] = len(strategies) if isinstance(strategies, list) else 0
            except:
                pass
        
        # Contar estrategias provisionales
        if self.provisional_path.exists():
            try:
                with self.provisional_path.open("r") as f:
                    info["provisional_count"] = sum(1 for _ in f)
            except:
                pass
        
        # Contar estrategias malas
        if self.bad_strategies_path.exists():
            try:
                with self.bad_strategies_path.open("r") as f:
                    bad_strategies = json.load(f)
                    info["bad_strategies_count"] = len(bad_strategies) if isinstance(bad_strategies, list) else 0
            except:
                pass
        
        # Contar checkpoints
        checkpoints_dir = self.symbol_dir / "checkpoints"
        if checkpoints_dir.exists():
            info["checkpoints_count"] = len(list(checkpoints_dir.glob("checkpoint_*.zip")))
        
        # Contar runs
        if self.runs_path.exists():
            try:
                with self.runs_path.open("r") as f:
                    info["runs_count"] = sum(1 for _ in f)
            except:
                pass
        
        return info
    
    def print_summary(self) -> None:
        """Imprime un resumen del estado del modelo."""
        info = self.get_model_info()
        
        print(f"\n[MODEL-MANAGER] Resumen para {self.symbol}:")
        print(f"  Modelo principal: {'OK' if info['model_exists'] else 'NO'}")
        print(f"  Backup: {'OK' if info['backup_exists'] else 'NO'}")
        print(f"  Estrategias: {info['strategies_count']}")
        print(f"  Provisional: {info['provisional_count']}")
        print(f"  Estrategias malas: {info['bad_strategies_count']}")
        print(f"  Checkpoints: {info['checkpoints_count']}")
        print(f"  Runs: {info['runs_count']}")
    
    def get_file_paths(self) -> Dict[str, Path]:
        """
        Obtiene las rutas de todos los archivos del modelo.
        
        Returns:
            Diccionario con las rutas de archivos
        """
        return {
            "model": self.model_path,
            "backup": self.backup_path,
            "strategies": self.strategies_path,
            "provisional": self.provisional_path,
            "bad_strategies": self.bad_strategies_path,
            "progress": self.progress_path,
            "runs": self.runs_path,
            "checkpoints_dir": self.symbol_dir / "checkpoints"
        }
    
    def cleanup_provisional(self) -> None:
        """Limpia el archivo de estrategias provisionales."""
        if self.provisional_path.exists():
            self.provisional_path.unlink()
            print(f"[MODEL-MANAGER] Provisional limpiado: {self.provisional_path}")
    
    def ensure_safe_save(self, model: PPO) -> None:
        """
        Guarda el modelo de forma segura (con backup automático).
        
        Args:
            model: Modelo PPO a guardar
        """
        # Crear backup si existe modelo anterior
        if self.model_path.exists():
            shutil.copy2(self.model_path, self.backup_path)
        
        # Guardar modelo
        model.save(str(self.model_path))
        
        print(f"[MODEL-MANAGER] Guardado seguro completado para {self.symbol}")
