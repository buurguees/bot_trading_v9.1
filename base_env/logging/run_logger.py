# base_env/logging/run_logger.py
# Descripción: Logger de runs acumulados en models/{symbol}/

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import json
import numpy as np
from ..metrics.trade_metrics import TradeMetrics, TradeRecord
from ..utils.timestamp_utils import add_utc_timestamps

def _convert_numpy_types(obj):
    """Convierte tipos NumPy a tipos nativos de Python para serialización JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj

class RunLogger:
    def __init__(self, symbol: str, models_root: str = "models", max_records: int = 2000, prune_strategy: str = "fifo"):
        self.symbol = symbol
        self.dir = Path(models_root) / symbol
        self.dir.mkdir(parents=True, exist_ok=True)
        self.runs_file = self.dir / f"{symbol}_runs.jsonl"
        self.progress_file = self.dir / f"{symbol}_progress.json"
        self._active: Optional[Dict[str, Any]] = None
        # ← NUEVO: Recolector de métricas de trades
        self._trade_metrics = TradeMetrics()
        # ← NUEVO: Configuración de retención
        self.max_records = max_records
        self.prune_strategy = prune_strategy

    def start(self, market: str, initial_balance: float, target_balance: float, initial_equity: float, ts_start: int, segment_id: int = 0):
        # ← NUEVO: Resetear métricas de trades para el nuevo run
        self._trade_metrics.reset()
        
        import os
        import uuid
        
        self._active = {
            "symbol": self.symbol,
            "market": market,
            "initial_balance": float(initial_balance),
            "target_balance": float(target_balance),
            "initial_equity": float(initial_equity),
            "final_balance": None,
            "final_equity": None,
            "ts_start": int(ts_start),
            "ts_end": None,
            "hit_target": False,
            # ← NUEVO: Contadores para análisis de actividad
            "trades_count": 0,
            "elapsed_steps": 0,
            "reasons_counter": {},  # Contador de razones por las que no operó
            "equity_min": float(initial_equity),
            "bankruptcy_step": None,
            "cumulative_reward": 0.0,
            # ← NUEVO: Control de segmentos para soft reset
            "segment_id": int(segment_id),
            "soft_reset_count": 0,
            # ← NUEVO: Identificadores únicos para deduplicación
            "env_id": int(os.getenv("VEC_ENV_ID", -1)),
            "uuid": str(uuid.uuid4()),
        }

    def update_trades_count(self, count: int):
        """← NUEVO: Actualiza el contador de trades ejecutados"""
        if self._active:
            self._active["trades_count"] = count

    def update_elapsed_steps(self, steps: int):
        """← NUEVO: Actualiza el contador de pasos transcurridos"""
        if self._active:
            self._active["elapsed_steps"] = steps

    def update_equity_min(self, equity_now: float):
        if self._active:
            self._active["equity_min"] = min(float(equity_now), float(self._active.get("equity_min", equity_now)))

    def add_cumulative_reward(self, r: float):
        if self._active:
            self._active["cumulative_reward"] = float(self._active.get("cumulative_reward", 0.0)) + float(r)

    def set_bankruptcy_step(self, step_idx: int):
        if self._active:
            self._active["bankruptcy_step"] = int(step_idx)

    def add_reason(self, reason: str):
        """← NUEVO: Añade una razón por la que no se operó"""
        if self._active:
            if "reasons_counter" not in self._active:
                self._active["reasons_counter"] = {}
            self._active["reasons_counter"][reason] = self._active["reasons_counter"].get(reason, 0) + 1

    def add_trade_record(self, entry_price: float, exit_price: float, qty: float, side: int, 
                        realized_pnl: float, bars_held: int, leverage_used: float = 3.0,
                        open_ts: Optional[int] = None, close_ts: Optional[int] = None, 
                        sl: Optional[float] = None, tp: Optional[float] = None, 
                        roi_pct: float = 0.0, r_multiple: float = 0.0, risk_pct: float = 0.0):
        """← NUEVO: Registra un trade cerrado para métricas profesionales"""
        trade = TradeRecord(
            entry_price=entry_price,
            exit_price=exit_price,
            qty=qty,
            side=side,
            realized_pnl=realized_pnl,
            bars_held=bars_held,
            leverage_used=leverage_used,
            open_ts=open_ts,
            close_ts=close_ts,
            sl=sl,
            tp=tp,
            roi_pct=roi_pct,
            r_multiple=r_multiple,
            risk_pct=risk_pct
        )
        self._trade_metrics.add_trade(trade)

    def finish(self, final_balance: float, final_equity: float, ts_end: int, bankruptcy: bool = False, penalty_reward: float = 0.0, soft_reset: bool = False, reset_count: int = 0):
        """
        Finaliza el run activo de forma robusta. Nunca mata el worker por logging.
        """
        try:
            if not self._active:
                print("WARNING: RunLogger.finish() llamado sin run activo")
                return

            # ← NUEVO: Validación de datos para prevenir duplicados
            if self._active.get("ts_end") is not None:
                print(f"🚫 Run ya finalizado, ignorando finish() duplicado")
                return

            # Actualizar datos finales
            self._active.update({
                "final_balance": self._safe_float(final_balance),
                "final_equity": self._safe_float(final_equity),
                "ts_end": int(ts_end),
                "bankruptcy": bool(bankruptcy),
                "penalty_reward": self._safe_float(penalty_reward),
                "soft_reset": bool(soft_reset),
                "soft_reset_count": int(reset_count),
                "hit_target": final_balance >= self._active.get("target_balance", 0),
            })

            # Calcular métricas profesionales de trades
            trade_metrics = self._trade_metrics.calculate_metrics()
            self._active.update(trade_metrics)
            
            # información de quiebra y soft reset
            if soft_reset:
                self._active["run_result"] = "SOFT_RESET"
            elif bankruptcy:
                self._active["drawdown_pct"] = ((final_equity - self._active.get("initial_equity", 0)) / max(self._active.get("initial_equity", 1), 1)) * 100.0
                self._active["run_result"] = "BANKRUPTCY"
            else:
                self._active["run_result"] = "COMPLETED" if self._active["hit_target"] else "INCOMPLETE"

            # 1) persiste primero
            self._add_run_with_rotation(self._active)
            # 2) luego recalcula progreso con el run activo incluido
            self._update_progress(self._active)

        except Exception as e:
            # jamas matar el worker por logging
            print(f"[RunLogger.finish] fallo no fatal: {e}")
        finally:
            self._active = None

    def _add_run_with_rotation(self, run_data: Dict[str, Any]):
        """Añade un nuevo run con rotación FIFO, manteniendo máximo configurable de runs"""
        MAX_RUNS = self.max_records
        
        # ← NUEVO: Validación anti-duplicados
        if self._is_duplicate_run(run_data):
            print(f"🚫 Run duplicado detectado - NO se añade:")
            print(f"   Balance: {run_data['final_balance']:.2f}, Equity: {run_data['final_equity']:.2f}")
            print(f"   Timestamp: {run_data.get('ts_end', 'N/A')}")
            return
        
        # Cargar runs existentes
        existing_runs = []
        if self.runs_file.exists():
            with self.runs_file.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        existing_runs.append(json.loads(line))
                    except Exception:
                        continue
        
        # Añadir timestamps UTC legibles al run
        run_data_with_utc = add_utc_timestamps(run_data)
        
        # Añadir el nuevo run
        existing_runs.append(run_data_with_utc)
        
        # Si excede el límite, eliminar los más antiguos (FIFO)
        if len(existing_runs) > MAX_RUNS:
            runs_to_remove = len(existing_runs) - MAX_RUNS
            removed_runs = existing_runs[:runs_to_remove]
            existing_runs = existing_runs[runs_to_remove:]
            
            print(f"🔄 Rotación FIFO: eliminados {runs_to_remove} runs antiguos, manteniendo {MAX_RUNS} más recientes")
            print(f"   Runs eliminados: {[r.get('ts_start', 0) for r in removed_runs[:3]]}...")
        
        # Reescribir el archivo completo
        with self.runs_file.open("w", encoding="utf-8") as f:
            for run in existing_runs:
                converted_run = _convert_numpy_types(run)
                f.write(json.dumps(converted_run, ensure_ascii=False) + "\n")
        
        print(f"OK Run añadido: balance {run_data['final_balance']:.2f}, equity {run_data['final_equity']:.2f}")
        print(f"   Total runs en archivo: {len(existing_runs)}/{MAX_RUNS}")
    
    def _run_key(self, r: dict) -> tuple:
        """Genera clave única para deduplicación usando (start_ts, env_id, uuid)"""
        import os
        import uuid
        
        return (
            int(r.get("start_ts", 0)),
            str(r.get("env_id", os.getpid() % 100000)),
            str(r.get("uuid", "")),
        )

    def _is_duplicate_run(self, new_run: Dict[str, Any]) -> bool:
        """← NUEVO: Detecta si un run es duplicado usando clave robusta"""
        if not self.runs_file.exists():
            return False
        
        try:
            # Cargar últimos 20 runs para análisis de patrones
            recent_runs = []
            with self.runs_file.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-20:]:  # Últimos 20 para detectar patrones
                    try:
                        recent_runs.append(json.loads(line.strip()))
                    except Exception:
                        continue
            
            if not recent_runs:
                return False
            
            # Usar clave robusta para deduplicación
            new_key = self._run_key(new_run)
            
            # Verificar duplicados por clave
            for existing_run in recent_runs:
                existing_key = self._run_key(existing_run)
                if new_key == existing_key:
                    print(f"🚫 Run duplicado detectado - NO se añade: Clave {new_key}")
                    return True
            
            # Criterios adicionales de duplicación por contenido
            new_balance = float(new_run.get("final_balance", 0.0))
            new_equity = float(new_run.get("final_equity", 0.0))
            
            # 1. 🚨 DETECCIÓN DE CONGELAMIENTO: Múltiples runs idénticos
            identical_count = 0
            for existing_run in recent_runs:
                existing_balance = float(existing_run.get("final_balance", 0.0))
                existing_equity = float(existing_run.get("final_equity", 0.0))
                
                if abs(new_balance - existing_balance) < 0.001 and abs(new_equity - existing_equity) < 0.001:
                    identical_count += 1
            
            # Si hay más de 5 runs idénticos → CONGELAMIENTO detectado (menos estricto)
            if identical_count >= 5:
                print(f"🚨 CONGELAMIENTO DETECTADO: {identical_count} runs idénticos - NO se loguea")
                return True
                
                # Balance y equity EXACTAMENTE idénticos
                if abs(new_balance - existing_balance) < 0.001 and abs(new_equity - existing_equity) < 0.001:
                    print(f"🔍 Duplicado detectado: valores idénticos")
                    return True
                
                # Timestamp EXACTAMENTE igual
                if new_ts == existing_ts:
                    print(f"🔍 Duplicado detectado: timestamp idéntico")
                    return True
            
            return False
            
        except Exception as e:
            print(f"WARNING: Error en detección de duplicados: {e}")
            return False  # En caso de error, permitir el run

    def cleanup_existing_runs(self):
        """Limpia el archivo existente aplicando el límite configurable de runs (mantiene los más recientes)"""
        MAX_RUNS = self.max_records
        
        if not self.runs_file.exists():
            return
        
        # Cargar todos los runs existentes
        existing_runs = []
        with self.runs_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_runs.append(json.loads(line))
                except Exception:
                    continue
        
        if len(existing_runs) <= MAX_RUNS:
            print(f"OK Archivo ya está dentro del límite: {len(existing_runs)}/{MAX_RUNS} runs")
            return
        
        # Mantener solo los últimos MAX_RUNS runs
        runs_to_remove = len(existing_runs) - MAX_RUNS
        existing_runs = existing_runs[-MAX_RUNS:]  # Mantener los más recientes
        
        # Reescribir el archivo
        with self.runs_file.open("w", encoding="utf-8") as f:
            for run in existing_runs:
                converted_run = _convert_numpy_types(run)
                f.write(json.dumps(converted_run, ensure_ascii=False) + "\n")
        
        print(f"🧹 Limpieza aplicada: eliminados {runs_to_remove} runs antiguos")
        print(f"   Archivo ahora contiene: {len(existing_runs)}/{MAX_RUNS} runs más recientes")

    def _safe_float(self, x, default=0.0):
        """Convierte a float de forma segura con valor por defecto"""
        try:
            return float(x)
        except Exception:
            return default

    def _update_progress(self, last_run: Dict[str, Any]):
        """
        Recalcula progreso global. Tolerante a archivo vacío o sin 'final_equity'.
        """
        try:
            runs = []
            if self.runs_file.exists():
                with self.runs_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            runs.append(json.loads(line))
                        except Exception:
                            continue
        except Exception:
            runs = []

        # si nos pasan el run activo, lo incluimos/actualizamos en memoria antes de calcular
        if last_run:
            runs = runs + [last_run]

        if not runs:
            self._progress = {
                "best_equity": 0.0,
                "best_balance": 0.0,
                "count": 0,
            }
            return

        final_equities = [
            self._safe_float(r.get("final_equity"), None)
            for r in runs if r.get("final_equity") is not None
        ]
        final_balances = [
            self._safe_float(r.get("final_balance"), None)
            for r in runs if r.get("final_balance") is not None
        ]

        best_equity = max(final_equities, default=0.0)
        best_balance = max(final_balances, default=0.0)
        last = runs[-1] if runs else {}

        progress = {
            "symbol": self.symbol,
            "runs_completed": len(runs),
            "best_equity": best_equity,
            "best_balance": best_balance,
            "last_run": last,
            "progress_pct": round((best_balance / last.get("target_balance", 1)) * 100, 2) if last else 0.0,
        }

        converted_progress = _convert_numpy_types(progress)
        self.progress_file.write_text(json.dumps(converted_progress, ensure_ascii=False, indent=2), encoding="utf-8")

    def update_cumulative_reward(self, reward: float):
        """← NUEVO: Actualiza el reward acumulado"""
        if self._active:
            self._active["cumulative_reward"] = float(reward)

    def update_segment_id(self, segment_id: int):
        """← NUEVO: Actualiza el ID del segmento actual"""
        if self._active:
            self._active["segment_id"] = int(segment_id)

    def update_soft_reset_count(self, count: int):
        """← NUEVO: Actualiza el contador de soft resets"""
        if self._active:
            self._active["soft_reset_count"] = int(count)
