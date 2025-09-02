# base_env/logging/run_logger.py
# Descripción: Logger de runs acumulados en models/{symbol}/

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import json

class RunLogger:
    def __init__(self, symbol: str, models_root: str = "models"):
        self.symbol = symbol
        self.dir = Path(models_root) / symbol
        self.dir.mkdir(parents=True, exist_ok=True)
        self.runs_file = self.dir / f"{symbol}_runs.jsonl"
        self.progress_file = self.dir / f"{symbol}_progress.json"
        self._active: Optional[Dict[str, Any]] = None

    def start(self, market: str, initial_balance: float, target_balance: float, initial_equity: float, ts_start: int):
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
        }

    def finish(self, final_balance: float, final_equity: float, ts_end: int, bankruptcy: bool = False, penalty_reward: float = 0.0):
        if not self._active:
            print("⚠️ RunLogger.finish() llamado sin run activo")
            return

        # ← NUEVO: Validación de datos para prevenir duplicados
        if self._active.get("ts_end") is not None:
            print(f"🚫 Run ya finalizado, ignorando finish() duplicado")
            return

        # ← NUEVO: Validación de episodios vacíos
        ts_start = self._active.get("ts_start", 0)
        if ts_end - ts_start < 1000:  # Menos de 1 segundo
            print(f"🚫 Episodio demasiado corto ({ts_end - ts_start}ms) - NO se loguea")
            self._active = None
            return

        # ← NUEVO: Validación de actividad real
        initial_balance = self._active.get("initial_balance", 0.0)
        if abs(final_balance - initial_balance) < 0.01 and abs(final_equity - initial_balance) < 0.01:
            print(f"🚫 Episodio sin actividad real (balance/equity sin cambios) - NO se loguea")
            self._active = None
            return

        self._active["final_balance"] = float(final_balance)
        self._active["final_equity"] = float(final_equity)
        self._active["ts_end"] = int(ts_end)
        self._active["hit_target"] = final_balance >= self._active["target_balance"]
        
        # ← NUEVO: información de quiebra
        self._active["bankruptcy"] = bankruptcy
        if bankruptcy:
            self._active["penalty_reward"] = float(penalty_reward)
            self._active["drawdown_pct"] = ((final_equity - self._active["initial_equity"]) / self._active["initial_equity"]) * 100.0
            self._active["run_result"] = "BANKRUPTCY"
        else:
            self._active["run_result"] = "COMPLETED" if self._active["hit_target"] else "INCOMPLETE"

        # ← NUEVO: Filtro de calidad - no loguear runs con balance muy negativo
        if final_balance < -90000.0:
            print(f"🚫 Run descartado: balance {final_balance:.2f} < -90,000 (muy negativo)")
            self._active = None
            return

        # ← NUEVO: Sistema de rotación FIFO con límite de 400 runs
        self._add_run_with_rotation(self._active)

        # Actualizar snapshot maestro
        self._update_progress(self._active)

        # reset
        self._active = None

    def _add_run_with_rotation(self, run_data: Dict[str, Any]):
        """Añade un nuevo run con rotación FIFO, manteniendo máximo 400 runs"""
        MAX_RUNS = 400
        
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
        
        # Añadir el nuevo run
        existing_runs.append(run_data)
        
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
                f.write(json.dumps(run, ensure_ascii=False) + "\n")
        
        print(f"✅ Run añadido: balance {run_data['final_balance']:.2f}, equity {run_data['final_equity']:.2f}")
        print(f"   Total runs en archivo: {len(existing_runs)}/{MAX_RUNS}")
    
    def _is_duplicate_run(self, new_run: Dict[str, Any]) -> bool:
        """← NUEVO: Detecta si un run es duplicado con lógica anti-congelamiento mejorada"""
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
            
            # Criterios de duplicación ANTI-CONGELAMIENTO
            new_balance = float(new_run.get("final_balance", 0.0))
            new_equity = float(new_run.get("final_equity", 0.0))
            new_ts = int(new_run.get("ts_end", 0))
            
            # 1. 🚨 DETECCIÓN DE CONGELAMIENTO: Múltiples runs idénticos
            identical_count = 0
            for existing_run in recent_runs:
                existing_balance = float(existing_run.get("final_balance", 0.0))
                existing_equity = float(existing_run.get("final_equity", 0.0))
                
                if abs(new_balance - existing_balance) < 0.001 and abs(new_equity - existing_equity) < 0.001:
                    identical_count += 1
            
            # Si hay más de 3 runs idénticos → CONGELAMIENTO detectado
            if identical_count >= 3:
                print(f"🚨 CONGELAMIENTO DETECTADO: {identical_count} runs idénticos - NO se loguea")
                return True
            
            # 2. 🔍 DETECCIÓN DE DUPLICADOS EXACTOS
            for existing_run in recent_runs:
                existing_balance = float(existing_run.get("final_balance", 0.0))
                existing_equity = float(existing_run.get("final_equity", 0.0))
                existing_ts = int(existing_run.get("ts_end", 0))
                
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
            print(f"⚠️ Error en detección de duplicados: {e}")
            return False  # En caso de error, permitir el run

    def cleanup_existing_runs(self):
        """Limpia el archivo existente aplicando el límite de 400 runs (mantiene los más recientes)"""
        MAX_RUNS = 400
        
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
            print(f"✅ Archivo ya está dentro del límite: {len(existing_runs)}/{MAX_RUNS} runs")
            return
        
        # Mantener solo los últimos MAX_RUNS runs
        runs_to_remove = len(existing_runs) - MAX_RUNS
        existing_runs = existing_runs[-MAX_RUNS:]  # Mantener los más recientes
        
        # Reescribir el archivo
        with self.runs_file.open("w", encoding="utf-8") as f:
            for run in existing_runs:
                f.write(json.dumps(run, ensure_ascii=False) + "\n")
        
        print(f"🧹 Limpieza aplicada: eliminados {runs_to_remove} runs antiguos")
        print(f"   Archivo ahora contiene: {len(existing_runs)}/{MAX_RUNS} runs más recientes")

    def _update_progress(self, last_run: Dict[str, Any]):
        runs = []
        if self.runs_file.exists():
            with self.runs_file.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        runs.append(json.loads(line))
                    except Exception:
                        continue

        total_runs = len(runs)
        best_equity = max(r.get("final_equity", 0) for r in runs if r.get("final_equity") is not None)
        best_balance = max(r.get("final_balance", 0) for r in runs if r.get("final_balance") is not None)
        last = runs[-1] if runs else {}

        progress = {
            "symbol": self.symbol,
            "runs_completed": total_runs,
            "best_equity": best_equity,
            "best_balance": best_balance,
            "last_run": last,
            "progress_pct": round((best_balance / last.get("target_balance", 1)) * 100, 2) if last else 0.0,
        }

        self.progress_file.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
