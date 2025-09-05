"""
Gestor de artefactos de modelo por símbolo optimizado.
Maneja modelos PPO, checkpoints, backups y estrategias de forma segura y eficiente.

Estructura:
models/{symbol}/
├── {symbol}_PPO.zip                       # Modelo principal
├── {symbol}_PPO.zip.backup                # Backup del modelo
├── {symbol}_PPO.checksum                  # SHA256 del modelo principal
├── {symbol}_strategies.json               # Mejores estrategias (TOP-K)
├── {symbol}_strategies_provisional.jsonl  # Estrategias provisionales (append-only)
├── {symbol}_bad_strategies.json           # Estrategias descartadas
├── {symbol}_progress.json                 # Progreso del entrenamiento
├── {symbol}_runs.jsonl                    # Historial de runs
└── checkpoints/
    ├── checkpoint_1000000.zip
    ├── checkpoint_2000000.zip
    └── ...
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm


class ModelManager:
    """
    Gestor centralizado de artefactos de modelo por símbolo (robusto y seguro).

    Mejoras:
    - Carga lazy y caché con verificación de checksum.
    - Escrituras atómicas (tmp → rename) y backups automáticos.
    - Validación de archivos y checksums SHA256.
    - Checkpoints asíncronos y limpieza controlada.
    - Pruning paralelo de estrategias (dedupe + scoring compuesto).
    - Timeouts y locks para thread safety.
    """

    # ---------------------------
    # Construcción / setup
    # ---------------------------
    def __init__(
        self,
        symbol: str,
        models_root: str = "models",
        overwrite: bool = False,
        enable_cache: bool = True,
        cache_size_mb: int = 256,           # reservado por si en el futuro añadimos policy cache binario
        validation_enabled: bool = True,
        timeout_seconds: int = 30,
        logger: Optional[logging.Logger] = None,
    ):
        self.symbol = symbol.upper().strip()
        self.models_root = Path(models_root)
        self.symbol_dir = self.models_root / self.symbol
        self.overwrite = overwrite
        self.enable_cache = enable_cache
        self.cache_size_mb = cache_size_mb
        self.validation_enabled = validation_enabled
        self.timeout_seconds = timeout_seconds

        # Thread-safety
        self._lock = threading.RLock()
        self._model_cache: Optional[PPO] = None
        self._cache_timestamp = 0.0
        self._cache_checksum: Optional[str] = None

        # Logging
        self.logger = logger or logging.getLogger(f"ModelManager-{self.symbol}")
        if not self.logger.handlers:
            _h = logging.StreamHandler()
            _h.setFormatter(logging.Formatter(f"[MODEL-MGR-{self.symbol}] %(levelname)s: %(message)s"))
            self.logger.addHandler(_h)
        self.logger.setLevel(logging.INFO)

        # Estructura mínima
        with self._lock:
            self.symbol_dir.mkdir(parents=True, exist_ok=True)
            (self.symbol_dir / "checkpoints").mkdir(exist_ok=True)
            (self.symbol_dir / "temp").mkdir(exist_ok=True)

        # Rutas
        self._setup_file_paths()

        # Executor para I/O async
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"ModelMgr-{self.symbol}")

    def __del__(self):
        try:
            if hasattr(self, "_executor") and self._executor:
                self._executor.shutdown(wait=False)
            self._clear_model_cache()
        except Exception:
            pass

    def _setup_file_paths(self) -> None:
        s = self.symbol
        d = self.symbol_dir
        self.model_path = d / f"{s}_PPO.zip"
        self.backup_path = d / f"{s}_PPO.zip.backup"
        self.checksum_path = d / f"{s}_PPO.checksum"
        self.strategies_path = d / f"{s}_strategies.json"
        self.provisional_path = d / f"{s}_strategies_provisional.jsonl"
        self.bad_strategies_path = d / f"{s}_bad_strategies.json"
        self.progress_path = d / f"{s}_progress.json"
        self.runs_path = d / f"{s}_runs.jsonl"
        self.temp_dir = d / "temp"
        self.ckpt_dir = d / "checkpoints"

    # ---------------------------
    # Utilidades internas
    # ---------------------------
    @contextmanager
    def _timeout_context(self, op: str):
        start = time.time()
        try:
            yield
            elapsed = time.time() - start
            if elapsed > self.timeout_seconds * 0.8:
                self.logger.warning(f"{op} tardó {elapsed:.2f}s (límite {self.timeout_seconds}s)")
        except Exception as e:
            elapsed = time.time() - start
            self.logger.error(f"{op} falló tras {elapsed:.2f}s: {e}")
            raise

    def _calculate_checksum(self, path: Path) -> str:
        if not path.exists():
            return ""
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _validate_model_file(self, path: Path) -> bool:
        if not path.exists():
            return False
        if not self.validation_enabled:
            return True
        try:
            if path.stat().st_size < 1024:
                return False
            # Si hay checksum guardado, verifícalo
            chkfile = path.with_suffix(f"{path.suffix}.checksum")
            if chkfile.exists():
                expected = chkfile.read_text().strip()
                actual = self._calculate_checksum(path)
                return expected == actual
            return True
        except Exception as e:
            self.logger.error(f"Validando {path}: {e}")
            return False

    def _save_checksum(self, path: Path) -> None:
        if not self.validation_enabled or not path.exists():
            return
        try:
            chk = self._calculate_checksum(path)
            chkfile = path.with_suffix(f"{path.suffix}.checksum")
            chkfile.write_text(chk)
        except Exception as e:
            self.logger.warning(f"No se pudo guardar checksum para {path}: {e}")

    def _atomic_move(self, src: Path, dst: Path) -> None:
        os.replace(src, dst)

    def _safe_copy(self, src: Path, dst: Path) -> bool:
        try:
            with tempfile.NamedTemporaryFile(dir=self.temp_dir, delete=False, suffix=".tmp") as tf:
                tmp = Path(tf.name)
            shutil.copy2(src, tmp)
            if self.validation_enabled:
                if self._calculate_checksum(src) != self._calculate_checksum(tmp):
                    tmp.unlink(missing_ok=True)
                    return False
            self._atomic_move(tmp, dst)
            return True
        except Exception as e:
            self.logger.error(f"Copia segura {src} -> {dst} falló: {e}")
            try:
                tmp.unlink(missing_ok=True)  # type: ignore
            except Exception:
                pass
            return False

    def _clear_model_cache(self) -> None:
        with self._lock:
            if self._model_cache is not None:
                try:
                    if hasattr(self._model_cache, "policy"):
                        del self._model_cache.policy
                    if hasattr(self._model_cache, "env"):
                        self._model_cache.env = None
                except Exception:
                    pass
                del self._model_cache
                self._model_cache = None
                self._cache_timestamp = 0.0
                self._cache_checksum = None
                gc.collect()

    def _is_cache_valid(self) -> bool:
        if not self.enable_cache or self._model_cache is None:
            return False
        try:
            file_mtime = self.model_path.stat().st_mtime
            if file_mtime > self._cache_timestamp:
                return False
            if self.validation_enabled and self._cache_checksum:
                return self._calculate_checksum(self.model_path) == self._cache_checksum
            return True
        except Exception:
            return False

    # ---------------------------
    # Carga / guardado de modelo
    # ---------------------------
    def load_model(self, env=None, force_reload: bool = False, **ppo_kwargs) -> Optional[PPO]:
        """
        Carga el modelo principal, cae a backup o checkpoint si es necesario;
        si no existe, crea uno nuevo con kwargs dados.
        """
        with self._lock:
            if not force_reload and self._is_cache_valid():
                self.logger.info("Usando modelo desde caché")
                if env is not None and self._model_cache is not None:
                    self._model_cache.set_env(env)
                return self._model_cache

            self._clear_model_cache()
            model = self._load_existing_model(env, **ppo_kwargs)

            if model is not None and self.enable_cache:
                self._model_cache = model
                self._cache_timestamp = time.time()
                if self.validation_enabled and self.model_path.exists():
                    self._cache_checksum = self._calculate_checksum(self.model_path)
            return model

    def _load_existing_model(self, env=None, **ppo_kwargs) -> Optional[PPO]:
        # Intenta principal
        if self.model_path.exists() and not self.overwrite:
            if self._validate_model_file(self.model_path):
                try:
                    with self._timeout_context("Carga modelo principal"):
                        self.logger.info(f"Cargando modelo: {self.model_path}")
                        return PPO.load(str(self.model_path), env=env)
                except Exception as e:
                    self.logger.error(f"Error cargando modelo principal: {e}")
            else:
                self.logger.warning("Modelo principal falló validación")

        # Intenta backup
        if self.backup_path.exists():
            if self._validate_model_file(self.backup_path):
                try:
                    with self._timeout_context("Carga backup"):
                        self.logger.info(f"Cargando backup: {self.backup_path}")
                        model = PPO.load(str(self.backup_path), env=env)
                        # Restaurar como principal
                        self._safe_copy(self.backup_path, self.model_path)
                        self._save_checksum(self.model_path)
                        self.logger.info("Backup restaurado como principal")
                        return model
                except Exception as e:
                    self.logger.error(f"Error cargando backup: {e}")
            else:
                self.logger.warning("Backup falló validación")

        # Intenta mejor checkpoint
        ckpt_model = self._load_best_checkpoint_internal(env)
        if ckpt_model is not None:
            self.logger.info("Modelo cargado desde mejor checkpoint")
            self.save_model(ckpt_model, create_backup=False)
            return ckpt_model

        # Crea nuevo
        return self._create_new_model(env, **ppo_kwargs)

    def _create_new_model(self, env=None, **ppo_kwargs) -> PPO:
        if env is None:
            raise ValueError("Se requiere 'env' para crear un nuevo modelo")
        self.logger.info(f"Creando nuevo modelo PPO para {self.symbol}")
        default_kwargs = dict(
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
        )
        default_kwargs.update(ppo_kwargs)
        with self._timeout_context("Creación de nuevo modelo"):
            model = PPO("MlpPolicy", env=env, **default_kwargs)
        self.save_model(model, create_backup=False)
        return model

    def save_model(self, model: PPO, create_backup: bool = True) -> bool:
        """
        Guarda el modelo de forma atómica:
        - Crea backup del anterior si existe
        - Escribe a un tmp y renombra
        - Escribe checksum
        - Actualiza caché
        """
        with self._lock:
            try:
                if create_backup and self.model_path.exists():
                    if not self._safe_copy(self.model_path, self.backup_path):
                        self.logger.warning("No se pudo crear backup, continuo…")
                    else:
                        self.logger.info(f"Backup creado: {self.backup_path}")

                with tempfile.NamedTemporaryFile(dir=self.temp_dir, delete=False, suffix=".zip") as tf:
                    tmp = Path(tf.name)

                with self._timeout_context("Guardado modelo"):
                    model.save(str(tmp))

                # valida tmp
                if not self._validate_model_file(tmp):
                    tmp.unlink(missing_ok=True)
                    raise ValueError("Archivo temporal de modelo no válido")

                # mueve a final
                if not self._safe_copy(tmp, self.model_path):
                    tmp.unlink(missing_ok=True)
                    raise ValueError("No se pudo mover el archivo de modelo a destino")
                tmp.unlink(missing_ok=True)

                self._save_checksum(self.model_path)

                # cache
                if self.enable_cache:
                    self._clear_model_cache()
                    self._model_cache = model
                    self._cache_timestamp = time.time()
                    if self.validation_enabled:
                        self._cache_checksum = self._calculate_checksum(self.model_path)

                self.logger.info(f"Modelo guardado: {self.model_path}")
                return True
            except Exception as e:
                self.logger.error(f"Error guardando modelo: {e}")
                return False

    # ---------------------------
    # Checkpoints
    # ---------------------------
    def save_checkpoint(self, model: PPO, timesteps: int) -> bool:
        """Guarda un checkpoint de forma asíncrona (no bloqueante)."""
        try:
            ckpt_path = self.ckpt_dir / f"checkpoint_{timesteps}.zip"
            fut = self._executor.submit(self._save_checkpoint_async, model, ckpt_path)

            def _done(f):
                try:
                    ok = f.result()
                    if ok:
                        self.logger.info(f"Checkpoint guardado: {ckpt_path}")
                    else:
                        self.logger.error(f"Error guardando checkpoint: {ckpt_path}")
                except Exception as ex:
                    self.logger.error(f"Excepción en checkpoint async: {ex}")

            fut.add_done_callback(_done)
            return True
        except Exception as e:
            self.logger.error(f"Error iniciando guardado de checkpoint: {e}")
            return False

    def _save_checkpoint_async(self, model: PPO, ckpt_path: Path) -> bool:
        try:
            with tempfile.NamedTemporaryFile(dir=self.temp_dir, delete=False, suffix=".zip") as tf:
                tmp = Path(tf.name)
            model.save(str(tmp))
            if self._validate_model_file(tmp):
                if self._safe_copy(tmp, ckpt_path):
                    self._save_checksum(ckpt_path)
                    tmp.unlink(missing_ok=True)
                    return True
            tmp.unlink(missing_ok=True)
            return False
        except Exception as e:
            self.logger.error(f"_save_checkpoint_async: {e}")
            return False

    def _load_best_checkpoint_internal(self, env=None) -> Optional[PPO]:
        if not self.ckpt_dir.exists():
            return None
        valid: List[Tuple[int, Path]] = []
        for f in self.ckpt_dir.glob("checkpoint_*.zip"):
            if not self._validate_model_file(f):
                continue
            try:
                ts = int(f.stem.split("_")[1])
                valid.append((ts, f))
            except Exception:
                continue
        if not valid:
            return None
        valid.sort(key=lambda x: x[0], reverse=True)
        best = valid[0][1]
        try:
            with self._timeout_context("Carga mejor checkpoint"):
                self.logger.info(f"Cargando mejor checkpoint: {best}")
                return PPO.load(str(best), env=env)
        except Exception as e:
            self.logger.error(f"Error cargando checkpoint: {e}")
            return None

    def load_best_checkpoint(self, env=None) -> Optional[PPO]:
        with self._lock:
            return self._load_best_checkpoint_internal(env)

    def cleanup_old_checkpoints(self, keep_last: int = 5) -> None:
        """Elimina checkpoints antiguos (asíncrono), conservando los últimos N."""
        try:
            self._executor.submit(self._cleanup_checkpoints_async, keep_last)
        except Exception as e:
            self.logger.error(f"Error iniciando limpieza de checkpoints: {e}")

    def _cleanup_checkpoints_async(self, keep_last: int) -> None:
        try:
            if not self.ckpt_dir.exists():
                return
            files = list(self.ckpt_dir.glob("checkpoint_*.zip"))
            if len(files) <= keep_last:
                return
            files.sort(key=lambda x: int(x.stem.split("_")[1]) if x.stem.split("_")[1].isdigit() else -1, reverse=True)
            for old in files[keep_last:]:
                try:
                    old.unlink(missing_ok=True)
                    chk = old.with_suffix(f"{old.suffix}.checksum")
                    chk.unlink(missing_ok=True)
                    self.logger.info(f"Checkpoint eliminado: {old}")
                except Exception as e:
                    self.logger.warning(f"No se pudo eliminar {old}: {e}")
        except Exception as e:
            self.logger.error(f"Limpieza async checkpoints: {e}")

    # ---------------------------
    # Estrategias (Top-K)
    # ---------------------------
    def prune_strategies(self, top_k: int = 1000) -> None:
        """
        Podado de estrategias:
        - Dedupe por rasgos clave
        - Scoring compuesto (ProfitFactor, WinRate, PnL)
        - Guarda Top-K de forma segura
        """
        if not self.strategies_path.exists():
            self.logger.info(f"No hay estrategias para podar en {self.strategies_path}")
            return
        try:
            with self._timeout_context("Pruning estrategias"):
                strategies = json.loads(self.strategies_path.read_text(encoding="utf-8"))
                if not isinstance(strategies, list) or not strategies:
                    self.logger.info("Estrategias vacías o inválidas")
                    return

                self.logger.info(f"Podando {len(strategies)} estrategias…")

                # Dedupe
                uniq = self._remove_duplicates(strategies)

                # Scoring (paralelo si es grande)
                if len(uniq) > 1000:
                    scored = self._calculate_strategy_scores_parallel(uniq)
                else:
                    scored = self._calculate_strategy_scores(uniq)

                # Orden y top-k
                scored.sort(key=lambda s: s.get("_composite_score", 0.0), reverse=True)
                top = scored[:top_k]

                # Limpia claves temporales
                for s in top:
                    for k in ("_composite_score", "_profit_factor", "_win_rate", "_pnl_score"):
                        s.pop(k, None)

                # Guardado seguro
                self._save_strategies_safe(top)
                self.logger.info(f"Pruning completado: {len(top)}/{top_k} estrategias")
                self._print_pruning_stats(top)
        except Exception as e:
            self.logger.error(f"Error durante pruning: {e}")

    def _remove_duplicates(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for s in strategies:
            key = (
                round(float(s.get("entry_price", 0.0)), 8),
                round(float(s.get("exit_price", 0.0)), 8),
                round(float(s.get("sl", 0.0)), 8),
                round(float(s.get("tp", 0.0)), 8),
                round(float(s.get("leverage", 1.0)), 4),
                str(s.get("exec_tf", "")),
                int(s.get("bars_held", 0)),
            )
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    def _remove_duplicates_parallel(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Disponible por si quieres forzar paralelismo en dedupe
        chunks = max(4, os.cpu_count() or 4)
        step = max(100, len(strategies) // chunks)
        parts = [strategies[i : i + step] for i in range(0, len(strategies), step)]

        def _proc(chunk):
            seen = set()
            uniq = []
            for s in chunk:
                key = (
                    round(float(s.get("entry_price", 0.0)), 8),
                    round(float(s.get("exit_price", 0.0)), 8),
                    round(float(s.get("sl", 0.0)), 8),
                    round(float(s.get("tp", 0.0)), 8),
                    round(float(s.get("leverage", 1.0)), 4),
                    str(s.get("exec_tf", "")),
                    int(s.get("bars_held", 0)),
                )
                if key not in seen:
                    seen.add(key)
                    uniq.append(s)
            return uniq

        out, global_seen = [], set()
        with ThreadPoolExecutor(max_workers=chunks) as ex:
            for fut in as_completed([ex.submit(_proc, c) for c in parts]):
                for s in fut.result():
                    key = (
                        round(float(s.get("entry_price", 0.0)), 8),
                        round(float(s.get("exit_price", 0.0)), 8),
                        round(float(s.get("sl", 0.0)), 8),
                        round(float(s.get("tp", 0.0)), 8),
                        round(float(s.get("leverage", 1.0)), 4),
                        str(s.get("exec_tf", "")),
                        int(s.get("bars_held", 0)),
                    )
                    if key not in global_seen:
                        global_seen.add(key)
                        out.append(s)
        return out

    def _calculate_strategy_scores_parallel(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        workers = min(8, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(self._score_strategy, strategies))

    def _calculate_strategy_scores(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._score_strategy(s) for s in strategies]

    def _score_strategy(self, s: Dict[str, Any]) -> Dict[str, Any]:
        pf = self._calculate_profit_factor(s)
        wr = self._calculate_win_rate(s)
        pnl = self._calculate_pnl_score(s)
        # pesos: 40% PF, 30% WR, 30% PnL
        comp = 0.4 * pf + 0.3 * wr + 0.3 * pnl
        s["_profit_factor"] = pf
        s["_win_rate"] = wr
        s["_pnl_score"] = pnl
        s["_composite_score"] = comp
        return s

    def _calculate_profit_factor(self, s: Dict[str, Any]) -> float:
        """
        Preferencia:
        - Si hay 'gross_profit' y 'gross_loss' → PF = GP / |GL| (cap a 10.0).
        - Si hay 'wins'/'losses' + 'avg_win'/'avg_loss' → PF ≈ (wins*avg_win)/(|losses*avg_loss|).
        - Si no, cae a ROI heurístico.
        """
        gp = float(s.get("gross_profit", 0.0))
        gl = float(s.get("gross_loss", 0.0))
        if gp > 0 and gl < 0:
            return max(0.0, min(10.0, gp / abs(gl)))

        wins = int(s.get("wins", 0))
        losses = int(s.get("losses", 0))
        avg_win = float(s.get("avg_win", 0.0))
        avg_loss = float(s.get("avg_loss", 0.0))
        if wins + losses > 0 and (avg_win > 0 or avg_loss < 0):
            denom = abs(losses * avg_loss) if losses > 0 and avg_loss < 0 else 1.0
            num = wins * max(avg_win, 0.0)
            return max(0.0, min(10.0, num / denom)) if denom > 0 else 10.0

        roi = float(s.get("roi_pct", 0.0))
        if roi > 0:
            return min(10.0, roi / 10.0)
        return 0.0

    def _calculate_win_rate(self, s: Dict[str, Any]) -> float:
        """
        Si existen 'wins' y 'trades' → WR = 100*wins/trades (cap a 100).
        En su defecto, si ROI>0 → 60..100 lineal; si ROI<=0 → 0.
        Escalamos a 0..10 para el score compuesto.
        """
        wins = s.get("wins")
        trades = s.get("trades") or s.get("count")
        if isinstance(wins, (int, float)) and isinstance(trades, (int, float)) and trades > 0:
            wr_pct = max(0.0, min(100.0, 100.0 * float(wins) / float(trades)))
            return wr_pct / 10.0  # 0..10

        roi = float(s.get("roi_pct", 0.0))
        if roi > 0:
            # Heurística suave: ROI 0..50 → WR score 4..10
            return max(0.0, min(10.0, 4.0 + (roi / 50.0) * 6.0))
        return 0.0

    def _calculate_pnl_score(self, s: Dict[str, Any]) -> float:
        """
        PnL score 0..10 (cap):
        - Si existe 'realized_pnl' → escala log/lineal suave.
        - Si no, cae a ROI% / 10 cap 10.
        Penaliza ≤0 fuertemente (hasta -5 antes de truncar a 0).
        """
        pnl = float(s.get("realized_pnl", 0.0))
        if pnl > 0:
            return min(10.0, pnl / 100.0)  # sencillo, configurable
        if pnl < 0:
            return 0.0
        roi = float(s.get("roi_pct", 0.0))
        return max(0.0, min(10.0, roi / 10.0))

    def _save_strategies_safe(self, strategies: List[Dict[str, Any]]) -> None:
        # backup
        if self.strategies_path.exists():
            backup = self.strategies_path.with_suffix(".json.backup")
            shutil.copy2(self.strategies_path, backup)
        # tmp → rename
        with tempfile.NamedTemporaryFile(dir=self.symbol_dir, delete=False, suffix=".json.tmp") as tf:
            tmp = Path(tf.name)
        tmp.write_text(json.dumps(strategies, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.strategies_path)

    def _print_pruning_stats(self, top: List[Dict[str, Any]]) -> None:
        if not top:
            return
        self.logger.info("🏆 TOP ESTRATEGIAS (10):")
        for i, s in enumerate(top[:10], 1):
            roi = float(s.get("roi_pct", 0.0))
            pnl = float(s.get("realized_pnl", 0.0))
            tf = s.get("exec_tf", "N/A")
            lev = float(s.get("leverage", 1.0))
            bars = int(s.get("bars_held", 0))
            self.logger.info(f"{i:2d}. ROI={roi:6.1f}% | PnL={pnl:8.1f} | TF={tf} | Lev={lev:4.1f}x | Bars={bars:3d}")

    # ---------------------------
    # Info / utilidades varias
    # ---------------------------
    def get_model_info(self) -> Dict[str, Any]:
        info = {
            "symbol": self.symbol,
            "model_exists": self.model_path.exists(),
            "backup_exists": self.backup_path.exists(),
            "strategies_count": 0,
            "provisional_count": 0,
            "bad_strategies_count": 0,
            "checkpoints_count": 0,
            "runs_count": 0,
        }
        try:
            if self.strategies_path.exists():
                data = json.loads(self.strategies_path.read_text(encoding="utf-8"))
                info["strategies_count"] = len(data) if isinstance(data, List) else 0
        except Exception:
            pass
        try:
            if self.provisional_path.exists():
                with self.provisional_path.open("r", encoding="utf-8") as f:
                    info["provisional_count"] = sum(1 for _ in f)
        except Exception:
            pass
        try:
            if self.bad_strategies_path.exists():
                data = json.loads(self.bad_strategies_path.read_text(encoding="utf-8"))
                info["bad_strategies_count"] = len(data) if isinstance(data, List) else 0
        except Exception:
            pass
        if self.ckpt_dir.exists():
            info["checkpoints_count"] = len(list(self.ckpt_dir.glob("checkpoint_*.zip")))
        try:
            if self.runs_path.exists():
                with self.runs_path.open("r", encoding="utf-8") as f:
                    info["runs_count"] = sum(1 for _ in f)
        except Exception:
            pass
        return info

    def print_summary(self) -> None:
        i = self.get_model_info()
        self.logger.info(f"Resumen {self.symbol}: "
                         f"Modelo={'OK' if i['model_exists'] else 'NO'}, "
                         f"Backup={'OK' if i['backup_exists'] else 'NO'}, "
                         f"Strategies={i['strategies_count']}, "
                         f"Provisional={i['provisional_count']}, "
                         f"Bad={i['bad_strategies_count']}, "
                         f"Checkpoints={i['checkpoints_count']}, "
                         f"Runs={i['runs_count']}")

    def get_file_paths(self) -> Dict[str, Path]:
        return {
            "model": self.model_path,
            "backup": self.backup_path,
            "checksum": self.checksum_path,
            "strategies": self.strategies_path,
            "provisional": self.provisional_path,
            "bad_strategies": self.bad_strategies_path,
            "progress": self.progress_path,
            "runs": self.runs_path,
            "checkpoints_dir": self.ckpt_dir,
        }

    def cleanup_provisional(self) -> None:
        if self.provisional_path.exists():
            self.provisional_path.unlink(missing_ok=True)
            self.logger.info(f"Provisional limpiado: {self.provisional_path}")

    # Compat helper (API previa)
    def ensure_safe_save(self, model: PPO) -> None:
        _ = self.save_model(model, create_backup=True)
        self.logger.info(f"Guardado seguro completado para {self.symbol}")
