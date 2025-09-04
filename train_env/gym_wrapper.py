# train_env/gym_wrapper.py
from __future__ import annotations
import gymnasium as gym
import numpy as np
import random
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from base_env.base_env import BaseTradingEnv
from .reward_shaper import RewardShaper
from .strategy_curriculum import StrategyCurriculum

class TradingGymWrapper(gym.Env):
    """
    En SPOT:
      action_space = Discrete(5)  -> 0=policy, 1=close, 2=block, 3=force_long, 4=force_short
    En FUTURES:
      action_space = MultiDiscrete([5, Nlevers])
        - a[0] = acción trading anterior
        - a[1] = índice de leverage (map a valor por [min,max,step] desde YAML)
    """
    metadata = {"render_modes": []}

    def __init__(self, base_env: BaseTradingEnv, reward_yaml: str, tfs: List[str],
                 leverage_spec: Optional[dict] = None, strategy_log_path: Optional[str] = None):
        super().__init__()
        self.env = base_env
        self.tfs = tfs
        self.shaper = RewardShaper(reward_yaml)
        self.strategy_log_path = strategy_log_path
        
        # ← NUEVO: Sistema de deduplicación de decisiones por (bar_time, side)
        self._decision_cache = set()  # {(bar_time, side)}
        self._last_bar_time = None
        self._last_side = None
        
        # ← NUEVO: Curriculum learning basado en estrategias existentes
        self.curriculum = None
        if strategy_log_path:
            # Intentar cargar estrategias existentes para curriculum
            strategies_file = strategy_log_path.replace("_provisional.jsonl", "_strategies.json")
            try:
                self.curriculum = StrategyCurriculum(strategies_file, verbose=False)
                print(f"[CURRICULUM] Integrado en {self.__class__.__name__}")
            except Exception as e:
                print(f"[CURRICULUM] No se pudo cargar: {e}")

        # espacios
        self._lev_spec = None
        if leverage_spec:
            mn, mx, st = float(leverage_spec["min"]), float(leverage_spec["max"]), float(leverage_spec.get("step", 1.0))
            n_levels = int(round((mx - mn) / st)) + 1
            self._lev_spec = (mn, mx, st, n_levels)
            self.action_space = gym.spaces.MultiDiscrete([5, n_levels])
        else:
            self.action_space = gym.spaces.Discrete(5)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim(),), dtype=np.float32)

        # logger de estrategias (igual que antes, si lo usabas)
        from .strategy_logger import StrategyLogger
        self.strategy_log = StrategyLogger(
            strategy_log_path or "models/tmp/tmp_provisional.jsonl",
            segment_id=0
        )

    def _obs_dim(self) -> int:
        per_tf = 7; pos = 4; ana = 2
        return len(self.tfs)*per_tf + pos + ana

    def _flatten_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        vec: List[float] = []
        for tf in self.tfs:
            bar = obs["tfs"].get(tf, {}); feats = obs["features"].get(tf, {})
            vec.extend([
                float(bar.get("close", 0.0)),
                float(feats.get("ema20", 0.0) or 0.0),
                float(feats.get("ema50", 0.0) or 0.0),
                float(feats.get("rsi14", 50.0) or 50.0),
                float(feats.get("atr14", 0.0) or 0.0),
                float(feats.get("macd_hist", 0.0) or 0.0),
                float(feats.get("bb_p", 0.5) or 0.5),
            ])
        pos = obs.get("position", {})
        vec.extend([float(pos.get("side", 0)), float(pos.get("qty", 0.0)),
                    float(pos.get("entry_price", 0.0)), float(pos.get("unrealized_pnl", 0.0))])
        ana = obs.get("analysis", {})
        vec.extend([float(ana.get("confidence", 0.0)), float(ana.get("side_hint", 0))])
        return np.asarray(vec, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        obs = self.env.reset()
        
        # ← NUEVO: Inicializar sistema de milestones de progreso para el nuevo run
        self.shaper.advanced_rewards.initialize_run(
            initial_balance=self.env._init_cash,
            target_balance=self.env._target_cash
        )
        
        return self._flatten_obs(obs), {}

    def _lev_from_idx(self, idx: int) -> float:
        mn, mx, st, n = self._lev_spec
        return float(mn + idx * st)
    
    def _is_decision_duplicate(self, bar_time: str, side: int) -> bool:
        """
        Deduplicación estricta por (bar_time, side). Se limpia al cambiar de barra
        o al cerrar posición.
        """
        key = (bar_time, side)
        if key in self._decision_cache:
            return True
        self._decision_cache.add(key)
        return False
    
    def _clear_decision_cache_on_close(self, events: List[Dict[str, Any]], info: Dict[str, Any]):
        """
        Limpia el cache de decisiones cuando se cierra una posición.
        """
        # Limpiar cache si se cerró posición (por eventos o flag just_closed)
        if (any(e.get("kind") == "CLOSE" for e in events) or 
            info.get("just_closed", False)):
            self._decision_cache.clear()
            print(f"🧹 Cache de decisiones limpiado por cierre de posición")

    def _apply_default_levels_if_needed(self, decision, obs) -> None:
        """Completa SL/TP/TTL cuando faltan usando defaults del YAML + ATR fallback."""
        try:
            if not getattr(decision, 'should_open', False):
                return
            price = float(getattr(decision, 'price_hint', 0.0) or 0.0)
            if price <= 0:
                # intentar extraer del obs
                main_tf = self.tfs[0] if self.tfs else "1m"
                price = float(obs.get("tfs", {}).get(main_tf, {}).get("close", 0.0) or 0.0)
            # Leer flags de riesgo
            risk_common = getattr(self.env.cfg.risk, 'common', None)
            allow_fallback = True
            ttl_default = 180
            min_sl_pct = 1.0
            tp_r_multiple = 1.5
            atr_fb_enabled = True
            atr_min_mult = 1.2
            if risk_common is not None:
                allow_fallback = bool(getattr(risk_common, 'allow_open_without_levels_train', True))
                dl = getattr(risk_common, 'default_levels', None)
                if dl is not None:
                    ttl_default = int(getattr(dl, 'ttl_bars_default', 180))
                    min_sl_pct = float(getattr(dl, 'min_sl_pct', 1.0))
                    tp_r_multiple = float(getattr(dl, 'tp_r_multiple', 1.5))
                atr_fb = getattr(risk_common, 'atr_fallback', None)
                if atr_fb is not None:
                    atr_fb_enabled = bool(getattr(atr_fb, 'enabled', True))
                    atr_min_mult = float(getattr(atr_fb, 'min_sl_atr_mult', 1.2))

            # Si no se permite fallback, salir
            if not allow_fallback:
                return

            sl = getattr(decision, 'sl', None)
            tp = getattr(decision, 'tp', None)
            ttl = int(getattr(decision, 'ttl_bars', 0) or 0)
            side = int(getattr(decision, 'side', 0) or 0)
            if side == 0:
                return

            needs_levels = (sl is None or tp is None)
            needs_ttl = (ttl <= 0)
            if not needs_levels and not needs_ttl:
                return

            # Distancia mínima por %
            sl_dist_price = price * (min_sl_pct / 100.0)
            # ATR fallback a partir de features del obs
            if atr_fb_enabled:
                features = obs.get("features", {})
                atr_val = None
                for tf in self.tfs:
                    tf_feats = features.get(tf, {})
                    if "atr14" in tf_feats and float(tf_feats["atr14"]) > 0:
                        atr_val = float(tf_feats["atr14"])
                        break
                if atr_val is None or atr_val <= 0:
                    # último recurso: 1% del precio
                    atr_val = price * 0.01
                sl_dist_price = max(sl_dist_price, atr_val * atr_min_mult)

            if sl is None:
                decision.sl = price - sl_dist_price if side > 0 else price + sl_dist_price
            if tp is None:
                decision.tp = price + tp_r_multiple * sl_dist_price if side > 0 else price - tp_r_multiple * sl_dist_price
            if needs_ttl:
                decision.ttl_bars = ttl_default
        except Exception as e:
            # Silencioso: no bloquear por fallback
            print(f"⚠️ DEFAULT_LEVELS fallback error: {e}")
    
    def _get_atr_fallback_distance(self, current_price: float) -> Tuple[float, float]:
        """
        Calcula distancias de SL/TP usando ATR como fallback cuando no hay datos.
        Retorna (sl_distance, tp_distance) en términos de precio.
        """
        try:
            # Obtener ATR del timeframe principal
            main_tf = self.tfs[0] if self.tfs else "1m"
            obs = self.env.get_observation() if hasattr(self.env, 'get_observation') else {}
            tfs_data = obs.get("tfs", {})
            features = obs.get("features", {})
            
            # Buscar ATR en features
            atr_value = None
            for tf in self.tfs:
                tf_features = features.get(tf, {})
                if "atr14" in tf_features:
                    atr_value = float(tf_features["atr14"])
                    break
            
            if atr_value is None or atr_value <= 0:
                # Fallback: usar 1% del precio como ATR
                atr_value = current_price * 0.01
            
            # Distancias basadas en ATR
            sl_distance = atr_value * 2.0  # SL a 2 ATR
            tp_distance = atr_value * 3.0  # TP a 3 ATR (R:R = 1.5)
            
            return sl_distance, tp_distance
            
        except Exception as e:
            # Fallback final: usar porcentajes fijos
            sl_distance = current_price * 0.02  # 2%
            tp_distance = current_price * 0.03  # 3%
            return sl_distance, tp_distance

    def step(self, action):
        leverage = None
        trade_action = action
        if self._lev_spec is not None:
            # MultiDiscrete: [trade_action, leverage_idx]
            trade_action = int(action[0])
            lev_idx = int(action[1])
            leverage = self._lev_from_idx(lev_idx)
            
            # Validar límites de leverage
            mn, mx, st, n = self._lev_spec
            if lev_idx < 0 or lev_idx >= n:
                lev_idx = max(0, min(n-1, lev_idx))
                leverage = self._lev_from_idx(lev_idx)
                print(f"⚠️ Leverage index fuera de rango, ajustado a {lev_idx} → {leverage}x")

        # ← NUEVO: Curriculum learning - sugerir modificaciones basadas en estrategias exitosas
        if self.curriculum and random.random() < 0.05:  # 5% de las veces
            # Intentar cargar estrategias malas para evitarlas
            bad_strategies = []
            try:
                bad_strat_file = self.strategy_log_path.replace("_provisional.jsonl", "_bad_strategies.json")
                if Path(bad_strat_file).exists():
                    with open(bad_strat_file, 'r') as f:
                        bad_strategies = json.load(f)
            except:
                pass
                
            suggested_action = self.curriculum.suggest_action_modification(trade_action, {}, bad_strategies)
            if suggested_action is not None:
                trade_action = suggested_action
                print(f"[CURRICULUM] Acción modificada: {action} → {trade_action}")

        # ← NUEVO: Verificar deduplicación de decisiones y limpiar al cambiar de barra
        current_obs = self.env.get_observation() if hasattr(self.env, 'get_observation') else {}
        current_bar_time = None
        current_side = 0
        
        # Obtener información de la barra actual
        if current_obs and "tfs" in current_obs:
            main_tf = self.tfs[0] if self.tfs else "1m"
            current_bar = current_obs["tfs"].get(main_tf, {})
            current_bar_time = current_bar.get("ts")
            # Limpiar cache si cambió la barra
            if current_bar_time is not None and current_bar_time != self._last_bar_time:
                self._decision_cache.clear()
                self._last_bar_time = current_bar_time
            
            # Determinar side basado en la acción
            if trade_action == 3:  # force_long
                current_side = 1
            elif trade_action == 4:  # force_short
                current_side = -1
        
        # Verificar si es una decisión duplicada
        if current_bar_time and trade_action in [3, 4]:  # Solo para acciones de apertura
            if self._is_decision_duplicate(current_bar_time, current_side):
                # Bloquear trade duplicado, permitir en la siguiente barra
                trade_action = 2  # block
                print(f"🚫 Trade bloqueado por duplicado: ({current_bar_time}, {current_side})")

        # inyecta la acción y el leverage (si aplica)
        # ← FIX TEMPORAL: Si RL envía action=0 (dejar policy), forzar acción real
        if trade_action == 0:
            # Forzar acciones reales: 3=force_long, 4=force_short
            trade_action = random.choice([3, 4])
            print(f"FIX RL: action=0 → {trade_action} (force_long/force_short)")
        
        # ← NUEVO: Completar SL/TP/TTL cuando decide la policy o bypass
        # Si policy decide abrir sin niveles, los rellenamos aquí usando YAML
        if trade_action not in [1, 2]:  # acciones que no son solo cerrar/bloquear
            # Obtener la decisión propuesta por policy si no hay override (acción 0)
            if trade_action == 0:
                # La policy se evaluará dentro del env; aquí no conocemos la decisión.
                # Para asegurar niveles, dejamos que el env calcule y en el siguiente step
                # se aplicará de nuevo si faltaran.
                pass
            else:
                # Para bypass (3/4) ya hay defaults en _decision_from_action dentro del env.
                # Para robustez, dejamos fallback dinámico para SL/TP en el env si lo soporta.
                if hasattr(self.env, 'set_sl_tp_fallback'):
                    try:
                        current_price = current_obs.get("tfs", {}).get(main_tf, {}).get("close", 0.0)
                        if current_price > 0:
                            sl_distance, tp_distance = self._get_atr_fallback_distance(float(current_price))
                            self.env.set_sl_tp_fallback(sl_distance, tp_distance)
                    except Exception as e:
                        print(f"⚠️ Error aplicando fallback SL/TP: {e}")
        
        self.env.set_action_override(int(trade_action), leverage_override=leverage, leverage_index=lev_idx if self._lev_spec else None)

        obs, base_r, done, info = self.env.step()
        evs = info.get("events", [])
        if evs:
            self.strategy_log.append_many(evs)
        
        # ← NUEVO: Limpiar cache de decisiones al cerrar posición
        self._clear_decision_cache_on_close(evs, info)
        
        # ← NUEVO: Obtener información de milestones y runs vacíos
        balance_milestones = info.get("balance_milestones", 0)
        empty_run = self.env._empty_runs_count > 0 and not evs and not done
        
        shaped, parts = self.shaper.compute(
            obs, base_r, evs, empty_run, balance_milestones,
            initial_balance=self.env._init_cash,
            target_balance=self.env._target_cash
        )
        return self._flatten_obs(obs), float(shaped), bool(done), False, {"r_parts": parts, **info}

    def needs_learning_rate_reset(self) -> bool:
        """← NUEVO: Expone el método del entorno base para SubprocVecEnv"""
        return self.env.needs_learning_rate_reset()

    def reset_learning_rate_flag(self):
        """← NUEVO: Expone el método del entorno base para SubprocVecEnv"""
        self.env.reset_learning_rate_flag()
