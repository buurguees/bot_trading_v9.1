
"""
base_env/base_env.py
Descripción: Orquestador del entorno base (Spot & Futuros). Conecta:
- io/broker: ingestión de datos histórico/live
- tfs/alignment: alineación multi-timeframe coherente (bar_time TF base)
- features/pipeline + smc/detector: features técnicos + SMC
- analysis/hierarchical: dirección/confirmación/ejecución → confidence
- policy/gating: confluencias, deduplicación, decisión
- risk/manager: sizing, exposición, apalancamiento ≤3x, circuit breakers
- accounting/ledger: balances, fees, PnL realizado/no, MFE/MAE, DD
- events/domain: eventos de dominio para logs/dashboard
Config: lee parámetros desde config/*.yaml (no aquí directamente, sino a través de config/models.py).
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Protocol, Tuple

from .config.models import EnvConfig
from .io.broker import DataBroker
from .tfs.alignment import MTFAligner
from .features.pipeline import FeaturePipeline
from .smc.detector import SMCDetector
from .analysis.hierarchical import HierarchicalAnalyzer, HierarchicalResult
from .policy.gating import PolicyEngine, Decision
from .policy.rules import sl_tp_from_atr
from .risk.manager import RiskManager
from .accounting.ledger import PositionState, PortfolioState, Accounting
from .events.domain import EventBus, DomainEvent
from .events.bus import SimpleEventBus
from .risk.manager import RiskManager, SizedDecision
from .policy.gating import PolicyEngine
from .risk.manager import RiskManager, SizedDecision
from .accounting.ledger import Accounting, PortfolioState, PositionState
from .logging.run_logger import RunLogger
from .telemetry.reason_tracker import ReasonTracker, NoTradeReason
from .actions import LeverageTimeframeSelector, BankruptcyManager

class OMSAdapter(Protocol):
    """Interfaz mínima para ejecución (Sim/Paper/Live). Slippage se aplica fuera del core."""
    def open(self, side: int, qty: float, price_hint: float, sl: Optional[float], tp: Optional[float]) -> Dict[str, Any]: ...
    def close(self, qty: float, price_hint: float) -> Dict[str, Any]: ...


class BaseTradingEnv:
    """Entorno base canónico (idéntico en train/backtest/live; cambian adapters)."""

    def __init__(self,
        cfg: EnvConfig,
        broker: DataBroker,
        oms: OMSAdapter,
        *,
        initial_cash: float = 1000.0,
        target_cash: float = 1_000_000.0,
        models_root: str = "models",
        antifreeze_enabled: bool = False,
        runs_log_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = cfg
        self.broker = broker
        self.oms = oms
        self.antifreeze_enabled = bool(antifreeze_enabled)

        # Sub-sistemas
        self.mtf = MTFAligner(strict=cfg.pipeline.strict_alignment)
        self.features = FeaturePipeline(cfg.pipeline)
        self.smc = SMCDetector(cfg.pipeline)
        self.hier = HierarchicalAnalyzer(cfg.hierarchical)
        self.policy = PolicyEngine(cfg.hierarchical, base_tf=cfg.tfs[0])
        self.risk = RiskManager(cfg.risk, cfg.symbol_meta)
        
        # NUEVO: Selector dinámico de leverage y timeframe
        available_leverages = self._get_available_leverages()
        available_timeframes = cfg.hierarchical.execute_tfs
        self.leverage_tf_selector = LeverageTimeframeSelector(
            available_leverages=available_leverages,
            available_timeframes=available_timeframes,
            symbol_config=cfg.symbol_meta.model_dump() if hasattr(cfg.symbol_meta, "model_dump") else cfg.symbol_meta.__dict__
        )
        
        # Estado
        self.pos = PositionState()
        self.portfolio = PortfolioState(market=cfg.market)
        self.accounting = Accounting(fees_cfg=cfg.fees.model_dump() if hasattr(cfg.fees,"model_dump") else cfg.fees.__dict__, market=cfg.market)
        self.events_bus = SimpleEventBus()
        
        # NUEVO: Variables para leverage y timeframe seleccionados
        self._selected_leverage = 3.0  # Default
        self._selected_timeframe = cfg.tfs[0]  # Default

        # NUEVO: Sistema unificado de telemetría (mover aquí para evitar errores)
        from .telemetry.reason_tracker import ReasonTracker
        self._reason_tracker = ReasonTracker()

        # NUEVO: balances y logger
        self._init_cash = float(initial_cash)
        self._target_cash = float(target_cash)
        
        # NUEVO: RunLogger (mover aquí para evitar errores)
        max_records = int((runs_log_cfg or {}).get("max_records", 2000))
        prune_strategy = (runs_log_cfg or {}).get("prune_strategy", "fifo")
        
        self._run_logger = RunLogger(
            cfg.symbol_meta.symbol, 
            models_root=models_root,
            max_records=max_records,
            prune_strategy=prune_strategy
        )

    def _get_available_leverages(self) -> list[float]:
        """Obtiene la lista de leverages disponibles basado en la configuración del símbolo"""
        leverage_config = getattr(self.cfg.symbol_meta, 'leverage', {})
        if isinstance(leverage_config, dict):
            min_lev = leverage_config.get('min', 2.0)
            max_lev = leverage_config.get('max', 25.0)
            step = leverage_config.get('step', 1.0)
            
            # Generar lista de leverages disponibles
            leverages = []
            current = min_lev
            while current <= max_lev:
                leverages.append(current)
                current += step
            return leverages
        else:
            # Fallback si no hay configuración
            return [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0]
        
        # ← NUEVO: Limpiar archivo existente aplicando límite configurable
        self._run_logger.cleanup_existing_runs()

        # Control
        self._done = False
        self._action_override = None  # acción externa (wrapper RL)
        self._leverage_override = None  # leverage externo (wrapper RL, solo futures)
        self._leverage_index = None  # índice de leverage en action space
        self._bankruptcy_detected = False  # flag de quiebra detectada
        self._trades_executed = 0  # contador de trades ejecutados
        # ← NUEVO: Tracking de leverage usado
        self._current_leverage_used = 1.0  # leverage del trade actual
        # ← NUEVO: Telemetría de fugas de equity sin posición
        self._equity_drift_without_position = 0
        
        # ← NUEVO: Sistema unificado de telemetría (ya inicializado arriba)
        
        # ← NUEVO: Control de soft reset
        self._soft_reset_count = 0  # contador de resets en el mismo run
        self._cooldown_bars_remaining = 0  # barras restantes de cooldown
        self._current_segment_id = 0  # ID del segmento actual
        self._leverage_cap_active = None  # leverage máximo temporal (None = sin límite)

        # ← NUEVO: Gestor de bancarrota externo (configurable)
        self.bankruptcy_manager = BankruptcyManager(self)
        
        # ← NUEVO: Control de runs vacíos para learning rate reset
        self._empty_runs_count = 0  # contador de runs vacíos consecutivos
        self._max_empty_runs = 30  # ← NUEVO: Umbral más bajo para activación más temprana
        
        # ← NUEVO: Control de milestones de balance
        self._last_balance_milestone = 0  # último milestone alcanzado (en centenas)
        self._balance_milestones_this_run = 0  # milestones alcanzados en este run  # ← NUEVO: Sincronizado con el callback
        self._learning_rate_reset_needed = False  # flag para reset
        # ← NUEVO: Verbosidad para logs de milestones (por defecto desactivado)
        self._milestones_verbose = False
        
        # ← NUEVO: Control de verbosity para prints de debug
        self._verbosity = getattr(cfg, 'verbosity', 'low')
        self._debug_print_interval = {"low": 1000, "medium": 100, "high": 1}.get(self._verbosity, 1000)

    # ------------- API pública -------------
    def reset(self):
        """Reset completo del entorno para nuevo episodio cronológico"""
        # Reset estado interno
        self.pos.reset()
        self.portfolio.reset(initial_cash=self._init_cash, target_cash=self._target_cash)
        self._done = False
        self._bankruptcy_detected = False  # reset flag de quiebra
        self._trades_executed = 0  # reset contador de trades
        self._step_count = 0  # reset contador de steps
        
        # ← NUEVO: Reset de control de soft reset
        self._soft_reset_count = 0
        self._cooldown_bars_remaining = 0
        self._current_segment_id = 0
        self._leverage_cap_active = None
        
        # ← NUEVO: Reset del contador de runs vacíos
        self._empty_runs_count = 0
        self._learning_rate_reset_needed = False
        
        # ← NUEVO: Reset de telemetría
        self._equity_drift_without_position = 0
        self._reason_tracker.reset()
        self._drift_logged_this_episode = False  # Reset flag de drift logging
        
        # ← NUEVO: Reset de milestones de balance
        self._balance_milestones_this_run = 0
        self._last_balance_milestone = 0
        
        # ← NUEVO: Reset cronológico del broker al inicio del histórico
        self.broker.reset_to_start()
        self._debug_print(f"🔄 RESET CRONOLÓGICO: Reiniciando al inicio del histórico")
        
        # inicio de run
        obs = self._build_observation()
        self._run_logger.start(
            market=self.cfg.market,
            initial_balance=self.portfolio.cash_quote,
            target_balance=self.portfolio.target_quote,
            initial_equity=self.portfolio.equity_quote,
            ts_start=int(obs["ts"]),
            segment_id=self._current_segment_id
        )
        
        # ← NUEVO: Contadores de telemetría para "por qué no opero"
        self._telemetry = {
            "COOLDOWN": 0,
            "WARMUP": 0,
            "NO_SIGNAL": 0,
            "RISK_BLOCKED": 0,
            "CIRCUIT_BREAKER": 0,
            "EXPOSURE_LIMIT": 0,
            "BROKER_EMPTY": 0,
            "DONE_EARLY": 0,
            "POLICY_NO_OPEN": 0,
            "SIZING_FAILED": 0,
            "BANKRUPTCY_RESTART": 0,
            "MIN_NOTIONAL_BLOCKED": 0,  # ← NUEVO: Bloqueado por minNotional
            "LOW_EQUITY": 0,            # ← NUEVO: Equity muy bajo
            "NO_SL_DISTANCE": 0,        # ← NUEVO: SL muy cerca
            "CONFIDENCE_TOO_LOW": 0,    # ← NUEVO: Confidence insuficiente
            "DEFAULT_LEVELS_APPLIED": 0, # ← NUEVO: SL/TP por defecto aplicados
        }
        return obs

    def set_action_override(self, action: int | None, leverage_override: float | None = None, leverage_index: int | None = None):
        """Permite a un wrapper externo (RL) inyectar una acción para el próximo step.
           Acciones: None=sin override, 0=dejar policy, 1=close_all, 3=force_long, 4=force_short, 2=block_open.
           Leverage: opcional, solo para futuros (sobrescribe cfg.leverage).
           Leverage Index: índice en el action space para futuros."""
        self._action_override = action
        self._leverage_override = leverage_override
        self._leverage_index = leverage_index

    def needs_learning_rate_reset(self) -> bool:
        """← NUEVO: Verifica si se necesita reset del learning rate por runs vacíos"""
        return self._learning_rate_reset_needed

    def reset_learning_rate_flag(self):
        """← NUEVO: Resetea el flag de learning rate reset"""
        self._learning_rate_reset_needed = False
        self._empty_runs_count = 0

    def _increment_telemetry(self, reason: str):
        """← NUEVO: Incrementa contador de telemetría usando sistema unificado"""
        # Mapear razones del sistema anterior al nuevo sistema
        reason_mapping = {
            "RISK_BLOCKED": NoTradeReason.RISK_BLOCKED,
            "NO_SIGNAL": NoTradeReason.NO_SIGNAL,
            "MIN_NOTIONAL_BLOCKED": NoTradeReason.MIN_NOTIONAL_BLOCKED,
            "LOW_EQUITY": NoTradeReason.LOW_EQUITY,
            "NO_SL_DISTANCE": NoTradeReason.NO_SL_DISTANCE,
            "POLICY_NO_OPEN": NoTradeReason.POLICY_NO_OPEN,
            "COOLDOWN_AFTER_RESET": NoTradeReason.COOLDOWN_AFTER_RESET,
            "BROKER_EMPTY": NoTradeReason.BROKER_EMPTY,
            "DONE_EARLY": NoTradeReason.DONE_EARLY,
            "BANKRUPTCY_RESTART": NoTradeReason.BANKRUPTCY_RESTART,
            "SHORTS_DISABLED": NoTradeReason.SHORTS_DISABLED,
            "LEVERAGE_CAP": NoTradeReason.LEVERAGE_CAP,
            "MARGIN_INSUFFICIENT": NoTradeReason.MARGIN_INSUFFICIENT,
            "POSITION_ALREADY_OPEN": NoTradeReason.POSITION_ALREADY_OPEN,
            "INVALID_ACTION": NoTradeReason.INVALID_ACTION
        }
        
        no_trade_reason = reason_mapping.get(reason, NoTradeReason.INVALID_ACTION)
        self._reason_tracker.increment(no_trade_reason)
        
        # También añadir al logger de runs
        self._run_logger.add_reason(reason)

    def _print_telemetry_summary(self):
        """← NUEVO: Imprime resumen de telemetría usando sistema unificado"""
        self._debug_print(f"\n📊 TELEMETRÍA - Step {self._step_count:,}")
        self._debug_print(f"   Trades ejecutados: {self._trades_executed}")
        self._debug_print(f"   Equity actual: ${self.portfolio.equity_quote:.2f}")
        self._debug_print(f"   Posición: {self.pos.side} (qty: {self.pos.qty:.6f})")
        
        # Usar el sistema unificado de telemetría
        self._reason_tracker.print_summary("RAZONES DE NO-TRADE")

    def _process_specific_events(self):
        """Procesa eventos específicos del RiskManager para incrementar telemetría"""
        # Obtener eventos sin drenarlos
        events = self.events_bus._buffer.copy()
        
        for event in events:
            if event.kind == "LOW_EQUITY":
                self._increment_telemetry("LOW_EQUITY")
            elif event.kind == "NO_SL_DISTANCE":
                self._increment_telemetry("NO_SL_DISTANCE")
            elif event.kind == "MIN_NOTIONAL_BLOCKED":
                self._increment_telemetry("MIN_NOTIONAL_BLOCKED")

    def step(self):
        if self._done:
            self._increment_telemetry("DONE_EARLY")
            return self._build_observation(), 0.0, True, {"events": []}

        # 1) Construir obs antes de decidir
        obs = self._build_observation()
        ts_now = int(obs["ts"])
        exec_tf = self.policy.exec_tf

        # 2) DECISIÓN de apertura/cierre por policy
        action = getattr(self, "_action_override", None)
        self._action_override = None
        # Limpiar leverage override después de usarlo
        self._leverage_override = None
        
        if action is None or action == 0:
            # Usar policy jerárquica normal
            decision = self.policy.decide(obs)
            # ← NUEVO: Telemetría para decisiones de policy
            if not decision.should_open and not decision.should_close_all and not decision.should_close_partial:
                self._increment_telemetry("NO_SIGNAL")
            # ← NUEVO: Completar SL/TP/TTL cuando la policy decide abrir sin niveles
            if decision.should_open and (decision.sl is None or decision.tp is None or int(getattr(decision, 'ttl_bars', 0) or 0) <= 0):
                price = float(getattr(decision, 'price_hint', 0.0) or 0.0) or float(obs["tfs"][self.policy.exec_tf]["close"]) 
                feats = obs.get("features", {})
                atr_exec = feats.get(self.policy.exec_tf, {}).get("atr14") or feats.get(self.policy.base_tf, {}).get("atr14")
                sl, tp = self._get_default_sl_tp(price, atr_exec, decision.side)
                # TTL por defecto desde YAML
                ttl_default = 180
                default_levels = getattr(self.cfg.risk.common, 'default_levels', None)
                if default_levels is not None:
                    ttl_default = int(getattr(default_levels, 'ttl_bars_default', 180))
                decision.sl = sl
                decision.tp = tp
                if int(getattr(decision, 'ttl_bars', 0) or 0) <= 0:
                    decision.ttl_bars = ttl_default
                self._increment_telemetry("DEFAULT_LEVELS_APPLIED")
                print(f"🔧 DEFAULT_LEVELS_APPLIED(policy): SL={float(sl) if sl else None}, TP={float(tp) if tp else None}, TTL={int(decision.ttl_bars)}")
        else:
            # BYPASS de policy: acción RL forzada
            decision = self._decision_from_action(action, obs)
            
            # ← NUEVO: SANEADOR CRÍTICO antes del BYPASS POLICY
            print(f"🔍 DECISION CHECK: should_open={decision.should_open}, side={decision.side}")
            if decision.should_open and decision.side != 0:
                print(f"🔧 ENTRANDO AL SANEADOR: should_open={decision.should_open}, side={decision.side}")
                from train_env.utils.sanitizers import sanitize_open_levels, OpenDecision
                
                # Convertir Decision a OpenDecision para el saneador
                raw_decision = OpenDecision(
                    should_open=decision.should_open,
                    side=int(decision.side),
                    price_hint=decision.price_hint,
                    sl=getattr(decision, "sl", None),
                    tp=getattr(decision, "tp", None),
                    ttl_bars=int(getattr(decision, "ttl_bars", 0) or 0),
                    trailing=bool(getattr(decision, "trailing", False)),
                )
                
                price = float(decision.price_hint)
                # Convertir RiskCommon a diccionario para el saneador con valores por defecto seguros
                try:
                    default_levels = getattr(self.cfg.risk.common, "default_levels", None)
                    atr_fallback = getattr(self.cfg.risk.common, "atr_fallback", None)
                    allow_fallback = getattr(self.cfg.risk.common, "allow_open_without_levels_train", True)
                except AttributeError:
                    # Valores por defecto si no existe la configuración
                    default_levels = None
                    atr_fallback = None
                    allow_fallback = True
                
                risk_cfg = {
                    "common": {
                        "default_levels": {
                            "min_sl_pct": getattr(default_levels, "min_sl_pct", 1.0) if default_levels else 1.0,
                            "tp_r_multiple": getattr(default_levels, "tp_r_multiple", 1.5) if default_levels else 1.5,
                            "ttl_bars_default": getattr(default_levels, "ttl_bars_default", 180) if default_levels else 180,
                        },
                        "allow_open_without_levels_train": allow_fallback,
                        "atr_fallback": {
                            "enabled": getattr(atr_fallback, "enabled", True) if atr_fallback else True,
                            "tf": getattr(atr_fallback, "tf", "1m") if atr_fallback else "1m",
                            "lookback": getattr(atr_fallback, "lookback", 14) if atr_fallback else 14,
                            "min_sl_atr_mult": getattr(atr_fallback, "min_sl_atr_mult", 1.2) if atr_fallback else 1.2,
                        }
                    }
                }
                
                # ⬇️ SANEADOR CRÍTICO: antes del BYPASS
                sanitized_decision = sanitize_open_levels(
                    decision=raw_decision,
                    price=price,
                    dv=self.broker,  # DataView para ATR
                    risk_cfg=risk_cfg,
                    is_train=True
                )
                
                # Actualizar la decisión original con los valores saneados
                decision.sl = sanitized_decision.sl
                decision.tp = sanitized_decision.tp
                decision.ttl_bars = sanitized_decision.ttl_bars
                decision.trailing = sanitized_decision.trailing
                # ← CRÍTICO: Actualizar price_hint si era None o 0.0
                if decision.price_hint is None or decision.price_hint <= 0.0:
                    decision.price_hint = price
                
                self._increment_telemetry("DEFAULT_LEVELS_APPLIED")
                print(f"🔧 SANEADOR APLICADO: SL={sanitized_decision.sl:.4f}, TP={sanitized_decision.tp:.4f}, TTL={sanitized_decision.ttl_bars}")
            
            print(f"🚀 BYPASS POLICY: Acción RL {action} → {decision}")
        # Aplicar sizing según el tipo de mercado
        # ← NUEVO: Verificar cooldown después de soft reset
        if self._cooldown_bars_remaining > 0:
            self._cooldown_bars_remaining -= 1
            if decision.should_open:
                self._increment_telemetry("COOLDOWN_AFTER_RESET")
                decision.should_open = False  # Bloquear apertura durante cooldown
            if self._cooldown_bars_remaining == 0:
                print(f"OK COOLDOWN FINALIZADO: Se permite trading nuevamente")
        
        # ← NUEVO: Aplicar leverage cap si está activo
        if self._leverage_cap_active is not None and self.cfg.mode.endswith("futures"):
            if self._leverage_override is not None and self._leverage_override > self._leverage_cap_active:
                self._leverage_override = self._leverage_cap_active
                print(f"🔒 LEVERAGE CAP APLICADO: Limitado a {self._leverage_cap_active}x")

        if self.cfg.mode.endswith("futures"):
            # Usar leverage override si está disponible, sino cfg.leverage, sino 2.0 por defecto
            lev = float(self._leverage_override) if self._leverage_override is not None else self.cfg.leverage
            sized = self.risk.size_futures(self.portfolio, decision, lev, self.portfolio.equity_quote, self.events_bus, ts_now)
        else:
            sized = self.risk.apply(self.portfolio, self.pos, decision, obs, self.events_bus, ts_now)

        reward = 0.0
        # ← NUEVO: Snapshot de equity previo para reward mark-to-market
        prev_equity = float(self.portfolio.equity_quote)

        # Inicializar risk_pct por defecto
        risk_pct = 0.0

        # 3) Ejecutar apertura
        print(f"🔍 RISK MANAGER RESULT: should_open={sized.should_open}, side={sized.side}, qty={getattr(sized, 'qty', 0.0)}, price_hint={getattr(sized, 'price_hint', 0.0)}")
        if sized.should_open:
            # ← NUEVO: Sanity clamp de tamaño antes de ejecutar
            price_exec = float(getattr(sized, "price_hint", 0.0) or 0.0)
            qty_exec = max(0.0, float(getattr(sized, "qty", 0.0) or 0.0))
            # ← NUEVO: Validar distancia al SL
            sl_candidate = getattr(sized, "sl", None)
            if sl_candidate is not None and price_exec > 0.0:
                sl_dist = abs(price_exec - float(sl_candidate))
                if sl_dist <= 0:
                    self._increment_telemetry("NO_SL_DISTANCE")
                    qty_exec = 0.0
            if price_exec <= 0.0:
                qty_exec = 0.0
            else:
                if self.cfg.market == "spot":
                    # No permitir notional mayor al cash disponible
                    max_qty_cash = max(0.0, float(self.portfolio.cash_quote) / price_exec)
                    qty_exec = min(qty_exec, max_qty_cash)
                else:
                    # Futuros: respetar tope de notional efectivo reportado por risk
                    notional_max = float(getattr(sized, "notional_max", 0.0) or 0.0)
                    if notional_max > 0.0:
                        max_qty_by_notional = notional_max / price_exec
                        qty_exec = min(qty_exec, max_qty_by_notional)

                # ← NUEVO: Forzar minNotional en TRAIN si procede
                try:
                    min_notional_cfg = float(self.cfg.symbol_meta.filters.get("minNotional", 0.0))
                    lot_step = float(self.cfg.symbol_meta.filters.get("lotStep", 0.0) or 0.0)
                except Exception:
                    min_notional_cfg = 0.0
                    lot_step = 0.0
                try:
                    train_force = bool(getattr(self.cfg.risk.common, 'train_force_min_notional', True))
                except Exception:
                    train_force = True
                if self.cfg.mode.endswith("futures") and train_force and price_exec > 0.0 and min_notional_cfg > 0.0 and qty_exec > 0.0:
                    notional_now = qty_exec * price_exec
                    if notional_now < min_notional_cfg and lot_step > 0.0:
                        target_qty = (int((min_notional_cfg / price_exec) / lot_step) + 1) * lot_step
                        qty_exec = max(qty_exec, float(target_qty))

            # ← NUEVO: Trazador de ejecución detallado
            notional = qty_exec * price_exec
            min_notional = float(self.cfg.symbol_meta.filters.get("minNotional", 5.0))
            leverage = getattr(sized, "leverage_used", 1.0)
            sl_str = f"{sized.sl:.4f}" if sized.sl else "None"
            tp_str = f"{sized.tp:.4f}" if sized.tp else "None"
            print(f"📈 OPEN_ATTEMPT: ts={ts_now}, side={sized.side}, price={price_exec:.4f}, qty={qty_exec:.6f}, "
                  f"notional={notional:.2f}, minNotional={min_notional:.2f}, leverage={leverage:.1f}x, "
                  f"sl={sl_str}, tp={tp_str}")

            fill = self.oms.open("LONG" if sized.side > 0 else "SHORT", qty_exec, price_exec, sized.sl, sized.tp)
            self.accounting.apply_open(fill, self.portfolio, self.pos, self.cfg)
            self._trades_executed += 1
            print(f"✅ FILL EJECUTADO: {self._trades_executed} trades totales")
            # TTL de la policy
        else:
            # ← NUEVO: Loggear razón de no abrir posición
            if decision.should_open:
                self._increment_telemetry("RISK_BLOCKED")
                self._run_logger.add_reason("risk_manager_blocked")
            else:
                self._increment_telemetry("POLICY_NO_OPEN")
                self._run_logger.add_reason("policy_no_open")
            
            # risk%
            risk_pct = 0.0
            if self.pos.sl is not None and self.pos.entry_price > 0:
                dist = abs(self.pos.entry_price - float(self.pos.sl))
                risk_pct = (dist / self.pos.entry_price) * 100.0

            # incluir análisis/TFs/indicadores básicos
        feats_exec = obs.get("features", {}).get(exec_tf, {})
        used_tfs = {
            "direction": getattr(self.cfg.hierarchical, "direction_tfs", []),
            "confirm": getattr(self.cfg.hierarchical, "confirm_tfs", []),
            "execute": getattr(self.cfg.hierarchical, "execute_tfs", []),
        }
        
        # ← NUEVO: información completa para futuros
        event_data = {
            "ts": ts_now,
            "side": ("LONG" if sized.side > 0 else "SHORT"),
            "qty": self.pos.qty,
            "price": self.pos.entry_price,
            "sl": self.pos.sl,
            "tp": self.pos.tp,
            "risk_pct": risk_pct,
            "analysis": obs.get("analysis", {}),
            "indicators": list(feats_exec.keys()),
            "used_tfs": used_tfs
        }
        
        # Añadir información de leverage si es futuros
        if self.cfg.market == "futures":
            event_data.update({
                "leverage_used": getattr(sized, "leverage_used", self.cfg.leverage),
                "notional_effective": getattr(sized, "notional_effective", 0.0),
                "notional_max": getattr(sized, "notional_max", 0.0),
                "leverage_max": self.cfg.leverage,
                "action_taken": getattr(self, "_action_override", 0),
                "leverage_index": getattr(self, "_leverage_index", 0)
            })
        
        self.events_bus.emit("OPEN", **event_data)

        # 4) Ejecutar cierres explícitos (policy/risk)
        if sized.should_close_all or sized.should_close_partial:
            qty_close = sized.close_qty if sized.should_close_partial else (self.pos.qty or 0.0)
            # ← NUEVO: No ejecutar cierre si no hay posición
            if qty_close and qty_close > 0 and self.pos.side != 0:
                # --- métricas previas al reset ---
                entry = float(self.pos.entry_price)
                qty_now = float(self.pos.qty)
                side_now = int(self.pos.side)
                sl_now = self.pos.sl
                # riesgo inicial (si existía SL al abrir)
                risk_pct = 0.0
                risk_val = 0.0
                if sl_now is not None and entry > 0:
                    risk_val = abs(entry - float(sl_now)) * qty_now
                    risk_pct = abs(entry - float(sl_now)) / entry * 100.0

                fill = self.oms.close(qty_close, sized.price_hint)
                realized = self.accounting.apply_close(fill, self.portfolio, self.pos, self.cfg)

                # métricas de cierre
                exit_price = float(fill["price"])
                notional = entry * qty_now if entry > 0 else 0.0
                roi_pct = (realized / notional) * 100.0 if notional > 0 else 0.0
                r_multiple = (realized / risk_val) if risk_val > 0 else 0.0

                self.events_bus.emit(
                    "CLOSE", ts=ts_now, qty=qty_close, price=exit_price,
                    realized_pnl=realized, entry_price=entry, entry_qty=qty_now,
                    roi_pct=roi_pct, r_multiple=r_multiple, risk_pct=risk_pct,
                    reason=("PARTIAL" if sized.should_close_partial else "ALL"),
                    # ← NUEVO: información temporal completa
                    open_ts=self.pos.open_ts,
                    duration_ms=ts_now - (self.pos.open_ts or ts_now),
                    bars_held=self.pos.bars_held,
                    exec_tf=exec_tf,
                    # ← NUEVO: leverage usado en el trade
                    leverage_used=self._current_leverage_used
                )
                
                # ← NUEVO: Registrar trade para métricas profesionales
                self._run_logger.add_trade_record(
                    entry_price=entry,
                    exit_price=exit_price,
                    qty=qty_close,
                    side=side_now,
                    realized_pnl=realized,
                    bars_held=self.pos.bars_held,
                    open_ts=self.pos.open_ts,
                    close_ts=ts_now,
                    sl=sl_now,
                    tp=self.pos.tp,
                    roi_pct=roi_pct,
                    r_multiple=r_multiple,
                    risk_pct=risk_pct
                )
                
                # NUEVO: Calcular reward avanzado usando el sistema de rewards por duración
                try:
                    from train_env.reward_shaper import RewardShaper
                    reward_shaper = RewardShaper("config/rewards.yaml")
                    
                    # Determinar la razón del cierre
                    close_reason = "manual"  # Por defecto
                    if auto_close:
                        if hasattr(auto_close, 'close_reason'):
                            close_reason = auto_close.close_reason
                        elif "SL_HIT" in [e.get("kind") for e in self.events_bus._buffer]:
                            close_reason = "sl_hit"
                        elif "TP_HIT" in [e.get("kind") for e in self.events_bus._buffer]:
                            close_reason = "tp_hit"
                        elif "TTL_HIT" in [e.get("kind") for e in self.events_bus._buffer]:
                            close_reason = "ttl_hit"
                    
                    # Calcular reward avanzado
                    advanced_reward = reward_shaper.compute_advanced_trade_reward(
                        realized_pnl=realized,
                        notional=abs(entry * qty_close),
                        leverage_used=self._current_leverage_used,
                        r_multiple=r_multiple,
                        close_reason=close_reason,
                        timeframe_used=getattr(self, '_selected_timeframe', '1m'),
                        bars_held=self.pos.bars_held
                    )
                    
                    reward += advanced_reward
                    print(f"🎯 REWARD AVANZADO: PnL={realized:.4f}, bars_held={self.pos.bars_held}, advanced_reward={advanced_reward:.4f}")
                    
                except Exception as e:
                    print(f"⚠️ Error calculando reward avanzado: {e}")
                    # Fallback al sistema anterior
                    reward += float(realized)

        # 5) Mantenimiento SL/TP/TTL/Trailing (puede devolver cierre total)
        auto_close = self.risk.maintenance(self.portfolio, self.pos, self.broker, self.events_bus, obs, exec_tf, ts_now)
        if auto_close is not None and (auto_close.should_close_all or auto_close.should_close_partial):
            qty_close = auto_close.close_qty if auto_close.should_close_partial else (self.pos.qty or 0.0)
            if qty_close and qty_close > 0:
                entry = float(self.pos.entry_price)
                qty_now = float(self.pos.qty)
                side_now = int(self.pos.side)
                sl_now = self.pos.sl
                risk_pct = 0.0
                risk_val = 0.0
                if sl_now is not None and entry > 0:
                    risk_val = abs(entry - float(sl_now)) * qty_now
                    risk_pct = abs(entry - float(sl_now)) / entry * 100.0

                fill = self.oms.close(qty_close, auto_close.price_hint)
                realized = self.accounting.apply_close(fill, self.portfolio, self.pos, self.cfg)

                exit_price = float(fill["price"])
                notional = entry * qty_now if entry > 0 else 0.0
                roi_pct = (realized / notional) * 100.0 if notional > 0 else 0.0
                r_multiple = (realized / risk_val) if risk_val > 0 else 0.0

                self.events_bus.emit(
                    "CLOSE", ts=ts_now, qty=qty_close, price=exit_price,
                    realized_pnl=realized, entry_price=entry, entry_qty=qty_now,
                    roi_pct=roi_pct, r_multiple=r_multiple, risk_pct=risk_pct,
                    reason=("AUTO_PARTIAL" if auto_close.should_close_partial else "AUTO_ALL"),
                    # ← NUEVO: información temporal completa
                    open_ts=self.pos.open_ts,
                    duration_ms=ts_now - (self.pos.open_ts or ts_now),
                    bars_held=self.pos.bars_held,
                    exec_tf=exec_tf,
                    # ← NUEVO: leverage usado en el trade
                    leverage_used=self._current_leverage_used
                )
                
                # ← NUEVO: Registrar trade automático para métricas profesionales
                self._run_logger.add_trade_record(
                    entry_price=entry,
                    exit_price=exit_price,
                    qty=qty_close,
                    side=side_now,
                    realized_pnl=realized,
                    bars_held=self.pos.bars_held,
                    open_ts=self.pos.open_ts,
                    close_ts=ts_now,
                    sl=sl_now,
                    tp=self.pos.tp,
                    roi_pct=roi_pct,
                    r_multiple=r_multiple,
                    risk_pct=risk_pct
                )
                
                # NUEVO: Calcular reward avanzado para cierre automático
                try:
                    from train_env.reward_shaper import RewardShaper
                    reward_shaper = RewardShaper("config/rewards.yaml")
                    
                    # Determinar la razón del cierre automático
                    close_reason = "manual"  # Por defecto
                    if "SL_HIT" in [e.get("kind") for e in self.events_bus._buffer]:
                        close_reason = "sl_hit"
                    elif "TP_HIT" in [e.get("kind") for e in self.events_bus._buffer]:
                        close_reason = "tp_hit"
                    elif "TTL_HIT" in [e.get("kind") for e in self.events_bus._buffer]:
                        close_reason = "ttl_hit"
                    
                    # Calcular reward avanzado
                    advanced_reward = reward_shaper.compute_advanced_trade_reward(
                        realized_pnl=realized,
                        notional=abs(entry * qty_close),
                        leverage_used=self._current_leverage_used,
                        r_multiple=r_multiple,
                        close_reason=close_reason,
                        timeframe_used=getattr(self, '_selected_timeframe', '1m'),
                        bars_held=self.pos.bars_held
                    )
                    
                    reward += advanced_reward
                    print(f"🎯 REWARD AVANZADO (AUTO): PnL={realized:.4f}, bars_held={self.pos.bars_held}, close_reason={close_reason}, advanced_reward={advanced_reward:.4f}")
                    
                except Exception as e:
                    print(f"⚠️ Error calculando reward avanzado (auto): {e}")
                    # Fallback al sistema anterior
                    reward += float(realized)

        # 6) ACTUALIZAR PRECIO Y CALCULAR UNREALIZED PnL (UNA SOLA VEZ)
        # El precio se actualiza automáticamente en broker.next()
        
        # Calcular unrealized PnL solo si hay posición abierta
        if self.pos.side != 0 and self.pos.qty > 0:
            current_price = float(self.broker.get_price() or self.pos.entry_price or 0.0)
            self.pos.unrealized_pnl = (current_price - self.pos.entry_price) * self.pos.qty * (1 if self.pos.side > 0 else -1)
            # Actualizar MFE/MAE
            self.pos.mfe = max(self.pos.mfe, self.pos.unrealized_pnl)
            self.pos.mae = min(self.pos.mae, self.pos.unrealized_pnl)
        else:
            # Sin posición: unrealized PnL = 0
            self.pos.unrealized_pnl = 0.0

        # 7) CALCULAR REWARD BASADO EN EQUITY (antes de aplicar acciones)
        # Equity = balance + unrealized PnL
        if self.pos.side != 0 and self.pos.qty > 0:
            equity_now = float(self.portfolio.cash_quote + self.pos.unrealized_pnl)
        else:
            equity_now = float(self.portfolio.cash_quote)
        
        # Actualizar equity en portfolio
        self.portfolio.equity_quote = equity_now
        
        # Reward = variación de equity
        denom = prev_equity if prev_equity > 0 else 1.0
        reward += (equity_now - prev_equity) / denom

        # 8) VERIFICAR QUIEBRA (solo si está habilitado)
        bankruptcy_enabled = bool(self.cfg.risk.common.bankruptcy.enabled)
        if bankruptcy_enabled and not self._bankruptcy_detected:
            bankruptcy_occurred = self.risk.check_bankruptcy(
                self.portfolio, 
                self._init_cash, 
                self.events_bus, 
                ts_now
            )
            if bankruptcy_occurred:
                return self._handle_bankruptcy(reward, ts_now, obs)

        # 9) INVARIANTES Y GUARD-RAILS
        # Sin posición: equity debe igualar balance exactamente
        if self.pos.side == 0 or self.pos.qty == 0.0:
            drift = abs(self.portfolio.equity_quote - self.portfolio.cash_quote)
            if drift > 1e-6:
                # Solo loggear una vez por episodio para evitar spam
                if not hasattr(self, '_drift_logged_this_episode'):
                    print(f"🔧 CORRIGIENDO DRIFT: Sin posición, equity={self.portfolio.equity_quote:.8f} → cash={self.portfolio.cash_quote:.8f} (drift: {drift:.8f})")
                    self._drift_logged_this_episode = True
                self._equity_drift_without_position += 1
                self.portfolio.equity_quote = float(self.portfolio.cash_quote)
                self.portfolio.used_margin = 0.0
                self.pos.unrealized_pnl = 0.0
        
        # Validar valores NaN/Inf
        if not (self.portfolio.equity_quote == self.portfolio.equity_quote) or self.portfolio.equity_quote in (float('inf'), float('-inf')):
            self.portfolio.equity_quote = float(self.portfolio.cash_quote)
            reward = 0.0
        self.broker.next()
        
        # ← NUEVO: Calcular leverage usado para el trade actual
        self._current_leverage_used = self._calculate_leverage_used()
        
        # ← NUEVO: Actualizar contadores de actividad
        self._step_count += 1
        self._run_logger.update_elapsed_steps(self._step_count)
        self._run_logger.update_trades_count(self._trades_executed)
        
        # ← NUEVO: Resumen de telemetría cada 1000 steps
        if self._step_count % 1000 == 0:
            self._print_telemetry_summary()
        
        # ← NUEVO: incrementar barras que estuvo abierta la posición
        if self.pos.side != 0 and self.pos.open_ts is not None:
            self.pos.bars_held += 1
        
        # ← NUEVO: Calcular milestones de balance (cada +100 que supere 0)
        current_balance = self.portfolio.equity_quote
        # Evitar spam de milestones al iniciar (considerar init_cash como baseline)
        baseline = max(0.0, self._init_cash)
        if current_balance > baseline:
            current_milestone = int((current_balance - baseline) // 100)  # Milestone desde baseline
            if current_milestone > self._last_balance_milestone:
                new_milestones = current_milestone - self._last_balance_milestone
                self._balance_milestones_this_run += max(0, new_milestones)
                self._last_balance_milestone = current_milestone
                # Mensaje menos frecuente y más claro
                if new_milestones > 0 and self._milestones_verbose:
                    print(f"🎯 Milestone: equity {baseline + current_milestone * 100:.0f} USDT (+{new_milestones})")
        else:
            # Si el balance cae por debajo del baseline, resetear milestones
            self._last_balance_milestone = 0

        # 8) Siguiente observación y eventos
        next_obs = self._build_observation()
        
        # ← NUEVO: Procesar eventos específicos antes de drenar
        self._process_specific_events()
        
        events = self.events_bus.drain()
        info = {
            "events": events,
            "balance_milestones": self._balance_milestones_this_run,
            "current_balance": current_balance
        }
        # ← NUEVO: métrica de fugas de equity
        info["equity_drift_without_position"] = int(self._equity_drift_without_position)

        # ← NUEVO: Detección explícita del final del histórico
        done = self.broker.is_end_of_data()
        if done:
            self._increment_telemetry("END_OF_HISTORY")
            print(f"🏁 FIN DEL HISTÓRICO: Completada pasada cronológica completa")
            
            # ← NUEVO: Solo loguear si hubo actividad real (trades, quiebra, o progreso significativo)
            should_log_run = (
                self._bankruptcy_detected or  # Quiebra
                self._trades_executed > 0 or  # Hubo trades
                abs(self.portfolio.equity_quote - self._init_cash) > 10.0  # Cambio significativo (>$10)
            )
            
            if should_log_run:
                # log de run completo
                self._run_logger.finish(
                    final_balance=self.portfolio.cash_quote,
                    final_equity=self.portfolio.equity_quote,
                    ts_end=int(next_obs["ts"]),
                    run_result="END_OF_HISTORY"
                )
                print(f"✅ RUN COMPLETADO: {self._trades_executed} trades, equity final: {self.portfolio.equity_quote:.2f}")
                # ← NUEVO: Reset contador de runs vacíos al tener actividad
                self._empty_runs_count = 0
                # ← NUEVO: Reset milestones de balance al final del run
                self._balance_milestones_this_run = 0
            else:
                # ← NUEVO: Incrementar contador de runs vacíos
                self._empty_runs_count += 1
                print(f"🔄 Run sin actividad real - no logueando ({self._empty_runs_count}/{self._max_empty_runs})")
                # ← NUEVO: Reset milestones de balance en runs vacíos
                self._balance_milestones_this_run = 0
                
                # ← NUEVO: Activar reset de learning rate si se alcanza el umbral (solo si antifreeze está habilitado)
                if self.antifreeze_enabled and self._empty_runs_count >= self._max_empty_runs:
                    self._learning_rate_reset_needed = True
                    print(f"🚨 {self._max_empty_runs} runs vacíos consecutivos - ACTIVANDO LEARNING RATE RESET")
                    print(f"   El agente necesita explorar nuevas estrategias")
        
        self._done = done
        return next_obs, reward, done, info

    # ------------- Internos -------------
    def _calculate_leverage_used(self) -> float:
        """Calcula el leverage usado para el trade actual"""
        if self.cfg.market == "spot":
            # En spot, siempre leverage = 1.0
            return 1.0
        else:
            # NUEVO: Usar el leverage seleccionado dinámicamente
            selected_leverage = getattr(self, '_selected_leverage', 3.0)
            
            # Clamp al rango del símbolo si existe configuración
            if self.cfg.symbol_meta.leverage is not None:
                leverage_min = getattr(self.cfg.symbol_meta.leverage, 'min', 1.0)
                leverage_max = getattr(self.cfg.symbol_meta.leverage, 'max', 25.0)
                leverage_used = max(leverage_min, min(leverage_max, selected_leverage))
            else:
                leverage_used = selected_leverage
            
            return leverage_used

    def _decision_from_action(self, action: int, obs: dict) -> Decision:
        """Convierte acción RL en Decision, con SL/TP por defecto si faltan."""
        # NUEVO: Calcular condiciones del mercado para selección dinámica
        market_conditions = self.leverage_tf_selector.calculate_market_conditions(obs)
        confidence = market_conditions.get("confidence", 0.5)
        
        # NUEVO: Seleccionar leverage y timeframe dinámicamente
        leverage_tf_action = self.leverage_tf_selector.select_leverage_timeframe(
            action=action,
            market_conditions=market_conditions,
            confidence=confidence
        )
        
        # Usar el timeframe seleccionado para obtener precio y ATR
        selected_tf = leverage_tf_action.timeframe
        price = float(obs["tfs"][selected_tf]["close"])
        feats = obs.get("features", {})
        atr_exec = feats.get(selected_tf, {}).get("atr14") or feats.get(self.policy.base_tf, {}).get("atr14")
        
        # Almacenar leverage seleccionado para uso posterior
        self._selected_leverage = leverage_tf_action.leverage
        self._selected_timeframe = selected_tf
        
        if action == 1:
            return Decision(should_close_all=True, price_hint=price)
        if action == 2:
            return Decision(should_open=False, price_hint=price)  # bloquear aperturas
        if action == 3:
            # Force LONG con SL/TP por defecto
            sl, tp = self._get_default_sl_tp(price, atr_exec, side=+1)
            return Decision(
                should_open=True, 
                side=+1, 
                price_hint=price, 
                sl=sl, 
                tp=tp, 
                ttl_bars=getattr(self.cfg.risk.common.default_levels, 'ttl_bars_default', 180), 
                trailing=True
            )
        if action == 4:
            # Force SHORT con SL/TP por defecto
            sl, tp = self._get_default_sl_tp(price, atr_exec, side=-1)
            return Decision(
                should_open=True, 
                side=-1, 
                price_hint=price, 
                sl=sl, 
                tp=tp, 
                ttl_bars=getattr(self.cfg.risk.common.default_levels, 'ttl_bars_default', 180), 
                trailing=True
            )
        # default: sin override → deja a la policy
        return Decision(should_open=False, price_hint=price)

    def _get_default_sl_tp(self, price: float, atr: Optional[float], side: int) -> Tuple[Optional[float], Optional[float]]:
        """Calcula SL/TP por defecto usando configuración YAML (ATR o fallback a porcentajes)."""
        # Obtener configuración de niveles por defecto desde risk.yaml
        default_levels = getattr(self.cfg.risk.common, 'default_levels', None)
        if default_levels is None:
            # Valores por defecto si no hay configuración
            use_atr = True
            atr_period = 14
            sl_atr_mult = 1.0
            min_sl_pct = getattr(self.cfg.risk.common.default_levels, 'min_sl_pct', 1.0)
            tp_r_multiple = getattr(self.cfg.risk.common.default_levels, 'tp_r_multiple', 1.5)
        else:
            # DefaultLevelsConfig es un @dataclass, acceder a atributos directamente
            use_atr = getattr(default_levels, 'use_atr', True)
            atr_period = getattr(default_levels, 'atr_period', 14)
            sl_atr_mult = getattr(default_levels, 'sl_atr_mult', 1.0)
            min_sl_pct = getattr(default_levels, 'min_sl_pct', 1.0)
            tp_r_multiple = getattr(default_levels, 'tp_r_multiple', 1.5)
        
        if use_atr and atr is not None and atr > 0:
            # Usar ATR con configuración YAML
            return sl_tp_from_atr(price, atr, side=side, k_sl=sl_atr_mult, k_tp=tp_r_multiple)
        else:
            # Fallback a porcentajes desde YAML
            sl_mult = min_sl_pct / 100.0  # Convertir % a decimal
            tp_mult = sl_mult * tp_r_multiple  # TP como múltiplo del riesgo
            if side > 0:  # Long
                sl = price * (1 - sl_mult)
                tp = price * (1 + tp_mult)
            else:  # Short
                sl = price * (1 + sl_mult)
                tp = price * (1 - tp_mult)
            return sl, tp

    def _build_observation(self) -> Dict[str, Any]:
        aligned = self.mtf.align(self.broker, required_tfs=self.cfg.tfs)
        feats = self.features.compute(aligned)
        smc = self.smc.detect(aligned, feats)
        analysis: HierarchicalResult = self.hier.analyze(feats, smc)
        obs = {
            "ts": self.broker.now_ts(),
            "tfs": aligned,
            "features": feats,
            "smc": smc,
            "analysis": analysis.model_dump() if hasattr(analysis, "model_dump") else dict(analysis.__dict__),
            "position": self.pos.to_dict(),
            "portfolio": self.portfolio.to_dict(),
            "mode": self.cfg.mode,
        }
        return obs

    def _handle_bankruptcy(self, reward: float, ts_now: int, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Delegar manejo de bancarrota al BankruptcyManager."""
        return self.bankruptcy_manager.handle_bankruptcy(reward, ts_now, obs)

    def _handle_soft_reset(self, penalty_reward: float, ts_now: int, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Delegado: la lógica vive en BankruptcyManager._soft_reset."""
        return self.bankruptcy_manager._soft_reset(penalty_reward, ts_now, obs)

    def _close_position_force(self):
        """Fuerza el cierre de la posición actual sin pasar por el OMS."""
        if self.pos.side == 0:
            return
        
        # Simular fill de cierre
        close_fill = {
            "side": -self.pos.side,  # lado opuesto
            "qty": self.pos.qty,
            "price": float(self.broker.get_price()),
            "fees": 0.0,
            "sl": None,
            "tp": None
        }
        
        # Aplicar cierre directamente en contabilidad
        self.accounting.apply_close(close_fill, self.portfolio, self.pos, self.cfg)
        self._trades_executed += 1

    def get_observation(self) -> dict:
        """Devuelve el último obs consolidado si lo tienes; si no, algo mínimo útil"""
        return getattr(self, "_last_obs", {}) or {}

    def set_sl_tp_fallback(self, sl_dist: float, tp_dist: float) -> None:
        """Guarda fallback para usar si la policy no trae niveles"""
        self._fallback_sl_dist = float(max(0.0, sl_dist))
        self._fallback_tp_dist = float(max(0.0, tp_dist))

    def _debug_print(self, message: str, step_interval: int = None) -> None:
        """Imprime mensaje de debug solo si la verbosity lo permite"""
        if step_interval is None:
            step_interval = self._debug_print_interval
        
        if self._verbosity == "high" or (self._verbosity == "medium" and hasattr(self, '_step_count') and self._step_count % step_interval == 0):
            print(message)
