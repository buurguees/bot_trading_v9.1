# base_env/base_env.py
"""
Orquestador del entorno base (Spot & Futuros) para RL.
- io/broker: ingest histórico/live
- tfs/alignment: alineación multi-TF coherente (bar_time TF base)
- features/pipeline + smc/detector: features técnicos + SMC
- analysis/hierarchical: dirección/confirm/exec → confidence
- policy/gating: confluencias, deduplicación, decisión
- risk/manager: sizing, exposición, apalancamiento, circuit breakers
- accounting/ledger: balances, fees, PnL realizado/no, MFE/MAE, DD
- events/domain: eventos para logs/dashboard
Config: se inyecta vía EnvConfig (derivado de YAMLs).
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Protocol, Tuple, Union, TypedDict, List
from decimal import Decimal
from dataclasses import dataclass
from enum import IntEnum
from contextlib import contextmanager
from pathlib import Path
import yaml

# ===== Sub-módulos del proyecto =====
from .config.models import EnvConfig
from .io.broker import DataBroker
from .tfs.alignment import MTFAligner
from .features.pipeline import FeaturePipeline
from .smc.detector import SMCDetector
from .analysis.hierarchical import HierarchicalAnalyzer, HierarchicalResult
from .policy.gating import PolicyEngine, Decision
from .policy.rules import sl_tp_from_atr
from .risk.manager import RiskManager, SizedDecision
from .accounting.ledger import PositionState, PortfolioState, Accounting
from .events.domain import EventBus
from .events.bus import SimpleEventBus
from .logging.run_logger import RunLogger
from .telemetry.reason_tracker import ReasonTracker, NoTradeReason
from .actions import LeverageTimeframeSelector, BankruptcyManager
from config.config_loader import load_env_config, validate_config_consistency
from .logging.error_handling import (
    TradingLogger,
    InvalidPositionStateError,
    DataIntegrityError,
    critical_operation,
    StateValidator,
    CircuitBreaker,
    CircuitBreakerConfig,
)

Number = Union[float, Decimal]


# ===== Tipos auxiliares =====
class FillResponse(TypedDict):
    """Respuesta estándar de ejecución (paper/live)."""
    success: bool
    side: str        # "LONG" | "SHORT"
    qty: float
    price: float
    notional: float
    fees: float
    ts: int
    order_id: str
    sl: Optional[float]
    tp: Optional[float]


class TradeAction(IntEnum):
    POLICY_DECIDE = 0
    CLOSE_ALL     = 1
    BLOCK_OPEN    = 2
    FORCE_LONG    = 3
    FORCE_SHORT   = 4


@dataclass(frozen=True)
class PositionSnapshot:
    side: int
    qty: float
    entry_price: float
    unrealized_pnl: float
    sl: Optional[float]
    tp: Optional[float]
    open_ts: Optional[int]
    bars_held: int
    mfe: float
    mae: float


class OMSAdapter(Protocol):
    def open(
        self, side: int, qty: Number, price_hint: Number,
        sl: Optional[Number], tp: Optional[Number]
    ) -> FillResponse: ...
    def close(self, qty: Number, price_hint: Number) -> FillResponse: ...


# ===== Entorno base =====
class BaseTradingEnv:
    """Entorno base canónico (idéntico en train/backtest/live; cambian adapters)."""

    # ----------------- INIT -----------------
    def __init__(
        self,
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

        # Verbosity básica (low/medium/high)
        self._verbosity = getattr(cfg, "verbosity", "low")
        self._debug_print_interval = {"low": 1000, "medium": 100, "high": 1}.get(self._verbosity, 1000)

        # Sub-sistemas
        self.mtf = MTFAligner(strict=cfg.pipeline.strict_alignment)
        self.features = FeaturePipeline(cfg.pipeline)
        self.smc = SMCDetector(cfg.pipeline)
        self.hier = HierarchicalAnalyzer(cfg.hierarchical)
        self.policy = PolicyEngine(cfg.hierarchical, cfg.risk, base_tf=cfg.tfs[0])
        self.risk = RiskManager(cfg.risk, cfg.symbol_meta)

        # Bancarrota / selección dinámica
        self.bankruptcy_manager = BankruptcyManager(self)
        self.leverage_tf_selector = LeverageTimeframeSelector(
            available_leverages=self._get_available_leverages(),
            available_timeframes=cfg.hierarchical.execute_tfs,
            symbol_config=(cfg.symbol_meta.model_dump() if hasattr(cfg.symbol_meta, "model_dump") else cfg.symbol_meta.__dict__),
        )

        # Estado
        self.pos = PositionState()
        self.portfolio = PortfolioState(market=cfg.market)
        self.accounting = Accounting(
            fees_cfg=(cfg.fees.model_dump() if hasattr(cfg.fees, "model_dump") else cfg.fees.__dict__),
            market=cfg.market,
        )
        self.events_bus: EventBus = SimpleEventBus()
        self._selected_leverage: float = getattr(cfg, "leverage", 3.0) or 3.0
        self._selected_timeframe: str = cfg.tfs[0]

        # Telemetría / logs
        self._reason_tracker = ReasonTracker()
        self.trading_logger = TradingLogger(f"BaseTradingEnv.{cfg.market}")
        self.oms_circuit_breaker = CircuitBreaker(
            "OMS",
            CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30.0, success_threshold=2),
        )
        self._init_cash = float(initial_cash)
        self._target_cash = float(target_cash)

        max_records = int((runs_log_cfg or {}).get("max_records", 2000))
        prune_strategy = (runs_log_cfg or {}).get("prune_strategy", "fifo")
        self._run_logger = RunLogger(cfg.symbol_meta.symbol, models_root=models_root, max_records=max_records, prune_strategy=prune_strategy)
        self._run_logger.cleanup_existing_runs()  # <- antes estaba en código inalcanzable

        # Control de ciclo
        self._done = False
        self._step_count = 0
        self._trades_executed = 0
        self._action_override: Optional[int] = None
        self._leverage_override: Optional[float] = None
        self._leverage_index_override: Optional[int] = None
        self._bankruptcy_detected = False
        self._current_leverage_used = 1.0
        self._equity_drift_without_position = 0

        # Antifreeze / LR reset
        self._empty_runs_count = 0
        self._max_empty_runs = 30
        self._learning_rate_reset_needed = False

        # Milestones
        self._last_balance_milestone = 0
        self._balance_milestones_this_run = 0
        self._milestones_verbose = False

        # Soft reset / cooldown
        self._soft_reset_count = 0
        self._cooldown_bars_remaining = 0
        self._current_segment_id = 0
        self._leverage_cap_active: Optional[float] = None

        # SL/TP fallback
        self._fallback_sl_dist: Optional[float] = None
        self._fallback_tp_dist: Optional[float] = None

        # Última observación materializada
        self._last_obs: Dict[str, Any] = {}

    @classmethod
    def from_yaml_dir(
        cls,
        config_dir: str,
        *,
        broker: DataBroker,
        oms: OMSAdapter,
        models_root: str = "models",
        antifreeze_enabled: bool = False,
    ) -> "BaseTradingEnv":
        """
        Crea el entorno totalmente autónomo leyendo .yaml de config/.
        Espera encontrar: settings.yaml, symbols.yaml, pipeline.yaml, hierarchical.yaml, risk.yaml, rewards.yaml (o rewards_optimized.yaml), train.yaml (si entrenas).
        """
        cfg_dir = Path(config_dir)
        if not cfg_dir.exists():
            raise FileNotFoundError(f"Config dir no existe: {cfg_dir}")

        # (1) Cargar y validar todo el stack de config
        env_cfg = load_env_config(cfg_dir)   # <- ensamblado tipado
        problems = validate_config_consistency(cfg_dir)
        if problems:
            # no abortamos entrenamiento, pero sí avisamos con precisión
            print("⚠️ [CONFIG] Duplicidades/solapes detectados:")
            for p in problems:
                print("   -", p)

        # (2) Instanciar el entorno base con cash target de train.yaml o settings.yaml
        initial_cash = getattr(env_cfg, "initial_balance", 1000.0)
        target_cash = getattr(env_cfg, "target_balance", 1_000_000.0)

        env = cls(
            cfg=env_cfg,
            broker=broker,
            oms=oms,
            initial_cash=initial_cash,
            target_cash=target_cash,
            models_root=models_root,
            antifreeze_enabled=antifreeze_enabled,
            runs_log_cfg={"max_records": 2000, "prune_strategy": "fifo"},
        )
        return env

    def _get_available_leverages(self) -> List[float]:
        lev_cfg = getattr(self.cfg.symbol_meta, "leverage", None)
        if isinstance(lev_cfg, dict):
            mn = float(lev_cfg.get("min", 1.0))
            mx = float(lev_cfg.get("max", 25.0))
            st = float(lev_cfg.get("step", 1.0))
            if mx < mn or st <= 0:
                mn, mx, st = 1.0, 25.0, 1.0
            vals = []
            cur = mn
            while cur <= mx + 1e-9:
                vals.append(round(cur, 10))
                cur += st
            return vals
        return [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0]

    # ----------------- API PÚBLICA -----------------
    def seed(self, seed: Optional[int]) -> None:
        """Compatibilidad con wrappers que llaman env.seed()."""
        # El broker y la policy deberían gestionar su propia semilla
        return None

    def reset(self) -> Dict[str, Any]:
        """Reset completo para nuevo episodio cronológico."""
        self.pos.reset()
        self.portfolio.reset(initial_cash=self._init_cash, target_cash=self._target_cash)
        self._done = False
        self._bankruptcy_detected = False
        self._trades_executed = 0
        self._step_count = 0
        self._soft_reset_count = 0
        self._cooldown_bars_remaining = 0
        self._current_segment_id = 0
        self._leverage_cap_active = None
        self._empty_runs_count = 0
        self._learning_rate_reset_needed = False
        self._equity_drift_without_position = 0
        self._balance_milestones_this_run = 0
        self._last_balance_milestone = 0

        # Cooldown configurable tras reset (si existe en YAML)
        cd = getattr(self.cfg, "startup_cooldown_steps", 0)
        self._cooldown_bars_remaining = int(cd or 0)

        # Reset cronológico del broker
        self.broker.reset_to_start()
        self._debug_print("🔄 RESET: inicio de histórico")

        obs = self._build_observation()
        if not validate_observation(obs):
            raise RuntimeError("Observación inválida en reset()")

        self._run_logger.start(
            market=self.cfg.market,
            initial_balance=self.portfolio.cash_quote,
            target_balance=self.portfolio.target_quote,
            initial_equity=self.portfolio.equity_quote,
            ts_start=int(obs["ts"]),
            segment_id=self._current_segment_id,
        )
        return obs

    def set_action_override(
        self,
        action: TradeAction | int | None,
        leverage_override: float | None = None,
        leverage_index: int | None = None,
    ):
        """Override de acción para el próximo step (usado por el wrapper RL)."""
        self._action_override = int(action) if action is not None else None
        if leverage_override is not None:
            self._leverage_override = float(leverage_override)
        if leverage_index is not None:
            self._leverage_index_override = int(leverage_index)

    def needs_learning_rate_reset(self) -> bool:
        return self._learning_rate_reset_needed

    def reset_learning_rate_flag(self) -> None:
        self._learning_rate_reset_needed = False
        self._empty_runs_count = 0

    def get_observation(self) -> dict:
        """Última observación consolidada (o {} si aún no hay)."""
        return dict(self._last_obs)

    def set_sl_tp_fallback(self, sl_dist: float, tp_dist: float) -> None:
        """Guardar fallback SL/TP (distancias en precio) para usar si la policy no trae niveles."""
        self._fallback_sl_dist = float(max(0.0, sl_dist))
        self._fallback_tp_dist = float(max(0.0, tp_dist))

    # ----------------- STEP -----------------
    @critical_operation("trading_step")
    def step(self, payload: Optional[Dict[str, Any]] = None):
        if self._done:
            self._increment_telemetry("DONE_EARLY")
            return self._build_observation(), 0.0, True, {"events": []}

        # Payload del wrapper (p.ej. allow_open por dedup)
        self._allow_open = bool((payload or {}).get("allow_open", True))

        # (0) Avanzar broker primero
        self.broker.next()

        # (1) Construir observación
        obs = self._build_observation()
        if not validate_observation(obs):
            self.trading_logger.system_error(
                DataIntegrityError("Observación inválida en step()"),
                {"obs_keys": list(obs.keys())},
            )
        ts_now = int(obs["ts"])
        exec_tf = self.policy.exec_tf

        # (1.1) Validar estado
        validation = StateValidator.validate_complete_state(self.portfolio, self.pos, self.cfg)
        if not validation.is_valid:
            self.trading_logger.system_error(
                InvalidPositionStateError("Estado inválido"), {"errors": validation.errors}
            )
        for w in validation.warnings or []:
            self.trading_logger._log_structured("WARNING", "STATE_WARNING", {"warning": w})

        # (2) Decisión
        action = self._action_override
        self._action_override = None  # consumir override
        # nota: el leverage_override se consumirá en sizing
        if action is None or action == TradeAction.POLICY_DECIDE:
            decision = self.policy.decide(obs)
            if not (decision.should_open or decision.should_close_all or getattr(decision, "should_close_partial", False)):
                self._increment_telemetry("NO_SIGNAL")

            # Completar niveles por defecto si faltan
            if decision.should_open and (decision.sl is None or decision.tp is None or int(getattr(decision, "ttl_bars", 0) or 0) <= 0):
                price = float(getattr(decision, "price_hint", 0.0) or obs["tfs"][exec_tf]["close"])
                feats = obs.get("features", {})
                atr_exec = feats.get(exec_tf, {}).get("atr14") or feats.get(self.policy.base_tf, {}).get("atr14")
                sl, tp = self._default_sl_tp(price, atr_exec, decision.side)
                ttl_default = int(getattr(self.cfg.risk.common, "default_levels", None).ttl_bars_default) if getattr(self.cfg.risk.common, "default_levels", None) else 180
                decision.sl, decision.tp = sl, tp
                decision.ttl_bars = ttl_default
                self._increment_telemetry("DEFAULT_LEVELS_APPLIED")
        else:
            decision = self._decision_from_action(int(action), obs)

        # (2.1) Cooldown tras soft-reset
        if self._cooldown_bars_remaining > 0:
            self._cooldown_bars_remaining -= 1
            if decision.should_open:
                self._increment_telemetry("COOLDOWN_AFTER_RESET")
                decision.should_open = False
            if self._cooldown_bars_remaining == 0:
                self._debug_print("✅ COOLDOWN FINALIZADO")

        # (2.2) Leverage cap (futuros)
        if self._leverage_cap_active is not None and self.cfg.mode.endswith("futures"):
            if self._leverage_override is not None and self._leverage_override > self._leverage_cap_active:
                self._leverage_override = self._leverage_cap_active
                self._debug_print(f"🔒 LEVERAGE CAP: {self._leverage_cap_active}x")

        # (3) Sizing
        if self.cfg.mode.endswith("futures"):
            lev = float(self._leverage_override) if self._leverage_override is not None else float(getattr(self.cfg, "leverage", 2.0) or 2.0)
            sized = self.risk.size_futures(self.portfolio, decision, lev, self.portfolio.equity_quote, self.events_bus, ts_now)
        else:
            sized = self.risk.apply(self.portfolio, self.pos, decision, obs, self.events_bus, ts_now)

        # (4) Ejecutar apertura si procede
        prev_equity = float(self.portfolio.equity_quote)
        if sized.should_open:
            price_exec = float(getattr(sized, "price_hint", 0.0) or 0.0)
            qty_exec = max(0.0, float(getattr(sized, "qty", 0.0) or 0.0))

            # Validación SL distancia > 0
            sl_candidate = getattr(sized, "sl", None)
            if sl_candidate is not None and price_exec > 0.0:
                if abs(price_exec - float(sl_candidate)) <= 0:
                    self._increment_telemetry("NO_SL_DISTANCE")
                    qty_exec = 0.0

            # Cash/notional clamps
            if price_exec <= 0.0:
                qty_exec = 0.0
            else:
                if self.cfg.market == "spot":
                    max_qty_cash = max(0.0, float(self.portfolio.cash_quote) / price_exec)
                    qty_exec = min(qty_exec, max_qty_cash)
                else:
                    notional_max = float(getattr(sized, "notional_max", 0.0) or 0.0)
                    if notional_max > 0.0:
                        qty_exec = min(qty_exec, notional_max / price_exec)

                # MinNotional (futuros)
                try:
                    min_notional_cfg = float(self.cfg.symbol_meta.filters.get("minNotional", 0.0))
                    lot_step = float(self.cfg.symbol_meta.filters.get("lotStep", 0.0) or 0.0)
                    train_force = bool(getattr(self.cfg.risk.common, "train_force_min_notional", True))
                except Exception:
                    min_notional_cfg, lot_step, train_force = 0.0, 0.0, True

                if self.cfg.mode.endswith("futures") and train_force and price_exec > 0.0 and min_notional_cfg > 0.0 and qty_exec > 0.0:
                    notional_now = qty_exec * price_exec
                    if notional_now < min_notional_cfg and lot_step > 0.0:
                        target_qty = (int((min_notional_cfg / price_exec) / lot_step) + 1) * lot_step
                        qty_exec = max(qty_exec, float(target_qty))

            # OPEN_ATTEMPT (respetando allow_open)
            if self._allow_open and qty_exec > 0.0:
                notional = qty_exec * price_exec
                leverage = getattr(sized, "leverage_used", 1.0) or 1.0
                sl_str = f"{getattr(sized, 'sl', None):.4f}" if getattr(sized, "sl", None) else "None"
                tp_str = f"{getattr(sized, 'tp', None):.4f}" if getattr(sized, "tp", None) else "None"
                self._debug_print(
                    f"📈 OPEN_ATTEMPT: ts={ts_now} side={sized.side} price={price_exec:.4f} qty={qty_exec:.6f} "
                    f"notional={notional:.2f} lev={leverage:.1f}x sl={sl_str} tp={tp_str}"
                )

                fill = self.oms_circuit_breaker.call(
                    self.oms.open,
                    ("LONG" if sized.side > 0 else "SHORT"),
                    _to_float(qty_exec),
                    _to_float(price_exec),
                    _to_float(getattr(sized, "sl", None)),
                    _to_float(getattr(sized, "tp", None)),
                )
                self.accounting.apply_open(fill, self.portfolio, self.pos, self.cfg)
                self._trades_executed += 1

                # Evento OPEN (enriquecido)
                feats_exec = obs.get("features", {}).get(exec_tf, {})
                used_tfs = {
                    "direction": getattr(self.cfg.hierarchical, "direction_tfs", []),
                    "confirm": getattr(self.cfg.hierarchical, "confirm_tfs", []),
                    "execute": getattr(self.cfg.hierarchical, "execute_tfs", []),
                }
                event = {
                    "ts": ts_now,
                    "kind": "OPEN",
                    "side": ("LONG" if sized.side > 0 else "SHORT"),
                    "qty": self.pos.qty,
                    "price": self.pos.entry_price,
                    "sl": self.pos.sl,
                    "tp": self.pos.tp,
                    "risk_pct": 0.0,
                    "analysis": obs.get("analysis", {}),
                    "indicators": list(feats_exec.keys()),
                    "used_tfs": used_tfs,
                }
                if self.cfg.market == "futures":
                    event.update({
                        "leverage_used": getattr(sized, "leverage_used", self.cfg.leverage),
                        "notional_effective": getattr(sized, "notional_effective", 0.0),
                        "notional_max": getattr(sized, "notional_max", 0.0),
                        "leverage_max": getattr(self.cfg, "leverage", 1.0),
                        "action_taken": action or 0,
                        "leverage_index": getattr(self, "_leverage_index_override", 0),
                    })
                self.events_bus.emit("OPEN", **event)
            else:
                # Bloqueo por deduplicación/pipeline
                self._increment_telemetry("RISK_BLOCKED" if decision.should_open else "POLICY_NO_OPEN")
                self._run_logger.add_reason("risk_manager_blocked" if decision.should_open else "policy_no_open")

        # (5) Mantenimiento SL/TP/TTL/Trailing (puede cerrar)
        auto_close = self.risk.maintenance(self.portfolio, self.pos, self.broker, self.events_bus, obs, exec_tf, ts_now)
        if auto_close is not None and (auto_close.should_close_all or auto_close.should_close_partial):
            self._do_close(auto_close, ts_now, exec_tf, obs)

        # (6) Unrealized PnL + equity mark-to-market
        if self.pos.side != 0 and self.pos.qty > 0:
            current_price = float(self.broker.get_price() or self.pos.entry_price or 0.0)
            self.pos.unrealized_pnl = (current_price - self.pos.entry_price) * self.pos.qty * (1 if self.pos.side > 0 else -1)
            self.pos.mfe = max(self.pos.mfe, self.pos.unrealized_pnl)
            self.pos.mae = min(self.pos.mae, self.pos.unrealized_pnl)
        else:
            self.pos.unrealized_pnl = 0.0

        # (7) Reward basado en equity delta (el shaping adicional lo hace el wrapper)
        equity_now = float(self.portfolio.cash_quote + self.pos.unrealized_pnl) if self.pos.side != 0 and self.pos.qty > 0 else float(self.portfolio.cash_quote)
        self.portfolio.equity_quote = equity_now
        denom = prev_equity if prev_equity > 0 else 1.0
        reward = (equity_now - prev_equity) / denom

        # (8) Bancarrota (si está habilitada en cfg)
        bankruptcy_enabled = bool(getattr(self.cfg.risk.common.bankruptcy, "enabled", False))
        if bankruptcy_enabled and not self._bankruptcy_detected:
            if self.risk.check_bankruptcy(self.portfolio, self._init_cash, self.events_bus, ts_now):
                return self._handle_bankruptcy(reward, ts_now, obs)

        # (9) Guard-rails: sin posición, equity == cash (con tolerancia)
        if self.pos.side == 0 or self.pos.qty == 0.0:
            equity = self.portfolio.equity_quote
            cash = self.portfolio.cash_quote
            drift = abs(equity - cash)
            max_repair_pct, abs_cap = 0.001, 1.0
            max_drift = max(max_repair_pct * max(cash, 1.0), abs_cap)
            if 1e-6 < drift <= max_drift:
                self._equity_drift_without_position += 1
                self.portfolio.equity_quote = float(cash)
                self.portfolio.used_margin = 0.0
                self.pos.unrealized_pnl = 0.0

        # (10) Telemetría y milestones
        self._current_leverage_used = self._calculate_leverage_used()
        self._step_count += 1
        self._run_logger.update_elapsed_steps(self._step_count)
        self._run_logger.update_trades_count(self._trades_executed)
        if self._step_count % 1000 == 0:
            self._print_telemetry_summary()

        if self.pos.side != 0 and self.pos.open_ts is not None:
            self.pos.bars_held += 1

        current_balance = self.portfolio.equity_quote
        baseline = max(0.0, self._init_cash)
        if current_balance > baseline:
            current_milestone = int((current_balance - baseline) // 100)
            if current_milestone > self._last_balance_milestone:
                new_m = current_milestone - self._last_balance_milestone
                self._balance_milestones_this_run += max(0, new_m)
                self._last_balance_milestone = current_milestone

        # (11) Siguiente obs
        next_obs = self._build_observation()
        if not validate_observation(next_obs):
            self._debug_print("[OBS-VALIDATION] next_obs inválida")

        # Procesar eventos (antes de drenar)
        self._process_specific_events()
        events = self.events_bus.drain()
        info = {
            "events": events,
            "balance_milestones": self._balance_milestones_this_run,
            "current_balance": current_balance,
            "equity_drift_without_position": int(self._equity_drift_without_position),
        }

        # (12) Fin de histórico
        done = self.broker.is_end_of_data()
        if done:
            self._increment_telemetry("END_OF_HISTORY")
            should_log_run = (
                self._bankruptcy_detected or
                self._trades_executed > 0 or
                abs(self.portfolio.equity_quote - self._init_cash) > 10.0
            )
            if should_log_run:
                self._run_logger.finish(
                    final_balance=self.portfolio.cash_quote,
                    final_equity=self.portfolio.equity_quote,
                    ts_end=int(next_obs["ts"]),
                )
                self._empty_runs_count = 0
                self._balance_milestones_this_run = 0
            else:
                self._empty_runs_count += 1
                self._balance_milestones_this_run = 0
                if self.antifreeze_enabled and self._empty_runs_count >= self._max_empty_runs:
                    self._learning_rate_reset_needed = True

        self._done = done
        return next_obs, float(reward), bool(done), info

    # ----------------- Internos -----------------
    def _calculate_leverage_used(self) -> float:
        if self.cfg.market == "spot":
            return 1.0
        selected = float(getattr(self, "_selected_leverage", 3.0) or 3.0)
        lev_cfg = getattr(self.cfg.symbol_meta, "leverage", None)
        if lev_cfg is not None:
            lev_min = float(getattr(lev_cfg, "min", 1.0))
            lev_max = float(getattr(lev_cfg, "max", 25.0))
            return max(lev_min, min(lev_max, selected))
        return selected

    def _decision_from_action(self, action: int, obs: dict) -> Decision:
        """Convierte acción RL → Decision, con selección dinámica de TF/lev."""
        market_conditions = self.leverage_tf_selector.calculate_market_conditions(obs)
        conf = market_conditions.get("confidence", 0.5)
        ltfa = self.leverage_tf_selector.select_leverage_timeframe(action=action, market_conditions=market_conditions, confidence=conf)

        selected_tf = ltfa.timeframe
        self._selected_timeframe = selected_tf
        self._selected_leverage = float(ltfa.leverage)

        price = float(obs["tfs"][selected_tf]["close"])
        feats = obs.get("features", {})
        atr_exec = feats.get(selected_tf, {}).get("atr14") or feats.get(self.policy.base_tf, {}).get("atr14")

        if action == TradeAction.CLOSE_ALL:
            return Decision(should_close_all=True, price_hint=price)
        if action == TradeAction.BLOCK_OPEN:
            return Decision(should_open=False, price_hint=price)
        if action == TradeAction.FORCE_LONG:
            sl, tp = self._default_sl_tp(price, atr_exec, side=+1)
            return Decision(should_open=True, side=+1, price_hint=price, sl=sl, tp=tp, ttl_bars=self._ttl_default(), trailing=True)
        if action == TradeAction.FORCE_SHORT:
            sl, tp = self._default_sl_tp(price, atr_exec, side=-1)
            return Decision(should_open=True, side=-1, price_hint=price, sl=sl, tp=tp, ttl_bars=self._ttl_default(), trailing=True)
        return Decision(should_open=False, price_hint=price)

    def _ttl_default(self) -> int:
        dl = getattr(self.cfg.risk.common, "default_levels", None)
        return int(getattr(dl, "ttl_bars_default", 180)) if dl else 180

    def _default_sl_tp(self, price: float, atr: Optional[float], side: int) -> Tuple[Optional[float], Optional[float]]:
        dl = getattr(self.cfg.risk.common, "default_levels", None)
        use_atr = bool(getattr(dl, "use_atr", True)) if dl else True
        sl_atr_mult = float(getattr(dl, "sl_atr_mult", 1.0)) if dl else 1.0
        min_sl_pct = float(getattr(dl, "min_sl_pct", 1.0)) if dl else 1.0
        tp_r_multiple = float(getattr(dl, "tp_r_multiple", 1.5)) if dl else 1.5

        if use_atr and atr is not None and atr > 0:
            return sl_tp_from_atr(price, atr, side=side, k_sl=sl_atr_mult, k_tp=tp_r_multiple)
        else:
            sl_mult = min_sl_pct / 100.0
            tp_mult = sl_mult * tp_r_multiple
            if side > 0:
                return price * (1 - sl_mult), price * (1 + tp_mult)
            else:
                return price * (1 + sl_mult), price * (1 - tp_mult)

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
            "analysis": (analysis.model_dump() if hasattr(analysis, "model_dump") else dict(analysis.__dict__)),
            "position": self.pos.to_dict(),
            "portfolio": self.portfolio.to_dict(),
            "mode": self.cfg.mode,
        }

        # Señal de la policy (hint para el wrapper)
        dec = self.policy.decide(obs)
        obs["policy_signal"] = {
            "should_open": bool(dec.should_open),
            "side_hint": int(dec.side if dec.should_open else 0),
            "confidence": float(getattr(analysis, "confidence", 0.0) or 0.0),
            "should_close": bool(dec.should_close_all),
            "price_hint": float(getattr(dec, "price_hint", 0.0) or 0.0),
        }

        # Persistir última obs para get_observation()
        self._last_obs = obs
        return obs

    def _do_close(self, close_decision: SizedDecision, ts_now: int, exec_tf: str, obs: Dict[str, Any]) -> None:
        """Cierre parcial/total según decisión del RiskManager/maintenance."""
        qty_close = close_decision.close_qty if close_decision.should_close_partial else (self.pos.qty or 0.0)
        if not qty_close or qty_close <= 0 or self.pos.side == 0:
            return

        entry = float(self.pos.entry_price)
        qty_now = float(self.pos.qty)
        sl_now = self.pos.sl
        risk_pct = 0.0
        risk_val = 0.0
        if sl_now is not None and entry > 0:
            risk_val = abs(entry - float(sl_now)) * qty_now
            risk_pct = abs(entry - float(sl_now)) / entry * 100.0

        fill = self.oms.close(_to_float(qty_close), _to_float(close_decision.price_hint))
        realized = self.accounting.apply_close(fill, self.portfolio, self.pos, self.cfg)

        exit_price = float(fill["price"])
        notional = entry * qty_now if entry > 0 else 0.0
        roi_pct = (realized / notional) * 100.0 if notional > 0 else 0.0
        r_multiple = (realized / risk_val) if risk_val > 0 else 0.0

        self.events_bus.emit(
            "CLOSE",
            ts=ts_now,
            qty=qty_close,
            price=exit_price,
            realized_pnl=realized,
            entry_price=entry,
            entry_qty=qty_now,
            roi_pct=roi_pct,
            r_multiple=r_multiple,
            risk_pct=risk_pct,
            reason=("PARTIAL" if close_decision.should_close_partial else "ALL"),
            open_ts=self.pos.open_ts,
            duration_ms=ts_now - (self.pos.open_ts or ts_now),
            bars_held=self.pos.bars_held,
            exec_tf=exec_tf,
            leverage_used=self._current_leverage_used,
        )

        # Registrar en RunLogger
        self._run_logger.add_trade_record(
            entry_price=entry,
            exit_price=exit_price,
            qty=qty_close,
            side=int(self.pos.side),
            realized_pnl=realized,
            bars_held=self.pos.bars_held,
            open_ts=self.pos.open_ts,
            close_ts=ts_now,
            sl=sl_now,
            tp=self.pos.tp,
            roi_pct=roi_pct,
            r_multiple=r_multiple,
            risk_pct=risk_pct,
        )

    def _handle_bankruptcy(self, reward: float, ts_now: int, obs: Dict[str, Any]):
        return self.bankruptcy_manager.handle_bankruptcy(reward, ts_now, obs)

    # --------- Telemetría / utilidades ---------
    def _increment_telemetry(self, reason: str):
        mapping = {
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
            "INVALID_ACTION": NoTradeReason.INVALID_ACTION,
            "END_OF_HISTORY": NoTradeReason.END_OF_HISTORY,
        }
        self._reason_tracker.increment(mapping.get(reason, NoTradeReason.INVALID_ACTION))
        self._run_logger.add_reason(reason)

    def _print_telemetry_summary(self):
        self._debug_print(f"\n📊 TELEMETRÍA - Step {self._step_count:,}")
        self._debug_print(f"   Trades: {self._trades_executed}")
        self._debug_print(f"   Equity: {self.portfolio.equity_quote:.2f}")
        self._debug_print(f"   Pos: side={self.pos.side} qty={self.pos.qty:.6f}")
        self._reason_tracker.print_summary("RAZONES DE NO-TRADE")

    def _process_specific_events(self):
        events = getattr(self.events_bus, "_buffer", [])[:]
        for e in events:
            if e.kind == "LOW_EQUITY":
                self._increment_telemetry("LOW_EQUITY")
            elif e.kind == "NO_SL_DISTANCE":
                self._increment_telemetry("NO_SL_DISTANCE")
            elif e.kind == "MIN_NOTIONAL_BLOCKED":
                self._increment_telemetry("MIN_NOTIONAL_BLOCKED")

    def _debug_print(self, message: str, step_interval: Optional[int] = None) -> None:
        if step_interval is None:
            step_interval = self._debug_print_interval
        if self._verbosity == "high":
            print(message)
        elif self._verbosity == "medium" and self._step_count % step_interval == 0:
            print(message)


# ===== Utils =====
def _to_float(x: Number) -> Optional[float]:
    return float(x) if x is not None else None


def validate_observation(obs: Dict[str, Any]) -> bool:
    required = {"ts", "tfs", "features", "position", "portfolio"}
    if not required.issubset(obs):
        missing = required - set(obs)
        print(f"[OBS-VALIDATION] Faltan claves: {missing}")
        return False
    if not isinstance(obs["ts"], int) or obs["ts"] <= 0:
        print(f"[OBS-VALIDATION] Timestamp inválido: {obs['ts']}")
        return False
    return True


@contextmanager
def trade_transaction(env: "BaseTradingEnv"):
    """Transacción de trading con rollback automático."""
    snap = PositionSnapshot(
        side=int(env.pos.side),
        qty=float(env.pos.qty),
        entry_price=float(env.pos.entry_price or 0.0),
        unrealized_pnl=float(env.pos.unrealized_pnl or 0.0),
        sl=float(env.pos.sl) if env.pos.sl is not None else None,
        tp=float(env.pos.tp) if env.pos.tp is not None else None,
        open_ts=env.pos.open_ts,
        bars_held=int(env.pos.bars_held or 0),
        mfe=float(env.pos.mfe or 0.0),
        mae=float(env.pos.mae or 0.0),
    )
    try:
        yield
    except Exception as e:
        print(f"[TRADE-TX] Error, restaurando snapshot: {e}")
        env.pos.side = snap.side
        env.pos.qty = snap.qty
        env.pos.entry_price = snap.entry_price
        env.pos.unrealized_pnl = snap.unrealized_pnl
        env.pos.sl = snap.sl
        env.pos.tp = snap.tp
        env.pos.open_ts = snap.open_ts
        env.pos.bars_held = snap.bars_held
        env.pos.mfe = snap.mfe
        env.pos.mae = snap.mae
        raise
