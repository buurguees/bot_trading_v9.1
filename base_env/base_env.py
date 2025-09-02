
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

class OMSAdapter(Protocol):
    """Interfaz mínima para ejecución (Sim/Paper/Live). Slippage se aplica fuera del core."""
    def open(self, side: int, qty: float, price_hint: float, sl: Optional[float], tp: Optional[float]) -> Dict[str, Any]: ...
    def close(self, qty: float, price_hint: float) -> Dict[str, Any]: ...


class BaseTradingEnv:
    """Entorno base canónico (idéntico en train/backtest/live; cambian adapters)."""

    def __init__(self, cfg: EnvConfig, broker: DataBroker, oms: OMSAdapter, initial_cash: float = 10000.0, target_cash: float = 1_000_000.0, models_root: str = "models") -> None:
        self.cfg = cfg
        self.broker = broker
        self.oms = oms

        # Sub-sistemas
        self.mtf = MTFAligner(strict=cfg.pipeline.strict_alignment)
        self.features = FeaturePipeline(cfg.pipeline)
        self.smc = SMCDetector(cfg.pipeline)
        self.hier = HierarchicalAnalyzer(cfg.hierarchical)
        self.policy = PolicyEngine(cfg.hierarchical, base_tf=cfg.tfs[0])
        self.risk = RiskManager(cfg.risk, cfg.symbol_meta)
        
        # Estado
        self.pos = PositionState()
        self.portfolio = PortfolioState(market=cfg.market)
        self.accounting = Accounting(fees_cfg=cfg.fees.model_dump() if hasattr(cfg.fees,"model_dump") else cfg.fees.__dict__, market=cfg.market)
        self.events_bus = SimpleEventBus()

        # NUEVO: balances y logger
        self._init_cash = float(initial_cash)
        self._target_cash = float(target_cash)

        # NUEVO: RunLogger en models/{symbol}
        self._run_logger = RunLogger(cfg.symbol_meta.symbol, models_root=models_root)
        
        # ← NUEVO: Limpiar archivo existente aplicando límite de 400 runs
        self._run_logger.cleanup_existing_runs()

        # Control
        self._done = False
        self._action_override = None  # acción externa (wrapper RL)
        self._leverage_override = None  # leverage externo (wrapper RL, solo futures)
        self._leverage_index = None  # índice de leverage en action space
        self._bankruptcy_detected = False  # flag de quiebra detectada
        self._trades_executed = 0  # contador de trades ejecutados

    # ------------- API pública -------------
    def reset(self):
        self.pos.reset()
        self.portfolio.reset(initial_cash=self._init_cash, target_cash=self._target_cash)
        self._done = False
        self._bankruptcy_detected = False  # reset flag de quiebra
        self._trades_executed = 0  # reset contador de trades
        # inicio de run
        obs = self._build_observation()
        self._run_logger.start(
            market=self.cfg.market,
            initial_balance=self.portfolio.cash_quote,
            target_balance=self.portfolio.target_quote,
            initial_equity=self.portfolio.equity_quote,
            ts_start=int(obs["ts"])
        )
        return obs

    def set_action_override(self, action: int | None, leverage_override: float | None = None, leverage_index: int | None = None):
        """Permite a un wrapper externo (RL) inyectar una acción para el próximo step.
           Acciones: None=sin override, 0=dejar policy, 1=close_all, 3=force_long, 4=force_short, 2=block_open.
           Leverage: opcional, solo para futuros (sobrescribe cfg.leverage).
           Leverage Index: índice en el action space para futuros."""
        self._action_override = action
        self._leverage_override = leverage_override
        self._leverage_index = leverage_index

    def step(self):
        if self._done:
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
            decision = self.policy.decide(obs)
        else:
            decision = self._decision_from_action(action, obs)
        # Aplicar sizing según el tipo de mercado
        if self.cfg.mode.endswith("futures"):
            # Usar leverage override si está disponible, sino cfg.leverage, sino 2.0 por defecto
            lev = float(self._leverage_override) if self._leverage_override is not None else self.cfg.leverage
            sized = self.risk.size_futures(self.portfolio, decision, lev, self.portfolio.equity_quote)
        else:
            sized = self.risk.apply(self.portfolio, self.pos, decision, obs)

        reward = 0.0

        # Inicializar risk_pct por defecto
        risk_pct = 0.0

        # 3) Ejecutar apertura
        if sized.should_open:
            fill = self.oms.open("LONG" if sized.side > 0 else "SHORT", sized.qty, sized.price_hint, sized.sl, sized.tp)
            self.accounting.apply_open(fill, self.portfolio, self.pos, self.cfg)
            # TTL de la policy
            self.pos.ttl_bars = decision.ttl_bars
            
            # ← NUEVO: registrar timestamp de apertura
            self.pos.open_ts = ts_now
            self.pos.bars_held = 0
            
            # ← NUEVO: incrementar contador de trades
            self._trades_executed += 1
            
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
            if qty_close and qty_close > 0:
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
                    exec_tf=exec_tf
                )
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
                    exec_tf=exec_tf
                )
                reward += float(realized)

        # 6) Verificar quiebra ANTES de avanzar el broker
        if not self._bankruptcy_detected:
            bankruptcy_occurred = self.risk.check_bankruptcy(
                self.portfolio, 
                self._init_cash, 
                self.events_bus, 
                ts_now
            )
            if bankruptcy_occurred:
                self._bankruptcy_detected = True
                # Aplicar penalización fuerte por quiebra
                penalty_reward = float(getattr(self.cfg.risk.common, "bankruptcy", {}).get("penalty_reward", -10.0))
                reward += penalty_reward
                
                # Log de quiebra
                self._run_logger.finish(
                    final_balance=self.portfolio.cash_quote,
                    final_equity=self.portfolio.equity_quote,
                    ts_end=ts_now,
                    bankruptcy=True,
                    penalty_reward=penalty_reward
                )
                
                # ← NUEVO: TERMINACIÓN INMEDIATA - no ejecutar más código
                self._done = True
                events = self.events_bus.drain()
                info = {"events": events, "bankruptcy": True, "penalty_reward": penalty_reward}
                return obs, reward, True, info
        
        # ← NUEVO: Si llegamos aquí, NO hay quiebra - continuar normalmente

        # 7) PnL no realizado / avanzar broker
        self.accounting.update_unrealized(self.broker, self.pos, self.portfolio)
        self.broker.next()
        
        # ← NUEVO: incrementar barras que estuvo abierta la posición
        if self.pos.side != 0 and self.pos.open_ts is not None:
            self.pos.bars_held += 1

        # 8) Siguiente observación y eventos
        next_obs = self._build_observation()
        events = self.events_bus.drain()
        info = {"events": events}

        # Señal de fin de histórico (cuando no avanza ts)
        done = (next_obs["ts"] == obs["ts"])
        if done:
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
                    ts_end=int(next_obs["ts"])
                )
            else:
                # No loguear runs vacíos - solo resetear para siguiente episodio
                print(f"🔄 Episodio sin actividad real - no logueando run vacío")
        
        self._done = done
        return next_obs, reward, done, info

    # ------------- Internos -------------
    def _decision_from_action(self, action: int, obs: dict) -> Decision:
        price = float(obs["tfs"][self.policy.exec_tf]["close"])
        feats = obs.get("features", {})
        atr_exec = feats.get(self.policy.exec_tf, {}).get("atr14") or feats.get(self.policy.base_tf, {}).get("atr14")
        if action == 1:
            return Decision(should_close_all=True, price_hint=price)
        if action == 2:
            return Decision(should_open=False, price_hint=price)  # bloquear aperturas
        if action == 3:
            sl, tp = sl_tp_from_atr(price, atr_exec, side=+1, k_sl=1.5, k_tp=2.0)
            return Decision(should_open=True, side=+1, price_hint=price, sl=sl, tp=tp, ttl_bars=200, trailing=True)
        if action == 4:
            sl, tp = sl_tp_from_atr(price, atr_exec, side=-1, k_sl=1.5, k_tp=2.0)
            return Decision(should_open=True, side=-1, price_hint=price, sl=sl, tp=tp, ttl_bars=200, trailing=True)
        # default: sin override → deja a la policy
        return Decision(should_open=False, price_hint=price)

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
