
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
from .risk.manager import RiskManager
from .accounting.ledger import PositionState, PortfolioState, Accounting
from .events.domain import EventBus, DomainEvent
from .events.bus import SimpleEventBus
from .policy.gating import PolicyEngine
from .risk.manager import RiskManager, SizedDecision
from .accounting.ledger import Accounting, PortfolioState, PositionState

class OMSAdapter(Protocol):
    """Interfaz mínima para ejecución (Sim/Paper/Live). Slippage se aplica fuera del core."""
    def open(self, side: int, qty: float, price_hint: float, sl: Optional[float], tp: Optional[float]) -> Dict[str, Any]: ...
    def close(self, qty: float, price_hint: float) -> Dict[str, Any]: ...


class BaseTradingEnv:
    """Entorno base canónico (idéntico en train/backtest/live; cambian adapters)."""

    def __init__(self, cfg: EnvConfig, broker: DataBroker, oms: OMSAdapter) -> None:
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
        self.accounting = Accounting(fees_cfg=cfg.fees, market=cfg.market)
        self.events_bus = SimpleEventBus()

        # Control
        self._done = False

    # ------------- API pública -------------
    def reset(self):
        self.pos.reset()
        self.portfolio.reset()
        self._done = False
        return self._build_observation()

    def step(self):
        if self._done:
            return self._build_observation(), 0.0, True, {"events": []}

        # 1) Construir obs antes de decidir
        obs = self._build_observation()
        ts_now = int(obs["ts"])
        exec_tf = self.policy.exec_tf

        # 2) DECISIÓN de apertura/cierre por policy
        decision = self.policy.decide(obs)
        sized = self.risk.apply(self.portfolio, self.pos, decision, obs)

        reward = 0.0

        # 3) Ejecutar apertura
        if sized.should_open:
            fill = self.oms.open("LONG" if sized.side > 0 else "SHORT", sized.qty, sized.price_hint, sized.sl, sized.tp)
            self.accounting.apply_open(fill, self.portfolio, self.pos, self.cfg)
            # TTL de la policy
            self.pos.ttl_bars = decision.ttl_bars
            self.events_bus.emit("OPEN", ts=ts_now, side=("LONG" if sized.side > 0 else "SHORT"), qty=sized.qty, price=fill["price"], sl=sized.sl, tp=sized.tp)

        # 4) Ejecutar cierres explícitos (policy/risk)
        if sized.should_close_all or sized.should_close_partial:
            qty_close = sized.close_qty if sized.should_close_partial else (self.pos.qty or 0.0)
            if qty_close and qty_close > 0:
                fill = self.oms.close(qty_close, sized.price_hint)
                realized = self.accounting.apply_close(fill, self.portfolio, self.pos, self.cfg)
                reward += float(realized)
                self.events_bus.emit("CLOSE", ts=ts_now, qty=qty_close, price=fill["price"], reason=("PARTIAL" if sized.should_close_partial else "ALL"))

        # 5) Mantenimiento SL/TP/TTL/Trailing (puede devolver cierre total)
        auto_close = self.risk.maintenance(self.portfolio, self.pos, self.broker, self.events_bus, obs, exec_tf, ts_now)
        if auto_close is not None and (auto_close.should_close_all or auto_close.should_close_partial):
            qty_close = auto_close.close_qty if auto_close.should_close_partial else (self.pos.qty or 0.0)
            if qty_close and qty_close > 0:
                fill = self.oms.close(qty_close, auto_close.price_hint)
                realized = self.accounting.apply_close(fill, self.portfolio, self.pos, self.cfg)
                reward += float(realized)
                self.events_bus.emit("CLOSE", ts=ts_now, qty=qty_close, price=fill["price"], reason=("AUTO_PARTIAL" if auto_close.should_close_partial else "AUTO_ALL"))

        # 6) PnL no realizado / avanzar broker
        self.accounting.update_unrealized(self.broker, self.pos, self.portfolio)
        self.broker.next()

        # 7) Siguiente observación y eventos
        next_obs = self._build_observation()
        events = self.events_bus.drain()
        info = {"events": events}

        # Señal de fin (si base timeline terminó)
        done = (next_obs["ts"] == obs["ts"])  # si no avanzó, estamos al final
        self._done = done
        return next_obs, reward, done, info

    # ------------- Internos -------------
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
