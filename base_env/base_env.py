
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
        self.policy = PolicyEngine(cfg.hierarchical)
        self.risk = RiskManager(cfg.risk, cfg.symbol_meta)
        self.acc = Accounting(cfg.fees, cfg.market)
        self.events = EventBus()

        # Estado
        self.pos = PositionState()
        self.portfolio = PortfolioState(market=cfg.market)

        # Control
        self._done: bool = False

    # ------------- API pública -------------
    def reset(self) -> Dict[str, Any]:
        """Resetea estados internos y devuelve la primera observación."""
        self.pos.reset()
        self.portfolio.reset()
        self._done = False
        return self._build_observation()

    def step(self) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Ejecuta un paso completo: avanza datos, analiza, decide, aplica y contabiliza."""
        # 1) Avanzar broker
        self.broker.next()

        # 2) Observación base (alineación + features + smc + jerárquico)
        obs = self._build_observation()

        # 3) Política (gating + decisión)
        decision: Decision = self.policy.decide(obs)

        # 4) Riesgo (validación y sizing)
        sized = self.risk.apply(self.portfolio, self.pos, decision, obs)

        # 5) Ejecutar (OMS) y contabilizar
        reward = 0.0
        if sized.should_open:
            fill = self.oms.open(sized.side, sized.qty, sized.price_hint, sized.sl, sized.tp)
            self.acc.apply_open(fill, self.portfolio, self.pos, self.cfg)
            self.events.publish(DomainEvent.ts_now("order_opened", {"fill": fill}))
        if sized.should_close_partial or sized.should_close_all:
            close_qty = sized.close_qty
            fill = self.oms.close(close_qty, sized.price_hint)
            pnl_realized = self.acc.apply_close(fill, self.portfolio, self.pos, self.cfg)
            reward += float(pnl_realized)
            self.events.publish(DomainEvent.ts_now("order_closed", {"fill": fill, "pnl": pnl_realized}))

        # 6) PnL no realizado, trailing, TTL, breakers
        self.acc.update_unrealized(self.broker, self.pos, self.portfolio)
        self.risk.maintenance(self.portfolio, self.pos, self.broker, self.events)
        self._done = self.acc.is_end_of_data(self.broker)

        info = {"events": self.events.drain(), "decision": decision.model_dump() if hasattr(decision, "model_dump") else dict(decision.__dict__)}
        return obs, reward, self._done, info

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
