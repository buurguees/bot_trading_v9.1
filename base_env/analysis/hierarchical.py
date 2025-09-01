# base_env/analysis/hierarchical.py
# Descripción: Señales por TF (ema20 vs ema50) y agregación jerárquica con ponderaciones.
# Si un TF no está presente, se omite. Confidence en [0..1].

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
from ..config.models import HierarchicalConfig


@dataclass
class HierarchicalResult:
    by_tf: Dict[str, Dict[str, float]]
    confidence: float
    side_hint: int  # -1 short, +1 long, 0 neutral


def _signal_from_features(feats_tf: Dict[str, Any]) -> float:
    e20 = feats_tf.get("ema20")
    e50 = feats_tf.get("ema50")
    if e20 is None or e50 is None:
        return 0.0
    return 1.0 if e20 > e50 else -1.0


class HierarchicalAnalyzer:
    def __init__(self, cfg: HierarchicalConfig) -> None:
        self.cfg = cfg

    def analyze(self, features: Dict[str, Any], smc: Dict[str, Any]) -> HierarchicalResult:
        # 1) señales por TF disponibles
        all_tfs: List[str] = []
        all_tfs.extend(self.cfg.direction_tfs)
        all_tfs.extend(self.cfg.confirm_tfs)
        all_tfs.extend(self.cfg.execute_tfs)
        # Preservar orden pero sin duplicados
        seen = set()
        ordered_unique = [tf for tf in all_tfs if not (tf in seen or seen.add(tf))]

        by_tf: Dict[str, Dict[str, float]] = {}
        scores_dir: List[float] = []
        scores_conf: List[float] = []
        scores_exec: List[float] = []

        for tf in ordered_unique:
            feats_tf = features.get(tf)
            if not feats_tf:
                continue
            sig = _signal_from_features(feats_tf)
            by_tf[tf] = {"ema20_gt_ema50": 1.0 if sig > 0 else 0.0, "signal": sig}

            if tf in self.cfg.direction_tfs:
                scores_dir.append(sig)
            elif tf in self.cfg.confirm_tfs:
                scores_conf.append(sig)
            elif tf in self.cfg.execute_tfs:
                scores_exec.append(sig)

        # 2) agregación con ponderaciones (dir 0.5, conf 0.3, exec 0.2) normalizadas por TFs presentes
        def avg(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        s_dir = avg(scores_dir)
        s_conf = avg(scores_conf)
        s_exec = avg(scores_exec)

        # combinar y mapear a [0..1]
        combined = 0.5 * s_dir + 0.3 * s_conf + 0.2 * s_exec
        # combined ∈ [-1..1] → confidence ∈ [0..1]
        confidence = (combined + 1.0) / 2.0

        # side_hint por la capa de ejecución si está, sino por la confirmación, sino dirección
        side_hint = 0
        if scores_exec:
            side_hint = 1 if s_exec > 0 else -1
        elif scores_conf:
            side_hint = 1 if s_conf > 0 else -1
        elif scores_dir:
            side_hint = 1 if s_dir > 0 else -1

        return HierarchicalResult(by_tf=by_tf, confidence=max(0.0, min(1.0, confidence)), side_hint=side_hint)
