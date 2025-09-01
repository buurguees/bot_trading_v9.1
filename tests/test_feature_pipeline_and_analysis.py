# tests/test_feature_pipeline_and_analysis.py
from base_env.features.pipeline import FeaturePipeline
from base_env.config.models import PipelineConfig, HierarchicalConfig
from base_env.analysis.hierarchical import HierarchicalAnalyzer

def test_features_and_hierarchical(tmp_data_root):
    # simular 100 pasos sólo con 1m (push manual)
    fp = FeaturePipeline(PipelineConfig(strict_alignment=True))
    # barras dummy
    for i in range(30):
        bar = {"ts": 1743465600000 + i*60_000, "open": 100+i, "high": 101+i, "low": 99+i, "close": 100+i+(0.2 if i%3==0 else -0.1), "volume": 10+i}
        fp.compute({"1m": bar})
    feats = fp.compute({"1m": bar})
    assert "1m" in feats and feats["1m"]["ema20"] is not None

    hz = HierarchicalAnalyzer(HierarchicalConfig(direction_tfs=[], confirm_tfs=[], execute_tfs=["1m"], min_confidence=0.0))
    res = hz.analyze(features=feats, smc={})
    assert res.confidence >= 0.0 and res.side_hint in (-1,0,1)
