# tests/conftest.py
# Descripción: Fixtures comunes: dataset Parquet sintético (aligned), config mínima, broker y OMS fake.

import os
import shutil
import time
from pathlib import Path
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from base_env.config.models import (
    EnvConfig, SymbolMeta, PipelineConfig, HierarchicalConfig, RiskConfig, FeesConfig
)

@pytest.fixture
def tmp_data_root(tmp_path: Path):
    # Estructura: data/BTCUSDT/spot/aligned/{1m,5m}/year=2025/month=04/part-2025-04.parquet
    root = tmp_path / "data"
    symbol = "BTCUSDT"
    market = "spot"
    for tf, step_ms, count in [
        ("1m", 60_000, 600),
        ("5m", 5*60_000, 120),
    ]:
        base_dir = root / symbol / market / "aligned" / tf / "year=2025" / "month=04"
        base_dir.mkdir(parents=True, exist_ok=True)
        start = 1743465600000  # 2025-04-01 00:00:00 UTC aprox
        rows = []
        price = 80_000.0
        for i in range(count):
            ts = start + i * step_ms
            # mini recorrido con ruido suave
            price = price + (1 if i % 3 == 0 else -0.5)
            o = price
            h = price + 10
            l = price - 10
            c = price + (0.2 if i % 5 == 0 else -0.1)
            v = 10 + (i % 7)
            rows.append((ts, o, h, l, c, v, symbol, market, tf, int(time.time()*1000)))

        table = pa.Table.from_pylist(
            [
                {
                    "ts": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5],
                    "symbol": r[6], "market": r[7], "tf": r[8], "ingestion_ts": r[9]
                } for r in rows
            ],
            schema=pa.schema([
                ("ts", pa.int64()),
                ("open", pa.float64()),
                ("high", pa.float64()),
                ("low", pa.float64()),
                ("close", pa.float64()),
                ("volume", pa.float64()),
                ("symbol", pa.string()),
                ("market", pa.string()),
                ("tf", pa.string()),
                ("ingestion_ts", pa.int64()),
            ])
        )
        pq.write_table(table, base_dir / "part-2025-04.parquet", compression="zstd")
    return root

@pytest.fixture
def env_cfg():
    return EnvConfig(
        mode="backtest",
        market="spot",
        symbol_meta=SymbolMeta(symbol="BTCUSDT", market="spot", enabled_tfs=["1m","5m"], filters={"minNotional":5.0, "lotStep":0.001}),
        tfs=["1m","5m"],
        pipeline=PipelineConfig(strict_alignment=True),
        hierarchical=HierarchicalConfig(
            direction_tfs=["1d","4h"],  # no presentes, no pasa nada
            confirm_tfs=["5m"],
            execute_tfs=["1m"],
            min_confidence=0.0,  # abrir fácil para el test
            dedup_open_window_bars=1
        ),
        risk=RiskConfig(),
        fees=FeesConfig(),
    )

class MockOMS:
    def open(self, side, qty, price_hint, sl, tp):
        # devuelve fill simple
        return {"side": 1 if side == "LONG" else -1, "qty": float(qty), "price": float(price_hint), "fees": 0.0, "sl": sl, "tp": tp}
    def close(self, qty, price_hint):
        return {"qty": float(qty), "price": float(price_hint), "fees": 0.0}

@pytest.fixture
def mock_oms():
    return MockOMS()
