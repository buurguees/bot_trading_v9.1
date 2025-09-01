from __future__ import annotations
from pathlib import Path
import yaml

class BaseContext:
    def __init__(self, settings_path: str = "config/settings.yaml"):
        with open(settings_path, "r", encoding="utf-8") as f:
            self.settings = yaml.safe_load(f) or {}
        self.paths = self.settings.get("paths", {})
        self.ohlcv_dir = Path(self.paths.get("ohlcv_dir", "data/warehouse/ohlcv"))
        self.features_dir = Path(self.paths.get("features_dir", "data/warehouse/features"))
        self.smc_dir = Path(self.paths.get("smc_dir", "data/warehouse/smc"))
        self.reports_dir = Path(self.paths.get("reports_dir", "reports"))
        # Asegura que existan
        self.ohlcv_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.smc_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
