# base_env/config/symbols_loader.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
from pydantic import BaseModel, validator

VALID_MODES = {"train_spot","train_futures","backtest","live_spot","live_futures"}

class LeverageSpec(BaseModel):
    min: float
    max: float
    step: float = 1.0
    default: float = 2.0

    @validator("min", "max", "step", "default")
    def _pos(cls, v: float) -> float:
        if float(v) <= 0:
            raise ValueError("Leverage valores deben ser > 0")
        return float(v)

    @validator("max")
    def _range(cls, v, values):
        if "min" in values and float(v) < float(values["min"]):
            raise ValueError("leverage.max < leverage.min")
        return float(v)

    @validator("default")
    def _default_in_range(cls, v, values):
        mn, mx = float(values["min"]), float(values["max"])
        if not (mn <= float(v) <= mx):
            raise ValueError("leverage.default debe estar en [min, max]")
        return float(v)

class SymbolConfig(BaseModel):
    symbol: str
    mode: str
    enabled_tfs: List[str]
    filters: Dict[str, Any]
    leverage: Optional[LeverageSpec] = None  # sólo para futures

    @validator("mode")
    def _mode_ok(cls, v):
        if v not in VALID_MODES:
            raise ValueError(f"Modo inválido: {v}")
        return v

    @validator("leverage", always=True)
    def _lev_required_for_futures(cls, v, values):
        if values.get("mode","").endswith("futures"):
            if v is None:
                raise ValueError("En modos *futures* es obligatorio definir leverage{min,max[,step,default]}")
            # limites duros de exchange / política
            if not (2.0 <= v.min <= 25.0) or not (2.0 <= v.max <= 25.0):
                raise ValueError("Leverage permitido global: 2.0–25.0")
        else:
            return None
        return v

def load_symbols(path: str | Path = "config/symbols.yaml") -> List[SymbolConfig]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return [SymbolConfig(**it) for it in raw.get("symbols", [])]
