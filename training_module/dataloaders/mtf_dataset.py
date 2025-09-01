# training_module/dataloaders/mtf_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import IterableDataset
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    IterableDataset = object  # type: ignore


class MTFDataset(IterableDataset):  # type: ignore[misc]
    """
    DataLoader por ventanas deslizantes sobre el dataset entrenable parquet.
    Produce (obs [B,W,F], action_mask [B,3], target_y [B], meta dict).
    """

    def __init__(
        self,
        dataset_parquet: str | Path,
        window_size: int = 64,
        batch_size: int = 1024,
        normalize_obs: bool = True,
    ):
        super().__init__()  # type: ignore
        self.path = Path(dataset_parquet)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset parquet no encontrado: {self.path}")
        self.df = pd.read_parquet(self.path)
        self.window_size = int(window_size)
        self.batch_size = int(batch_size)
        self.normalize_obs = bool(normalize_obs)

        drop_cols = {"timestamp", "y", "tte"}
        self.feature_cols: List[str] = [
            c for c in self.df.columns
            if c not in drop_cols and pd.api.types.is_numeric_dtype(self.df[c])
        ]
        if len(self.feature_cols) == 0:
            raise ValueError("No hay columnas de features numéricas para la observación.")

    def __iter__(self) -> Iterator[Tuple]:
        W = self.window_size
        F = len(self.feature_cols)
        X_buf: List[np.ndarray] = []
        M_buf: List[np.ndarray] = []
        Y_buf: List[np.ndarray] = []
        N = len(self.df)

        for i in range(W, N):
            win = self.df.iloc[i - W : i]
            x = win[self.feature_cols].values.astype(np.float32)
            if self.normalize_obs:
                mu = x.mean(axis=0, keepdims=True)
                sd = x.std(axis=0, keepdims=True) + 1e-8
                x = (x - mu) / sd

            # máscara por defecto: todas permitidas (HOLD, ENTER, EXIT)
            mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)

            # target: y actual (de la última fila del window)
            y = np.int64(self.df.iloc[i]["y"]) if "y" in self.df.columns else np.int64(0)

            X_buf.append(x)
            M_buf.append(mask)
            Y_buf.append(np.array(y))

            if len(X_buf) >= self.batch_size:
                yield self._pack_batch(X_buf, M_buf, Y_buf)
                X_buf, M_buf, Y_buf = [], [], []

        if len(X_buf) > 0:
            yield self._pack_batch(X_buf, M_buf, Y_buf)

    def _pack_batch(self, X: List[np.ndarray], M: List[np.ndarray], Y: List[np.ndarray]):
        Xb = np.stack(X, axis=0)  # [B,W,F]
        Mb = np.stack(M, axis=0)  # [B,3]
        Yb = np.stack(Y, axis=0)  # [B]

        if _HAS_TORCH:
            return (
                torch.from_numpy(Xb).float(),
                torch.from_numpy(Mb).float(),
                torch.from_numpy(Yb).long(),
                {"path": str(self.path)},
            )
        else:
            return (Xb, Mb, Yb, {"path": str(self.path)})
