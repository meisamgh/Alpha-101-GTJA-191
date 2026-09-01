"""Pluggable panel loaders. Providers must supply point-in-time metadata themselves."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

REQUIRED_COLUMNS = {"date", "symbol", "open", "high", "low", "close", "volume"}


class PanelLoader(Protocol):
    def load(self) -> pd.DataFrame: ...


@dataclass(frozen=True)
class CSVPanelLoader:
    path: Path

    def load(self) -> pd.DataFrame:
        frame = pd.read_csv(self.path, parse_dates=["date"])
        return normalize_panel(frame)


def normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert(None).dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    if "adjusted_close" not in out:
        out["adjusted_close"] = out["close"]
    if "sector" not in out:
        out["sector"] = "UNKNOWN"
    return out.sort_values(["date", "symbol"]).set_index(["date", "symbol"])


def load_panel(loader: PanelLoader) -> pd.DataFrame:
    return loader.load()
