"""Local-cache public data fallback with explicit survivorship-bias disclosure."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def download_sp500_panel(
    cache_path: Path,
    start: str = "2014-01-01",
    end: str | None = None,
    max_assets: int = 300,
) -> pd.DataFrame:
    """Download adjusted daily OHLCV for current S&P 500 members and cache as Parquet.

    This fallback is not point-in-time and therefore has survivorship and membership
    look-ahead bias.
    Asset selection uses median dollar volume in the first 252 available dates, not the full sample.
    """
    if cache_path.exists():
        return pd.read_parquet(cache_path).set_index(["date", "symbol"]).sort_index()
    import requests
    import yfinance as yf

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0 research-cache/1.0"}, timeout=30
    )
    response.raise_for_status()
    from io import StringIO

    members = pd.read_html(StringIO(response.text))[0]
    sector_map = dict(zip(members.Symbol.str.replace(".", "-", regex=False),
                          members["GICS Sector"], strict=True))
    symbols = list(sector_map)
    downloaded = yf.download(symbols + ["SPY"], start=start, end=end, auto_adjust=True,
                             group_by="ticker", threads=True, progress=False)
    market = downloaded["SPY"]["Close"].pct_change(fill_method=None).rename("market_return")
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        if symbol not in downloaded.columns.get_level_values(0):
            continue
        asset = downloaded[symbol].rename(columns=str.lower)
        required = ["open", "high", "low", "close", "volume"]
        if not set(required).issubset(asset.columns):
            continue
        asset = asset[required].dropna(subset=required)
        asset["symbol"] = symbol
        asset["sector"] = sector_map[symbol]
        asset["adjusted_close"] = asset["close"]
        frames.append(asset.reset_index().rename(columns={"Date": "date"}))
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)
    initial_dates = sorted(panel.date.unique())[:252]
    initial = panel[panel.date.isin(initial_dates)]
    liquidity = (initial.close * initial.volume).groupby(initial.symbol).median()
    selected = liquidity.nlargest(max_assets).index
    panel = panel[panel.symbol.isin(selected)].copy()
    panel = panel.merge(market.rename_axis("date").reset_index(), on="date", how="left")
    panel = panel.sort_values(["date", "symbol"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache_path, index=False)
    return panel.set_index(["date", "symbol"])
