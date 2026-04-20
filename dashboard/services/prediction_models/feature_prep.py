from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard.services.portfolio_service import _daily_return_series_prefer_varying_close


DEFAULT_FEATURE_COLS: tuple[str, ...] = (
    "ret_1",
    "ret_3",
    "ret_5",
    "momentum_10",
    "volatility_10",
    "sma_ratio",
)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lag-safe features at time t and a t+1 target.

    Target is next bar's daily return; features only use information available through t.
    """
    out = df.copy()
    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
        out = out.dropna(subset=["trade_date"]).sort_values("trade_date")
        # One row per calendar day before pct_change — duplicate dates in gold make adjacent
        # returns ~0 and break ML targets + attach_ml_strategy_returns (flat OOS PnL).
        out = out.drop_duplicates(subset=["trade_date"], keep="last")

    # Prefer close-based returns; fall back to gold daily_return when close path is flat (portfolio_service).
    out["daily_return"] = _daily_return_series_prefer_varying_close(out)
    close = pd.to_numeric(out["close_price"], errors="coerce")

    out["target"] = out["daily_return"].shift(-1)

    out["ret_1"] = out["daily_return"]
    out["ret_3"] = out["daily_return"].rolling(3, min_periods=3).mean()
    out["ret_5"] = out["daily_return"].rolling(5, min_periods=5).mean()

    base_close = close.shift(10)
    out["momentum_10"] = (close / base_close) - 1.0

    out["volatility_10"] = out["daily_return"].rolling(10, min_periods=10).std()

    sma_20 = pd.to_numeric(out["sma_20"], errors="coerce") if "sma_20" in out.columns else pd.Series(np.nan, index=out.index)
    sma_50 = pd.to_numeric(out["sma_50"], errors="coerce") if "sma_50" in out.columns else pd.Series(np.nan, index=out.index)
    if sma_20.isna().all() or sma_50.isna().all():
        sma_20 = close.rolling(20, min_periods=20).mean()
        sma_50 = close.rolling(50, min_periods=50).mean()
    out["sma_ratio"] = sma_20 / sma_50

    out = out.dropna(subset=list(DEFAULT_FEATURE_COLS) + ["target"]).reset_index(drop=True)
    return out
