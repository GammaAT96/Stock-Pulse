from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _unique_index_returns_series(sr: pd.Series) -> pd.Series:
    """Coerce to numeric, sort by index, collapse duplicate timestamps with ``keep='last'`` (EOD refresh)."""
    s = pd.to_numeric(sr, errors="coerce").sort_index()
    if not s.index.is_unique:
        s = s.loc[~s.index.duplicated(keep="last")]
    return s


def aggregate_feature_importances(
    models: list[Any],
    feature_cols: list[str] | tuple[str, ...],
) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    """Mean and std of feature_importances_ across fitted models (tree ensembles)."""
    rows: list[list[float]] = []
    cols = list(feature_cols)
    for m in models:
        imp = getattr(m, "feature_importances_", None)
        if imp is None:
            continue
        rows.append(list(imp))
    if not rows:
        return None, None
    df = pd.DataFrame(rows, columns=cols)
    return df.mean().sort_values(ascending=False), df.std().reindex(df.mean().index)


def cross_sectional_inverse_vol_weights_rolling(
    panel: pd.DataFrame,
    *,
    vol_window: int = 60,
    min_periods: int = 20,
    vol_method: Literal["rolling_std", "ewma"] = "rolling_std",
    ewma_lambda: float = 0.94,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    **Causal** inverse-vol weights: ``rolling_std`` or **EWMA** variance of each leg with ``shift(1)``,
    then row-normalize 1/σ.

    Portfolio return ``(panel * weights).sum(axis=1)`` adapts over time without using future returns
    in the volatility estimate.
    """
    vw = max(5, int(vol_window))
    mp = max(5, int(min_periods))
    if vol_method == "rolling_std":
        vol = panel.rolling(vw, min_periods=mp).std().shift(1)
    elif vol_method == "ewma":
        lam = float(ewma_lambda)
        if not (0.0 < lam < 1.0):
            raise ValueError("ewma_lambda must lie strictly between 0 and 1.")
        alpha = 1.0 - lam
        ewma_var = panel.pow(2).ewm(alpha=alpha, adjust=False).mean()
        vol = np.sqrt(ewma_var).shift(1)
    else:
        raise ValueError(f"Unknown vol_method: {vol_method}")
    inv = 1.0 / vol.replace(0.0, np.nan)
    w = inv.div(inv.sum(axis=1), axis=0).fillna(0.0)
    port = (panel * w).sum(axis=1)
    port.name = "portfolio_return"
    return port, w


def cross_sectional_inverse_vol_weights(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Time-constant inverse-volatility weights from in-sample column volatilities.

    ``weights`` sums to 1; portfolio return is ``(panel * weights).sum(axis=1)``.
    """
    vol = panel.std(axis=0, ddof=1).replace(0.0, np.nan)
    inv = 1.0 / vol
    if inv.isna().all():
        raise ValueError("All volatilities invalid for inverse-vol weighting.")
    w = inv / inv.sum(skipna=True)
    w = w.fillna(0.0)
    port = (panel * w).sum(axis=1)
    port.name = "portfolio_return"
    return port, w


def cross_sectional_equal_weight(
    strategy_returns_by_symbol: dict[str, pd.Series],
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Align strategy daily returns on inner-joined dates, then equal-weight mean.

    Each series must be indexed by trade_date (datetime64). Duplicate timestamps
    on a leg are collapsed to the **last** value (same convention as pivot dedupe).
    """
    if len(strategy_returns_by_symbol) < 2:
        raise ValueError("Need at least 2 symbols for a portfolio.")
    mats: list[pd.Series] = []
    for sym, sr in strategy_returns_by_symbol.items():
        s = _unique_index_returns_series(sr)
        s.name = sym
        mats.append(s)
    panel = pd.concat(mats, axis=1, join="inner").sort_index()
    panel = panel.dropna(how="any")
    if panel.empty:
        raise ValueError("No overlapping dates across symbols after dropna.")
    port = panel.mean(axis=1)
    port.name = "portfolio_return"
    return port, panel


def portfolio_metrics_from_returns(daily: pd.Series) -> dict[str, float]:
    daily = pd.to_numeric(daily, errors="coerce").dropna()
    n = len(daily)
    if n < 2:
        return {"cagr": float("nan"), "sharpe": float("nan"), "max_dd": float("nan"), "n_days": float(n)}
    cum = (1.0 + daily).cumprod()
    years = n / TRADING_DAYS
    end = float(cum.iloc[-1])
    cagr = end ** (1.0 / years) - 1.0 if years > 0 and end > 0 else float("nan")
    peak = cum.cummax()
    max_dd = float((cum / peak - 1.0).min())
    std = float(daily.std(ddof=1))
    mu = float(daily.mean())
    sharpe = (mu / std) * math.sqrt(TRADING_DAYS) if std > 0 else float("nan")
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "n_days": float(n)}


def returns_correlation_matrix(
    market_returns_by_symbol: dict[str, pd.Series],
) -> pd.DataFrame:
    """Pearson correlation of aligned market daily returns (inner join on dates). Duplicate index labels per leg are collapsed to the last value."""
    if not market_returns_by_symbol:
        raise ValueError("No symbols.")
    mats: list[pd.Series] = []
    for sym, sr in market_returns_by_symbol.items():
        s = _unique_index_returns_series(sr)
        s.name = sym
        mats.append(s)
    panel = pd.concat(mats, axis=1, join="inner").sort_index().dropna(how="any")
    if panel.shape[1] < 2 or len(panel) < 5:
        raise ValueError("Not enough overlapping data for correlation matrix.")
    return panel.corr()
