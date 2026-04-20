"""Equal-weight multi-asset portfolio helpers (daily rebalanced implicitly)."""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from dashboard.services.duplicate_pivot_warnings import warn_once_duplicate_trade_symbol_rows

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def _daily_return_series_prefer_varying_close(g: pd.DataFrame) -> pd.Series:
    """
    Primary: ``close_price`` pct change. If that is effectively flat (broken OHLC in DB),
    fall back to ``daily_return`` from gold when present — fixes flat portfolio lines.
    """
    close = pd.to_numeric(g["close_price"], errors="coerce")
    dr = close.pct_change()
    n = int(dr.notna().sum())
    std_dr = float(dr.std(ddof=1)) if n > 2 else 0.0
    if std_dr > 1e-12:
        return dr.fillna(0.0)
    if "daily_return" in g.columns:
        alt = pd.to_numeric(g["daily_return"], errors="coerce")
        std_alt = float(alt.std(ddof=1)) if len(alt) > 2 else 0.0
        if std_alt > 1e-12 or float(alt.fillna(0.0).abs().sum()) > 1e-10:
            return alt.fillna(0.0)
    return dr.fillna(0.0)


def compute_portfolio(df: pd.DataFrame, symbols: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    df must contain: trade_date, symbol, daily_return

    Equal-weight: each day, portfolio_return = mean of constituent daily returns.

    Dates are restricted to the intersection where **all** selected symbols have a return
    (`dropna`), avoiding the downward bias from treating missing names as 0% days.
    """
    if not symbols or len(symbols) < 2:
        raise ValueError("Need at least 2 symbols")

    work = df[df["symbol"].isin(symbols)].copy()
    if work.empty:
        raise ValueError("No rows after filtering symbols")

    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work = work.dropna(subset=["trade_date", "symbol"])
    work["daily_return"] = pd.to_numeric(work["daily_return"], errors="coerce")
    work = work.sort_values(["trade_date", "symbol"])
    warn_once_duplicate_trade_symbol_rows(logger, "compute_portfolio", work)
    work = work.drop_duplicates(subset=["trade_date", "symbol"], keep="last")

    pivot = work.pivot_table(
        index="trade_date",
        columns="symbol",
        values="daily_return",
        aggfunc="last",
    ).sort_index()

    missing_cols = [s for s in symbols if s not in pivot.columns]
    if missing_cols:
        raise ValueError(f"Missing symbols after pivot: {missing_cols}")

    pivot = pivot[symbols].dropna(how="any")
    if pivot.empty:
        raise ValueError("No common trading dates across selected symbols (after dropna).")

    pivot["portfolio_return"] = pivot[symbols].mean(axis=1)

    pivot["cum_return"] = (1.0 + pivot["portfolio_return"]).cumprod()
    pivot["total_return"] = pivot["cum_return"] - 1.0

    n = len(pivot)
    years = n / TRADING_DAYS if n else 0.0
    final_cum = float(pivot["cum_return"].iloc[-1]) if n else float("nan")

    if years > 0 and final_cum == final_cum and final_cum > 0:
        cagr = final_cum ** (1.0 / years) - 1.0
    else:
        cagr = float("nan")

    peak = pivot["cum_return"].cummax()
    dd = (pivot["cum_return"] / peak) - 1.0
    max_dd = float(dd.min()) if n else float("nan")

    pr = pivot["portfolio_return"]
    sigma = float(pr.std(ddof=1)) if n > 1 else 0.0
    mu = float(pr.mean()) if n else float("nan")
    if sigma > 0 and not math.isnan(mu):
        sharpe = (mu / sigma) * math.sqrt(TRADING_DAYS)
    else:
        sharpe = float("nan")

    metrics = {
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "n_days": n,
    }
    return pivot, metrics


def build_symbol_returns_panel(
    symbols: list[str],
    start_date,
    end_date,
    fetch_fn,
) -> pd.DataFrame:
    """
    fetch_fn(symbol) -> DataFrame with trade_date + close_price (+ symbol optional)

    Daily returns: ``close_price`` pct change; if that is flat, uses ``daily_return`` from
    gold when available (avoids an all-zero portfolio when OHLC is bad but gold features exist).
    """
    rows: list[pd.DataFrame] = []
    for sym in symbols:
        raw = fetch_fn(sym)
        if raw is None or raw.empty:
            continue
        g = raw.copy()
        g["trade_date"] = pd.to_datetime(g["trade_date"], errors="coerce")
        g = g.dropna(subset=["trade_date"]).sort_values("trade_date")
        if "close_price" not in g.columns:
            continue
        # One row per calendar day before pct_change — duplicate (trade_date) rows in gold
        # (e.g. re-loads) make adjacent pct_change ≈ 0 and flatten the whole portfolio.
        g = g.drop_duplicates(subset=["trade_date"], keep="last")
        g["daily_return"] = _daily_return_series_prefer_varying_close(g)
        g["symbol"] = sym
        td = g["trade_date"].dt.date
        g = g[(td >= start_date) & (td <= end_date)]
        if g.empty:
            continue
        rows.append(g[["trade_date", "symbol", "daily_return"]])

    if not rows:
        return pd.DataFrame(columns=["trade_date", "symbol", "daily_return"])
    return pd.concat(rows, ignore_index=True)
