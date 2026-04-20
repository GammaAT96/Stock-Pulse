from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

SignalMode = Literal["long_only", "long_short_flat"]
ThresholdMode = Literal["static", "quantile_oos", "quantile_ref", "quantile_expanding"]
PositionStyle = Literal["discrete", "confidence"]
ConfidenceScale = Literal["oos_max", "rolling_max"]


def realized_turnover(df: pd.DataFrame) -> float:
    """Mean daily |Δposition| — comparable for discrete {-1,0,1} and continuous weights."""
    if "position" not in df.columns or df.empty:
        return float("nan")
    churn = df["position"].diff().abs().fillna(0.0).sum()
    return float(churn / max(len(df), 1))


def underwater_equity_curve(daily_returns: pd.Series) -> pd.Series:
    """Underwater / drawdown series from peak (negative values)."""
    dr = pd.to_numeric(daily_returns, errors="coerce").fillna(0.0)
    cum = (1.0 + dr).cumprod()
    peak = cum.cummax()
    return cum / peak - 1.0


def attach_ml_strategy_returns(
    oos: pd.DataFrame,
    *,
    rf_annual: float = 0.0,
    include_rf: bool = False,
    transaction_cost_pct: float = 0.0,
    prediction_threshold: float = 0.0,
    signal_mode: SignalMode = "long_only",
    threshold_mode: ThresholdMode = "static",
    quantile_hi: float = 0.7,
    quantile_lo: float = 0.3,
    quantile_expanding_min_periods: int = 40,
    reference_predictions: pd.Series | None = None,
    position_style: PositionStyle = "discrete",
    confidence_scale: ConfidenceScale = "oos_max",
    confidence_rolling_window: int = 50,
) -> pd.DataFrame:
    """
    Convert ML forecasts into positions with next-bar execution.

    threshold_mode:
      - static: fixed ``prediction_threshold`` hurdle.
      - quantile_oos: global OOS quantiles (non-causal within OOS).
      - quantile_ref: quantiles from ``reference_predictions``.
      - quantile_expanding: expanding quantile of **past** preds only, ``shift(1)`` before cut (causal path).

    confidence_scale:
      - oos_max: divide by max |pred| on the slice (legacy).
      - rolling_max: divide by rolling max |pred|, shifted one bar (causal scaling).
    """
    out = oos.copy()
    dr = pd.to_numeric(out["daily_return"], errors="coerce").fillna(0.0)
    pred = pd.to_numeric(out["prediction"], errors="coerce").fillna(0.0)

    thr_static = float(prediction_threshold)
    q_hi = float(quantile_hi)
    q_lo = float(quantile_lo)
    if not 0.0 < q_lo < q_hi < 1.0:
        raise ValueError("quantile_lo and quantile_hi must satisfy 0 < lo < hi < 1.")

    thr_long: float | pd.Series
    thr_short: float | pd.Series

    if threshold_mode == "static":
        thr_long = thr_static
        thr_short = -thr_static if signal_mode == "long_short_flat" else thr_static
    elif threshold_mode == "quantile_oos":
        thr_long = float(pred.quantile(q_hi))
        thr_short = float(pred.quantile(q_lo))
    elif threshold_mode == "quantile_ref":
        if reference_predictions is None or len(reference_predictions) == 0:
            raise ValueError("quantile_ref requires non-empty reference_predictions.")
        ref = pd.to_numeric(reference_predictions, errors="coerce").dropna()
        if ref.empty:
            raise ValueError("reference_predictions has no valid values.")
        thr_long = float(ref.quantile(q_hi))
        thr_short = float(ref.quantile(q_lo))
    elif threshold_mode == "quantile_expanding":
        mp = max(5, int(quantile_expanding_min_periods))
        roll_hi = pred.expanding(min_periods=mp).quantile(q_hi).shift(1)
        roll_lo = pred.expanding(min_periods=mp).quantile(q_lo).shift(1)
        thr_long = roll_hi
        thr_short = roll_lo
    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

    if position_style == "confidence":
        if confidence_scale == "oos_max":
            mx = float(pred.abs().max())
            scale = pd.Series(mx if mx >= 1e-12 else 1e-12, index=pred.index)
        elif confidence_scale == "rolling_max":
            rw = max(5, int(confidence_rolling_window))
            scale = pred.abs().rolling(rw, min_periods=max(3, rw // 5)).max().shift(1)
            scale = scale.clip(lower=1e-12).fillna(1e-12)
        else:
            raise ValueError(f"Unknown confidence_scale: {confidence_scale}")
        w = (pred / scale).clip(-1.0, 1.0)
        if signal_mode == "long_only":
            w = w.clip(0.0, 1.0)
        out["signal"] = w.astype(np.float64)
        out["position"] = out["signal"].shift(1).fillna(0.0).astype(np.float64)
        out["trade_change"] = out["position"].diff().fillna(0.0)
        out["trade"] = out["trade_change"].abs().fillna(0.0)
    elif position_style == "discrete":
        if isinstance(thr_long, pd.Series):
            valid_hi = thr_long.notna()
            if signal_mode == "long_only":
                out["signal"] = ((pred > thr_long) & valid_hi).astype(np.int8)
            elif signal_mode == "long_short_flat":
                valid_lo = thr_short.notna()
                out["signal"] = np.where(
                    (pred > thr_long) & valid_hi,
                    1,
                    np.where((pred < thr_short) & valid_lo, -1, 0),
                ).astype(np.int8)
            else:
                raise ValueError(f"Unknown signal_mode: {signal_mode}")
        else:
            if signal_mode == "long_only":
                out["signal"] = (pred > float(thr_long)).astype(np.int8)
            elif signal_mode == "long_short_flat":
                out["signal"] = np.where(pred > float(thr_long), 1, np.where(pred < float(thr_short), -1, 0)).astype(
                    np.int8
                )
            else:
                raise ValueError(f"Unknown signal_mode: {signal_mode}")
        out["position"] = out["signal"].shift(1).fillna(0).astype(np.int8)
        out["trade_change"] = out["position"].diff().fillna(0).astype(np.int8)
        out["trade"] = out["trade_change"].abs().fillna(0).astype(np.int8)
    else:
        raise ValueError(f"Unknown position_style: {position_style}")

    cost_drag = out["trade"] * float(transaction_cost_pct)
    out["cost_drag"] = cost_drag

    rf_daily = float(rf_annual) / TRADING_DAYS_PER_YEAR if include_rf else 0.0
    out["rf_daily"] = rf_daily
    pos_f = out["position"].astype(float)
    if signal_mode == "long_only":
        strat_dr = pos_f * dr + (1.0 - pos_f) * rf_daily
    else:
        flat = (np.abs(pos_f) < 1e-9).astype(float)
        strat_dr = pos_f * dr + flat * rf_daily
    out["strategy_return"] = strat_dr - cost_drag
    out["cum_market_return"] = (1.0 + dr).cumprod()
    out["cum_strategy_return"] = (1.0 + out["strategy_return"]).cumprod()
    return out


def rolling_information_coefficient(
    pred: pd.Series,
    target: pd.Series,
    window: int = 50,
    min_periods: int = 20,
) -> pd.Series:
    """Rolling Pearson correlation between prediction and realized next-day return."""
    p = pd.to_numeric(pred, errors="coerce")
    y = pd.to_numeric(target, errors="coerce")
    return p.rolling(int(window), min_periods=int(min_periods)).corr(y)


def ic_half_life_days(rolling_ic: pd.Series) -> float:
    """
    Persistence of IC from lag-1 autocorrelation (AR(1) half-life in **bars**).

    half_life = -ln(2) / ln(rho); undefined if rho is not in (0, 1).
    """
    s = pd.to_numeric(rolling_ic, errors="coerce").dropna()
    if len(s) < 30:
        return float("nan")
    rho = float(s.autocorr(lag=1))
    if rho <= 0.0 or rho >= 1.0 or math.isnan(rho):
        return float("nan")
    return float(-math.log(2.0) / math.log(rho))


def rebalance_k_from_ic_half_life(hl_bars: float) -> int:
    """
    Map IC half-life (bars) to a rebalance interval **k** in [1, 10].

    Heuristic: ``k = round(half_life / 2)`` (e.g. half-life 6 → rebalance every 3 days).
    """
    if hl_bars != hl_bars or hl_bars <= 0.0:
        return 1
    k = int(np.round(float(hl_bars) / 2.0))
    return int(np.clip(k, 1, 10))


def sharpe_daily(sr: pd.Series) -> float:
    sr = pd.to_numeric(sr, errors="coerce").dropna()
    if len(sr) <= 1:
        return float("nan")
    std = float(sr.std(ddof=1))
    if std <= 0 or math.isnan(std):
        return float("nan")
    return float(sr.mean() / std)


def max_drawdown_from_cum(cum: pd.Series) -> float:
    cum = pd.to_numeric(cum, errors="coerce").dropna()
    if cum.empty:
        return float("nan")
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return float(dd.min())


def cagr_from_cum(cum: pd.Series, trading_days_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    cum = pd.to_numeric(cum, errors="coerce").dropna()
    if cum.empty:
        return float("nan")
    n = len(cum)
    y = n / float(trading_days_per_year) if n else 0.0
    end = float(cum.iloc[-1])
    if y <= 0 or end <= 0:
        return float("nan")
    return float(end ** (1.0 / y) - 1.0)
