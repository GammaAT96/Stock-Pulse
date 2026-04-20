from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from dashboard.services.duplicate_pivot_warnings import warn_once_duplicate_trade_symbol_rows
from dashboard.services.prediction_models.feature_prep import DEFAULT_FEATURE_COLS, create_features

logger = logging.getLogger(__name__)

PANEL_FEATURE_COLS: tuple[str, ...] = (*DEFAULT_FEATURE_COLS, "sym_mu", "day_of_week")


def _l1_dollar_neutral(w: pd.DataFrame) -> pd.DataFrame:
    w = w.sub(w.mean(axis=1), axis=0)
    denom = w.abs().sum(axis=1).replace(0.0, np.nan)
    return w.div(denom, axis=0).fillna(0.0)


def _rolling_beta_vs_market(ret_wide: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-asset rolling beta vs cross-sectional mean return (market proxy), lagged one bar."""
    bw = max(10, int(window))
    mkt = ret_wide.mean(axis=1)
    var_m = mkt.rolling(bw, min_periods=max(5, bw // 4)).var().replace(0.0, np.nan)
    cols = {}
    for c in ret_wide.columns:
        cov_im = ret_wide[c].rolling(bw, min_periods=max(5, bw // 4)).cov(mkt)
        cols[c] = (cov_im / var_m).shift(1)
    return pd.DataFrame(cols, index=ret_wide.index)


def _beta_neutralize_weights(w: pd.DataFrame, beta: pd.DataFrame) -> pd.DataFrame:
    """Project rows so sum(w * beta) ≈ 0 (within finite precision)."""
    b = beta.reindex_like(w).fillna(0.0)
    num = (w * b).sum(axis=1)
    den = (b * b).sum(axis=1).replace(0.0, np.nan)
    k = num / den
    k = k.fillna(0.0)
    return w.sub(b.mul(k, axis=0), axis=0)


def _evolve_weights_executed(
    w_target: pd.DataFrame,
    *,
    cost_per_trade: float | None,
    max_turnover: float | None,
    no_trade_band: float | None = None,
) -> pd.DataFrame:
    """
    Day-by-day executed weights: optional **no-trade band** on ``Δw``, L1 proximal shrink, then ‖Δw‖₁ cap.

    ``w_target`` rows are **desired** dollar-neutral targets (e.g. after rebalance hold); ``w`` tracks
    positions carried with friction vs **previous executed** weights (not vs yesterday's target alone).
    """
    tc = float(cost_per_trade) if cost_per_trade is not None and float(cost_per_trade) > 0.0 else None
    cap = float(max_turnover) if max_turnover is not None and float(max_turnover) > 0.0 else None
    band = float(no_trade_band) if no_trade_band is not None and float(no_trade_band) > 0.0 else None
    if tc is None and cap is None and band is None:
        return w_target

    cols = w_target.columns
    w_cur = pd.Series(0.0, index=cols, dtype=float)
    rows: list[np.ndarray] = []
    for _i, d in enumerate(w_target.index):
        tgt = w_target.loc[d].reindex(cols).astype(np.float64).fillna(0.0)
        delta = (tgt - w_cur).to_numpy(dtype=np.float64)
        if band is not None:
            delta = np.where(np.abs(delta) <= band, 0.0, delta)
        if tc is not None:
            adj = np.sign(delta) * np.maximum(np.abs(delta) - tc, 0.0)
        else:
            adj = delta
        w_new = w_cur + pd.Series(adj, index=cols)
        if cap is not None:
            d2 = w_new - w_cur
            tv = float(d2.abs().sum())
            if tv > cap and tv > 0.0:
                w_new = w_cur + d2 * (cap / tv)
        w_cur = w_new
        rows.append(w_cur.to_numpy(dtype=np.float64).copy())
    return pd.DataFrame(rows, index=w_target.index, columns=cols)


def _vol_target_leverage(
    port_ret: pd.Series,
    *,
    vol_window: int = 20,
    target_vol_annual: float = 0.12,
    max_leverage: float = 2.0,
) -> pd.Series:
    """Causal leverage: prior-bar scale from rolling realized vol (annualized), clipped."""
    pr = pd.to_numeric(port_ret, errors="coerce")
    vw = max(5, int(vol_window))
    mp = max(3, vw // 4)
    rv = pr.rolling(vw, min_periods=mp).std() * np.sqrt(252.0)
    lev = target_vol_annual / rv.replace(0.0, np.nan)
    lev = lev.replace([np.inf, -np.inf], np.nan).clip(upper=float(max_leverage)).shift(1).fillna(1.0)
    return lev.fillna(1.0)


def cross_sectional_ic_spearman_by_date(pred_wide: pd.DataFrame, ret_wide: pd.DataFrame) -> pd.Series:
    """Per-date cross-sectional Spearman IC using **lagged** predictions vs same-date returns (aligned with portfolio)."""
    pred_wide = pred_wide.sort_index()
    ret_wide = ret_wide.sort_index()
    p = pred_wide.shift(1)
    idx = p.index.intersection(ret_wide.index)
    out: list[float] = []
    for d in idx:
        row_p = p.loc[d].dropna()
        if row_p.empty:
            out.append(float("nan"))
            continue
        row_r = ret_wide.loc[d].reindex(row_p.index).astype(np.float64)
        m = row_p.notna() & row_r.notna()
        if int(m.sum()) < 4:
            out.append(float("nan"))
            continue
        ic = float(row_p.loc[m].corr(row_r.loc[m], method="spearman"))
        out.append(ic)
    return pd.Series(out, index=idx, dtype=np.float64, name="ic_cs")


def build_ml_panel(symbols: list[str], fetch_fn: Callable[[str], pd.DataFrame]) -> pd.DataFrame:
    """
    Stack per-symbol feature rows into one long panel.

    Adds **sym_mu** (causal expanding mean of ``target`` within symbol — target encoding without leakage)
    and **day_of_week**.
    """
    parts: list[pd.DataFrame] = []
    for sym in symbols:
        raw = fetch_fn(sym)
        if raw is None or raw.empty:
            continue
        f = create_features(raw)
        if f.empty:
            continue
        g = f.copy()
        g["symbol"] = sym
        parts.append(g)
    if not parts:
        return pd.DataFrame()
    panel = pd.concat(parts, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel = panel.dropna(subset=["trade_date"])
    panel = panel.sort_values(["symbol", "trade_date"])
    panel["sym_mu"] = panel.groupby("symbol", sort=False)["target"].transform(lambda s: s.expanding().mean().shift(1))
    panel["sym_mu"] = pd.to_numeric(panel["sym_mu"], errors="coerce").fillna(0.0)
    panel["day_of_week"] = panel["trade_date"].dt.dayofweek.astype(np.float64)
    return panel.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def panel_expanding_date_walk_forward(
    panel: pd.DataFrame,
    model_func: Callable[[pd.DataFrame, list[str] | tuple[str, ...]], object],
    predict_func: Callable[[object, pd.DataFrame, list[str] | tuple[str, ...]], pd.Series],
    feature_cols: list[str] | tuple[str, ...],
    *,
    min_train_dates: int,
    test_chunk_dates: int,
    step_dates: int,
    embargo_trading_days: int = 0,
) -> tuple[pd.DataFrame, list[object]]:
    """
    Train one model on all (symbol, date) rows with trade_date in the expanding past window;
    predict the next calendar chunk of dates. Repeats advancing by step_dates.

    **embargo_trading_days** (López de Prado–style): exclude panel rows with
    ``trade_date >= test_start - embargo`` from training so labels near the test window
    do not overlap the test period (use 2–5 for a light embargo).
    """
    if panel.empty or "trade_date" not in panel.columns:
        raise ValueError("Empty panel or missing trade_date.")
    dates = np.array(sorted(panel["trade_date"].unique()))
    n_d = len(dates)
    if min_train_dates < 20 or test_chunk_dates < 1 or step_dates < 1:
        raise ValueError("Invalid walk-forward date parameters.")
    if min_train_dates + test_chunk_dates > n_d:
        raise ValueError("min_train_dates + test_chunk_dates exceeds available calendar length.")

    chunks: list[pd.DataFrame] = []
    models: list[object] = []
    i = int(min_train_dates)
    while i < n_d:
        end = min(i + int(test_chunk_dates), n_d)
        if end <= i:
            break
        test_dates = dates[i:end]
        test_start = dates[i]
        emb = max(0, int(embargo_trading_days))
        if emb > 0:
            cutoff = pd.Timestamp(test_start) - BDay(emb)
            train_df = panel[panel["trade_date"] < cutoff].copy()
            nu = int(train_df["trade_date"].nunique())
            need = max(40, int(min_train_dates) - emb)
            if nu < need:
                raise ValueError(
                    f"Embargo ({emb} B-days): only {nu} unique training dates before cutoff (need ≥ {need}). "
                    "Reduce embargo_trading_days or lower min_train_dates."
                )
        else:
            train_dates = dates[:i]
            train_df = panel[panel["trade_date"].isin(train_dates)].copy()
        test_df = panel[panel["trade_date"].isin(test_dates)].copy()
        if train_df.empty or test_df.empty:
            break
        model = model_func(train_df, feature_cols)
        test_df = test_df.copy()
        test_df["prediction"] = predict_func(model, test_df, feature_cols)
        chunks.append(test_df)
        models.append(model)
        i += int(step_dates)

    if not chunks:
        raise ValueError("No panel OOS chunks produced.")
    oos = pd.concat(chunks, ignore_index=True).sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return oos, models


def pivot_predictions_and_returns(oos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide matrices indexed by trade_date, columns symbol."""
    need = ["trade_date", "symbol", "prediction", "daily_return"]
    miss = [c for c in need if c not in oos.columns]
    if miss:
        raise ValueError(f"pivot_predictions_and_returns missing columns: {miss}")
    work = oos[need].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work = work.dropna(subset=["trade_date", "symbol"])
    work = work.sort_values(["trade_date", "symbol"])
    warn_once_duplicate_trade_symbol_rows(logger, "pivot_predictions_and_returns", work)
    work = work.drop_duplicates(subset=["trade_date", "symbol"], keep="last")
    pred = work.pivot_table(
        index="trade_date", columns="symbol", values="prediction", aggfunc="last"
    )
    ret = work.pivot_table(
        index="trade_date", columns="symbol", values="daily_return", aggfunc="last"
    )
    common = pred.index.intersection(ret.index)
    pred = pred.loc[common].sort_index()
    ret = ret.loc[common].sort_index()
    return pred, ret


def long_short_rank_portfolio_returns(
    pred_wide: pd.DataFrame,
    ret_wide: pd.DataFrame,
    *,
    top_n: int,
    bottom_n: int,
    demean_cs: bool = False,
) -> pd.DataFrame:
    """
    Dollar-neutral long top_n / short bottom_n by prior-day cross-sectional prediction rank.

    Optional **demean_cs** cross-sectionally demeans weights each day (approx. market-neutral tilt).
    """
    if top_n < 1 or bottom_n < 1:
        raise ValueError("top_n and bottom_n must be positive.")
    pred_wide = pred_wide.sort_index()
    ret_wide = ret_wide.sort_index()
    idx = pred_wide.index.intersection(ret_wide.index)
    pred_wide = pred_wide.loc[idx]
    ret_wide = ret_wide.loc[idx]
    if len(idx) < 3:
        raise ValueError("Not enough dates for rank portfolio.")

    rets: list[float] = []
    long_pnls: list[float] = []
    short_pnls: list[float] = []
    out_idx: list[pd.Timestamp] = []
    pred_lag = pred_wide.shift(1)

    for i, d in enumerate(idx):
        if i == 0:
            continue
        row_p = pred_lag.loc[d].dropna()
        if len(row_p) < top_n + bottom_n:
            continue
        asc = row_p.sort_values(ascending=True)
        short_syms = asc.index[:bottom_n].tolist()
        long_syms = asc.index[-top_n:].tolist()
        w = pd.Series(0.0, index=pred_wide.columns, dtype=float)
        w.loc[long_syms] = 1.0 / float(top_n)
        w.loc[short_syms] = -1.0 / float(bottom_n)
        if demean_cs:
            w = w - w.mean()
        r = ret_wide.loc[d].reindex(pred_wide.columns)
        pr = float((w * r.fillna(0.0)).sum())
        rets.append(pr)
        long_pnls.append(float((w.clip(lower=0.0) * r.fillna(0.0)).sum()))
        short_pnls.append(float((w.clip(upper=0.0) * r.fillna(0.0)).sum()))
        out_idx.append(d)

    out = pd.DataFrame(
        {
            "trade_date": out_idx,
            "strategy_return": rets,
            "long_attribution": long_pnls,
            "short_attribution": short_pnls,
        }
    )
    out["cum_strategy_return"] = (1.0 + out["strategy_return"]).cumprod()
    return out


def score_quantile_ls_portfolio(
    pred_wide: pd.DataFrame,
    ret_wide: pd.DataFrame,
    *,
    long_q: float = 0.8,
    short_q: float = 0.2,
    demean_cs: bool = True,
    max_turnover: float | None = None,
    cost_per_trade: float | None = None,
    beta_neutral: bool = False,
    beta_window: int = 60,
    rebalance_every_k: int = 1,
    coverage_weight_returns: bool = False,
    no_trade_band: float | None = None,
    vol_target: bool = False,
    target_vol_annual: float = 0.12,
    vol_target_window: int = 20,
    max_vol_target_leverage: float = 2.0,
) -> dict[str, pd.DataFrame | pd.Series | float]:
    """
    Score-weighted long/short: lagged preds, optional rebalance cadence, L1 dollar-neutral weights.

    Portfolio return uses **prior-day weights** × same-day returns: ``(w.shift(1) * r).sum(axis=1)``.

    - **cost_per_trade**: soft-threshold on ``Δw`` (proximal / L1 shrink toward zero trade), then optional turnover cap.
    - **no_trade_band**: per-name deadband on ``|Δw|`` before proximal (hysteresis; cuts small churn).
    - **beta_neutral**: project weights orthogonal to rolling **beta vs CS mean** (lagged).
    - **rebalance_every_k**: hold target weights fixed between rebalance days (forward-filled).
    - **vol_target**: scale returns toward ``target_vol_annual`` using rolling realized vol (causal ``shift(1)`` leverage).
    """
    if not 0.0 < short_q < long_q < 1.0:
        raise ValueError("Require 0 < short_q < long_q < 1.")
    pred_wide = pred_wide.sort_index()
    ret_wide = ret_wide.sort_index()
    idx = pred_wide.index.intersection(ret_wide.index)
    pred_wide = pred_wide.loc[idx]
    ret_wide = ret_wide.loc[idx]
    scores = pred_wide.shift(1)
    q_hi = scores.quantile(long_q, axis=1)
    q_lo = scores.quantile(short_q, axis=1)
    long_raw = scores.where(scores.ge(q_hi, axis=0), 0.0)
    short_raw = scores.where(scores.le(q_lo, axis=0), 0.0)
    w_star = long_raw - short_raw
    denom0 = w_star.abs().sum(axis=1).replace(0.0, np.nan)
    w_star = w_star.div(denom0, axis=0).fillna(0.0)
    if demean_cs:
        w_star = w_star.sub(w_star.mean(axis=1), axis=0)

    k = max(1, int(rebalance_every_k))
    if k > 1:
        arr = w_star.to_numpy(dtype=float)
        out_a = np.zeros_like(arr)
        hold = np.zeros(arr.shape[1])
        for i in range(len(arr)):
            if i % k == 0:
                hold = arr[i].copy()
            out_a[i] = hold
        w_star = pd.DataFrame(out_a, index=w_star.index, columns=w_star.columns)

    w_star = _l1_dollar_neutral(w_star)

    w = _evolve_weights_executed(
        w_star,
        cost_per_trade=cost_per_trade,
        max_turnover=max_turnover,
        no_trade_band=no_trade_band,
    )

    if beta_neutral:
        beta = _rolling_beta_vs_market(ret_wide, beta_window)
        w = _beta_neutralize_weights(w, beta)

    w = _l1_dollar_neutral(w)

    w_turn = w.diff().abs().sum(axis=1).fillna(0.0)
    avg_weight_turnover = float(w_turn.mean()) if len(w_turn) else float("nan")
    avg_dollar_turnover = float(0.5 * w_turn.mean()) if len(w_turn) else float("nan")
    gross_row = w.abs().sum(axis=1)
    net_row = w.sum(axis=1)
    avg_gross_exposure = float(gross_row.mean()) if len(gross_row) else float("nan")
    avg_net_exposure = float(net_row.mean()) if len(net_row) else float("nan")
    tcpv = float(cost_per_trade) if cost_per_trade is not None and float(cost_per_trade) > 0.0 else None
    implied_daily_cost = (
        float(avg_dollar_turnover * tcpv)
        if tcpv is not None and avg_dollar_turnover == avg_dollar_turnover
        else float("nan")
    )

    w_exec = w.shift(1).fillna(0.0)
    strat_pre_vol = (w_exec * ret_wide).sum(axis=1)
    leverage = pd.Series(1.0, index=strat_pre_vol.index, dtype=np.float64)
    if vol_target:
        leverage = _vol_target_leverage(
            strat_pre_vol,
            vol_window=int(vol_target_window),
            target_vol_annual=float(target_vol_annual),
            max_leverage=float(max_vol_target_leverage),
        )
    strat = strat_pre_vol * leverage
    coverage = (~pred_wide.isna()).mean(axis=1).reindex(strat.index).fillna(0.0)
    if coverage_weight_returns:
        strat = strat * coverage

    long_pnl = (w_exec.clip(lower=0.0) * ret_wide).sum(axis=1) * leverage
    short_pnl = (w_exec.clip(upper=0.0) * ret_wide).sum(axis=1) * leverage
    contrib = w_exec * ret_wide
    hit_rate = float((contrib > 0.0).stack().dropna().mean()) if not contrib.empty else float("nan")

    qt = w.abs().quantile(0.8, axis=1)
    topk = w.abs().ge(qt, axis=0).astype(float)
    topk_pnl = (topk * w_exec * ret_wide).sum(axis=1) * leverage
    rest_pnl = strat - topk_pnl

    strat_clean = strat.fillna(0.0)
    long_clean = long_pnl.fillna(0.0)
    short_clean = short_pnl.fillna(0.0)
    topk_clean = topk_pnl.fillna(0.0)
    rest_clean = rest_pnl.fillna(0.0)
    result = pd.DataFrame(
        {
            "trade_date": strat_clean.index,
            "strategy_return": strat_clean.to_numpy(),
            "long_attribution": long_clean.to_numpy(),
            "short_attribution": short_clean.to_numpy(),
            "topk_attribution": topk_clean.to_numpy(),
            "rest_attribution": rest_clean.to_numpy(),
        }
    )
    result["strategy_return_pre_vol"] = strat_pre_vol.reindex(result["trade_date"]).fillna(0.0).to_numpy()
    if vol_target:
        result["vol_target_leverage"] = leverage.reindex(result["trade_date"]).fillna(1.0).to_numpy()
    result["cum_strategy_return"] = (1.0 + result["strategy_return"]).cumprod()
    return {
        "result": result,
        "coverage": coverage,
        "hit_rate": hit_rate,
        "avg_coverage": float(coverage.mean()) if len(coverage) else float("nan"),
        "avg_weight_turnover": avg_weight_turnover,
        "avg_dollar_turnover": avg_dollar_turnover,
        "avg_gross_exposure": avg_gross_exposure,
        "avg_net_exposure": avg_net_exposure,
        "implied_daily_cost_drag": implied_daily_cost,
        "vol_target": bool(vol_target),
        "target_vol_annual": float(target_vol_annual) if vol_target else float("nan"),
        "weights_last": w.iloc[-1] if len(w) else pd.Series(dtype=float),
    }
