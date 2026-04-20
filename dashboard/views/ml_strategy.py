from __future__ import annotations

import math
from typing import Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.services.data_service import fetch_symbols, fetch_stock_data
from dashboard.services.portfolio_service import _daily_return_series_prefer_varying_close
from dashboard.services.prediction_models.feature_prep import DEFAULT_FEATURE_COLS, create_features
from dashboard.services.prediction_models.cross_sectional_ml import (
    PANEL_FEATURE_COLS,
    build_ml_panel,
    cross_sectional_ic_spearman_by_date,
    long_short_rank_portfolio_returns,
    panel_expanding_date_walk_forward,
    pivot_predictions_and_returns,
    score_quantile_ls_portfolio,
)
from dashboard.services.prediction_models.ml_backtest import (
    ConfidenceScale,
    PositionStyle,
    SignalMode,
    ThresholdMode,
    attach_ml_strategy_returns,
    cagr_from_cum,
    max_drawdown_from_cum,
    realized_turnover,
    ic_half_life_days,
    rebalance_k_from_ic_half_life,
    rolling_information_coefficient,
    sharpe_daily,
    underwater_equity_curve,
)
from dashboard.services.prediction_models.ml_portfolio import (
    aggregate_feature_importances,
    cross_sectional_equal_weight,
    cross_sectional_inverse_vol_weights,
    cross_sectional_inverse_vol_weights_rolling,
    portfolio_metrics_from_returns,
    returns_correlation_matrix,
)
from dashboard.services.prediction_models.model_evaluator import evaluate_model
from dashboard.services.prediction_models.random_forest import predict_rf, train_rf
from dashboard.services.prediction_models.walk_forward import rolling_walk_forward_predict, walk_forward_predict
from dashboard.services.prediction_models.xgboost_model import predict_xgb, train_xgb, xgboost_available
from dashboard.views.stock_analysis import _pipeline_df_view


def _annualize_sharpe(daily_sharpe: float, trading_days: int = 252) -> float:
    if daily_sharpe is None or math.isnan(daily_sharpe):
        return float("nan")
    return float(daily_sharpe * math.sqrt(trading_days))


def _model_trainers() -> dict[str, Callable[[pd.DataFrame, tuple[str, ...]], object]]:
    trainers: dict[str, Callable[[pd.DataFrame, tuple[str, ...]], object]] = {
        "RandomForest": train_rf,
    }
    if xgboost_available():
        trainers["XGBoost"] = train_xgb
    return trainers


def _predict_series(model, name: str, df: pd.DataFrame, feature_cols: tuple[str, ...]) -> pd.Series:
    if name == "RandomForest":
        return predict_rf(model, df, feature_cols)
    if name == "XGBoost":
        return predict_xgb(model, df, feature_cols)
    raise ValueError(f"Unknown model: {name}")


def _run_walk_forward(
    feats: pd.DataFrame,
    *,
    wf_mode: str,
    train_frac: float,
    rolling_params: tuple[int, int, int] | None,
    train_fn: Callable[[pd.DataFrame, tuple[str, ...]], object],
    predict_fn: Callable[[object, pd.DataFrame, list[str] | tuple[str, ...]], pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, object | None, list[object]]:
    if wf_mode.startswith("Single"):
        train, test, model, models = walk_forward_predict(
            feats,
            train_fn,
            predict_fn,
            DEFAULT_FEATURE_COLS,
            train_size=float(train_frac),
        )
        return train, test, model, models
    assert rolling_params is not None
    mtr, chunk, stp = rolling_params
    train, test, model, models = rolling_walk_forward_predict(
        feats,
        train_fn,
        predict_fn,
        DEFAULT_FEATURE_COLS,
        min_train_rows=mtr,
        test_chunk=chunk,
        step=stp,
    )
    return train, test, model, models


def show():
    st.title("ML Strategy (walk-forward)")
    st.caption(
        "Chronological validation, next-bar execution, optional risk-free carry on flat cash, "
        "head-to-head vs SMA, and optional cross-sectional ML portfolio."
    )

    trainers = _model_trainers()
    if "XGBoost" not in trainers:
        st.info("XGBoost is not installed in this environment; only Random Forest is available.")

    symbols = fetch_symbols()
    symbol = st.selectbox("Primary symbol (charts & single-name ML)", symbols, index=0 if symbols else None)
    model_names = st.multiselect(
        "Models",
        options=list(trainers.keys()),
        default=[list(trainers.keys())[0]],
    )

    wf_mode = st.radio(
        "Walk-forward mode",
        ["Single split (baseline)", "Rolling expanding window"],
        horizontal=True,
    )
    if wf_mode.startswith("Rolling"):
        st.caption(
            "Rolling mode: models are refit each step; feature importances aggregate **mean ± std** "
            "across folds when multiple fits exist."
        )

    train_frac_single = 0.7
    if wf_mode.startswith("Single"):
        train_frac_single = st.slider("Train fraction (initial)", min_value=0.55, max_value=0.90, value=0.70, step=0.01)
        rolling_params = None
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            min_train_rows = st.number_input(
                "Min training rows before first OOS chunk",
                min_value=60,
                max_value=2000,
                value=252,
                step=1,
            )
        with c2:
            test_chunk = st.number_input("OOS chunk size (bars)", min_value=5, max_value=260, value=21, step=1)
        with c3:
            step = st.number_input("Roll step (bars)", min_value=1, max_value=260, value=21, step=1)
        rolling_params = (int(min_train_rows), int(test_chunk), int(step))

    thr_mode_label = st.selectbox(
        "Threshold / signal cut",
        [
            "Static (fixed decimal hurdle)",
            "Quantile on OOS predictions (full-window calibration)",
            "Quantile from reference preds (train / last-train window)",
            "Expanding quantile on OOS path (causal: past-only + lag)",
        ],
    )
    if thr_mode_label.startswith("Expanding"):
        threshold_mode: ThresholdMode = "quantile_expanding"
        st.caption(
            "Expanding quantile: cutoffs use **only predictions observed so far**, shifted one row "
            "before comparing — no peeking at future OOS preds."
        )
    elif thr_mode_label.startswith("Quantile on OOS"):
        threshold_mode = "quantile_oos"
        st.caption(
            "OOS quantile uses the **entire** out-of-sample prediction column to set hi/lo cutoffs "
            "(useful for research; not strictly causal day-by-day)."
        )
    elif thr_mode_label.startswith("Quantile from reference"):
        threshold_mode = "quantile_ref"
    else:
        threshold_mode = "static"

    q_exp_min = 40
    if threshold_mode == "quantile_expanding":
        q_exp_min = st.number_input("Expanding quantile min periods (bars)", 10, 500, 40, step=5)

    qcol1, qcol2 = st.columns(2)
    with qcol1:
        quantile_hi = st.slider("Upper quantile (long cut)", 0.55, 0.95, 0.70, step=0.01)
    with qcol2:
        quantile_lo = st.slider("Lower quantile (short cut, long/short only)", 0.05, 0.45, 0.30, step=0.01)

    if float(quantile_lo) >= float(quantile_hi):
        st.error("Lower quantile must be strictly less than upper quantile.")
        return

    thr_col1, thr_col2 = st.columns(2)
    with thr_col1:
        pred_threshold = st.slider(
            "Static prediction hurdle (decimal return)",
            0.0,
            0.01,
            0.002,
            step=0.0001,
            format="%.4f",
            disabled=(threshold_mode != "static"),
        )
    with thr_col2:
        signal_mode_label = st.selectbox(
            "Signal style",
            ["Long only (flat below threshold)", "Long / short / flat (symmetric thresholds)"],
        )
    signal_mode: SignalMode = "long_short_flat" if "short" in signal_mode_label.lower() else "long_only"

    pos_style_label = st.selectbox(
        "Position sizing",
        ["Discrete (0/1 or -1/0/1)", "Confidence-weighted (fractional exposure, lagged)"],
    )
    position_style: PositionStyle = "confidence" if "Confidence" in pos_style_label else "discrete"

    conf_scale_label = st.selectbox(
        "Confidence scale (when using confidence-weighted positions)",
        ["OOS max |prediction| (legacy)", "Rolling max |prediction| (causal, lagged scale)"],
    )
    confidence_scale: ConfidenceScale = "rolling_max" if "Rolling" in conf_scale_label else "oos_max"
    conf_roll_w = st.slider("Rolling max window (bars)", 10, 200, 50, step=5, disabled=(confidence_scale != "rolling_max"))

    sma_col1, sma_col2 = st.columns(2)
    with sma_col1:
        sma_short_n = st.slider("SMA benchmark — short", 5, 50, 20)
    with sma_col2:
        long_lo = max(20, sma_short_n + 1)
        sma_long_n = st.slider("SMA benchmark — long", long_lo, 120, max(50, long_lo))

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        include_rf = st.checkbox("Include risk-free on cash (same framing as SMA tab)", value=False)
    with col_b:
        if include_rf:
            rf_annual = st.number_input(
                "Risk-free (annual, decimal)",
                min_value=0.0,
                max_value=0.2,
                value=float(st.session_state.get("ml_strategy_rf_annual", 0.05)),
                step=0.005,
                key="ml_strategy_rf_annual",
            )
        else:
            # Widget hidden when cash carry is off; keep last value for if user re-enables.
            rf_annual = float(st.session_state.get("ml_strategy_rf_annual", 0.05))
    with col_c:
        cost_pct = st.number_input("Round-trip cost per unit position change (decimal)", min_value=0.0, max_value=0.05, value=0.0, step=0.0001, format="%.4f")

    ic_window = st.slider("Rolling IC window (bars)", 20, 120, 50, step=5)

    if not symbols or symbol is None:
        st.warning("No symbols found in `stock_prices_gold`.")
        return

    raw = fetch_stock_data(symbol)
    if raw.empty:
        st.warning("No rows returned for that symbol.")
        return

    feats = create_features(raw)
    if len(feats) < 120:
        st.warning("Not enough clean history after feature engineering (need more non-NaN rows).")
        return

    if not model_names:
        st.warning("Pick at least one model.")
        return

    if wf_mode.startswith("Rolling") and rolling_params is not None:
        mtr, _, _ = rolling_params
        if mtr >= len(feats):
            st.warning("min_train_rows must be smaller than the feature row count.")
            return

    rows: list[dict[str, float | str]] = []
    fig_eq = go.Figure()
    fig_hist = go.Figure()
    fig_ic = go.Figure()
    fig_dd = go.Figure()
    importance_mean: dict[str, pd.Series] = {}
    importance_std: dict[str, pd.Series] = {}
    describe_blocks: list[str] = []
    ic_avg_by_model: dict[str, float] = {}
    ic_hl_by_model: dict[str, float] = {}

    mkt_sh = float("nan")
    mkt_cagr = float("nan")
    mkt_dd = float("nan")
    sma_on_chart = False
    oos_start = None
    oos_end = None

    for name in model_names:
        train_fn = trainers[name]

        def _train(train_df: pd.DataFrame, feature_cols: list[str] | tuple[str, ...], _train_fn=train_fn):
            return _train_fn(train_df, tuple(feature_cols))

        def _predict(model, df: pd.DataFrame, feature_cols: list[str] | tuple[str, ...], _name=name) -> pd.Series:
            return _predict_series(model, _name, df, tuple(feature_cols))

        try:
            train, test, model, models = _run_walk_forward(
                feats,
                wf_mode=wf_mode,
                train_frac=float(train_frac_single),
                rolling_params=rolling_params,
                train_fn=_train,
                predict_fn=_predict,
            )
        except ValueError as e:
            st.error(str(e))
            return

        if oos_start is None:
            oos_start = pd.to_datetime(test["trade_date"]).min().date()
            oos_end = pd.to_datetime(test["trade_date"]).max().date()

        ref_preds: pd.Series | None = None
        if threshold_mode == "quantile_ref":
            ref_model = model
            ref_train = train
            if ref_model is None or ref_train is None or ref_train.empty:
                st.error("Cannot build reference quantiles: empty train or model.")
                return
            ref_preds = _predict_series(ref_model, name, ref_train, DEFAULT_FEATURE_COLS)

        mean_imp, std_imp = aggregate_feature_importances(models, DEFAULT_FEATURE_COLS)
        if mean_imp is not None:
            importance_mean[name] = mean_imp
            if std_imp is not None:
                importance_std[name] = std_imp

        describe_blocks.append(f"**{name}** predictions (OOS):\n{test['prediction'].describe().to_string()}")

        metrics = evaluate_model(test)
        bt = attach_ml_strategy_returns(
            test,
            rf_annual=float(rf_annual),
            include_rf=bool(include_rf),
            transaction_cost_pct=float(cost_pct),
            prediction_threshold=float(pred_threshold),
            signal_mode=signal_mode,
            threshold_mode=threshold_mode,
            quantile_hi=float(quantile_hi),
            quantile_lo=float(quantile_lo),
            quantile_expanding_min_periods=int(q_exp_min),
            reference_predictions=ref_preds,
            position_style=position_style,
            confidence_scale=confidence_scale,
            confidence_rolling_window=int(conf_roll_w),
        )

        ric = rolling_information_coefficient(
            bt["prediction"], bt["target"], window=int(ic_window), min_periods=max(10, int(ic_window) // 3)
        )
        ic_avg_by_model[name] = float(ric.mean(skipna=True)) if len(ric) else float("nan")
        ic_hl_by_model[name] = ic_half_life_days(ric)
        fig_ic.add_trace(
            go.Scatter(
                x=bt["trade_date"],
                y=ric,
                name=f"{name} rolling IC",
                mode="lines",
            )
        )
        uw = underwater_equity_curve(bt["strategy_return"])
        fig_dd.add_trace(go.Scatter(x=bt["trade_date"], y=uw, name=f"{name} underwater", mode="lines"))

        strat_sh = sharpe_daily(bt["strategy_return"])
        turn = realized_turnover(bt)
        if math.isnan(mkt_sh):
            mkt_sh = sharpe_daily(bt["daily_return"])
            mkt_cagr = cagr_from_cum(bt["cum_market_return"])
            mkt_dd = max_drawdown_from_cum(bt["cum_market_return"])
            fig_eq.add_trace(
                go.Scatter(
                    x=bt["trade_date"],
                    y=bt["cum_market_return"] - 1.0,
                    name="Buy & hold",
                    line=dict(width=2, dash="dot"),
                )
            )
            fig_dd.add_trace(
                go.Scatter(
                    x=bt["trade_date"],
                    y=underwater_equity_curve(bt["daily_return"]),
                    name="Buy & hold underwater",
                    mode="lines",
                    line=dict(dash="dot"),
                )
            )

        strat_cagr = cagr_from_cum(bt["cum_strategy_return"])
        strat_dd = max_drawdown_from_cum(bt["cum_strategy_return"])
        rows.append(
            {
                "Strategy": name,
                "CAGR": strat_cagr,
                "Sharpe (ann.)": _annualize_sharpe(strat_sh),
                "Max DD": strat_dd,
                "Turnover": turn,
                "IC (Spearman, OOS)": metrics["ic_spearman"],
                "RMSE": metrics["rmse"],
            }
        )

        fig_eq.add_trace(go.Scatter(x=bt["trade_date"], y=bt["cum_strategy_return"] - 1.0, name=f"{name} (ML OOS)"))
        fig_hist.add_trace(go.Histogram(x=bt["prediction"], name=f"{name} preds", opacity=0.55, nbinsx=40))

    if oos_start is not None and oos_end is not None:
        sma_view = _pipeline_df_view(
            raw,
            oos_start,
            oos_end,
            int(sma_short_n),
            int(sma_long_n),
            float(cost_pct),
            float(rf_annual),
            bool(include_rf),
        )
        if not sma_view.empty:
            sma_sh = sharpe_daily(sma_view["strategy_return"])
            sma_turn = realized_turnover(sma_view) if "position" in sma_view.columns else float("nan")
            rows.append(
                {
                    "Strategy": f"SMA {sma_short_n}/{sma_long_n}",
                    "CAGR": cagr_from_cum(sma_view["cum_strategy_return"]),
                    "Sharpe (ann.)": _annualize_sharpe(sma_sh),
                    "Max DD": max_drawdown_from_cum(sma_view["cum_strategy_return"]),
                    "Turnover": sma_turn,
                    "IC (Spearman, OOS)": float("nan"),
                    "RMSE": float("nan"),
                }
            )
            fig_eq.add_trace(
                go.Scatter(
                    x=sma_view["trade_date"],
                    y=sma_view["cum_strategy_return"] - 1.0,
                    name=f"SMA {sma_short_n}/{sma_long_n} (rule)",
                    line=dict(width=2),
                )
            )
            sma_on_chart = True
            uw_sma = underwater_equity_curve(sma_view["strategy_return"])
            fig_dd.add_trace(
                go.Scatter(x=sma_view["trade_date"], y=uw_sma, name=f"SMA {sma_short_n}/{sma_long_n} underwater", mode="lines")
            )

    st.subheader("Decision layer — strategy comparison (OOS window)")
    comp = pd.DataFrame(rows)
    if not comp.empty:
        disp = comp.set_index("Strategy")
        st.dataframe(
            disp.style.format(
                {
                    "CAGR": "{:.2%}",
                    "Sharpe (ann.)": "{:.2f}",
                    "Max DD": "{:.2%}",
                    "Turnover": "{:.3f}",
                    "IC (Spearman, OOS)": "{:.3f}",
                    "RMSE": "{:.6f}",
                },
                na_rep="—",
            ),
            use_container_width=True,
        )

    st.subheader("Diagnostics")
    pcols = st.columns(4)
    pcols[0].metric("Buy&Hold CAGR", f"{mkt_cagr:.2%}")
    pcols[1].metric("Buy&Hold Sharpe (ann.)", f"{_annualize_sharpe(mkt_sh):.2f}")
    pcols[2].metric("Buy&Hold max DD", f"{mkt_dd:.2%}")
    pcols[3].metric("SMA on chart", "Yes" if sma_on_chart else "No")

    c1, c2 = st.columns(2)
    with c1:
        fig_eq.update_layout(title="Cumulative return — same calendar window", yaxis_tickformat=".1%")
        st.plotly_chart(fig_eq, use_container_width=True)

    with c2:
        fig_hist.update_layout(barmode="overlay", title="Prediction distribution (OOS)")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Rolling information coefficient (prediction vs next-day return)")
    fig_ic.update_layout(yaxis_title="IC", xaxis_title="Date")
    st.plotly_chart(fig_ic, use_container_width=True)

    if ic_avg_by_model:
        ic_cols = st.columns(len(ic_avg_by_model))
        for i, (mn, av) in enumerate(ic_avg_by_model.items()):
            with ic_cols[i]:
                st.metric(f"Avg rolling IC — {mn}", f"{av:.4f}" if av == av else "—")
                if av == av:
                    st.metric(f"IC×√252 — {mn}", f"{av * math.sqrt(252):.2f}")
                else:
                    st.metric(f"IC×√252 — {mn}", "—")
                hl = ic_hl_by_model.get(mn, float("nan"))
                st.metric(f"IC half-life (bars) — {mn}", f"{hl:.1f}" if hl == hl else "—")

    st.subheader("Drawdown (underwater) curves")
    fig_dd.update_layout(yaxis_title="Drawdown from peak", yaxis_tickformat=".1%", xaxis_title="Date")
    st.plotly_chart(fig_dd, use_container_width=True)

    with st.expander("Prediction summary (`describe`)", expanded=False):
        st.markdown("\n\n".join(describe_blocks))

    if importance_mean:
        st.subheader("Feature importances (mean across walk-forward fits)")
        for mname, ser in importance_mean.items():
            st.markdown(f"**{mname}**")
            st.bar_chart(ser)
            if mname in importance_std and importance_std[mname].notna().any():
                with st.expander(f"{mname} — importance std across folds"):
                    st.bar_chart(importance_std[mname].fillna(0.0))

    with st.expander("Cross-sectional ML portfolio (per-symbol models → combine)", expanded=False):
        st.caption(
            "Runs the **same** ML settings per symbol, aligns OOS strategy returns on common dates, "
            "then combines with **equal weight**, **global inverse-vol**, or **rolling inverse-vol** (causal σ)."
        )
        port_weight = st.radio(
            "Combine leg returns",
            ["Equal weight", "Inverse vol (global σ)", "Inverse vol (rolling σ, causal)"],
            horizontal=True,
        )
        port_vol_win = st.number_input(
            "Rolling vol window (days, rolling-std inv-vol only)",
            min_value=20,
            max_value=260,
            value=60,
            step=5,
            disabled=not port_weight.startswith("Inverse vol (rolling"),
        )
        port_vol_method = st.radio(
            "Rolling σ method (causal inv-vol)",
            ["rolling_std", "ewma"],
            horizontal=True,
            disabled=not port_weight.startswith("Inverse vol (rolling"),
            format_func=lambda x: "Rolling std" if x == "rolling_std" else "EWMA on r²",
        )
        port_ewma_lam = st.slider(
            "EWMA λ (variance decay; σ = √(EWMA[r²]))",
            min_value=0.85,
            max_value=0.995,
            value=0.94,
            step=0.005,
            disabled=not port_weight.startswith("Inverse vol (rolling") or port_vol_method != "ewma",
        )
        port_syms = st.multiselect("Symbols (min 2)", [s for s in symbols if s != symbol] + [symbol], default=[])
        port_syms = list(dict.fromkeys(port_syms))
        if len(port_syms) >= 2:
            if st.button("Build ML portfolio on shared OOS window"):
                rets: dict[str, pd.Series] = {}
                for psym in port_syms:
                    praw = fetch_stock_data(psym)
                    if praw.empty:
                        st.warning(f"No data for {psym}")
                        continue
                    pfeats = create_features(praw)
                    if len(pfeats) < 120:
                        st.warning(f"Skip {psym}: insufficient rows after features.")
                        continue
                    pfn = trainers[model_names[0]]

                    def _ptrain(td: pd.DataFrame, fc: list[str] | tuple[str, ...], _pfn=pfn):
                        return _pfn(td, tuple(fc))

                    def _ppredict(mo, td: pd.DataFrame, fc: list[str] | tuple[str, ...]) -> pd.Series:
                        return _predict_series(mo, model_names[0], td, tuple(fc))

                    try:
                        tr, te, pmodel, _ = _run_walk_forward(
                            pfeats,
                            wf_mode=wf_mode,
                            train_frac=float(train_frac_single),
                            rolling_params=rolling_params,
                            train_fn=_ptrain,
                            predict_fn=_ppredict,
                        )
                    except ValueError:
                        continue
                    ref = None
                    if threshold_mode == "quantile_ref" and pmodel is not None:
                        ref = _predict_series(pmodel, model_names[0], tr, DEFAULT_FEATURE_COLS)
                    btp = attach_ml_strategy_returns(
                        te,
                        rf_annual=float(rf_annual),
                        include_rf=bool(include_rf),
                        transaction_cost_pct=float(cost_pct),
                        prediction_threshold=float(pred_threshold),
                        signal_mode=signal_mode,
                        threshold_mode=threshold_mode,
                        quantile_hi=float(quantile_hi),
                        quantile_lo=float(quantile_lo),
                        quantile_expanding_min_periods=int(q_exp_min),
                        reference_predictions=ref,
                        position_style=position_style,
                        confidence_scale=confidence_scale,
                        confidence_rolling_window=int(conf_roll_w),
                    )
                    idx = pd.to_datetime(btp["trade_date"])
                    rets[psym] = btp.set_index(idx)["strategy_return"]
                if len(rets) >= 2:
                    port_r, panel = cross_sectional_equal_weight(rets)
                    if port_weight.startswith("Inverse vol (global"):
                        port_r, w_iv = cross_sectional_inverse_vol_weights(panel)
                        st.caption(
                            "Inverse-vol weights (constant): "
                            + ", ".join(f"{k}:{v:.3f}" for k, v in w_iv.items())
                        )
                    elif port_weight.startswith("Inverse vol (rolling"):
                        port_r, w_iv = cross_sectional_inverse_vol_weights_rolling(
                            panel,
                            vol_window=int(port_vol_win),
                            min_periods=max(10, int(port_vol_win) // 3),
                            vol_method=port_vol_method,
                            ewma_lambda=float(port_ewma_lam),
                        )
                        _sig = (
                            f"{port_vol_method}, λ={float(port_ewma_lam):.3f}"
                            if port_vol_method == "ewma"
                            else str(port_vol_method)
                        )
                        st.caption(f"Rolling inverse-vol: row weights from **lagged** σ per leg ({_sig}).")
                    pm = portfolio_metrics_from_returns(port_r)
                    st.metric("Portfolio CAGR", f"{pm['cagr']:.2%}")
                    st.metric("Portfolio Sharpe (ann.)", f"{pm['sharpe']:.2f}")
                    st.metric("Portfolio max DD", f"{pm['max_dd']:.2%}")
                    st.metric("Common OOS days", f"{int(pm['n_days'])}")
                    cum = (1.0 + port_r).cumprod()
                    figp = go.Figure()
                    if port_weight.startswith("Equal"):
                        lbl = "EW"
                    elif port_weight.startswith("Inverse vol (rolling"):
                        lbl = "Inv vol (roll)"
                    else:
                        lbl = "Inv vol (glob)"
                    figp.add_trace(go.Scatter(x=port_r.index, y=cum - 1.0, name=f"ML portfolio ({lbl})"))
                    figp.update_layout(title=f"Cross-sectional ML — {lbl}", yaxis_tickformat=".1%")
                    st.plotly_chart(figp, use_container_width=True)
                    st.dataframe(panel.tail(8), use_container_width=True)
                else:
                    st.warning("Need at least two symbols with valid OOS paths.")

    with st.expander("Panel ML — pooled model + cross-sectional portfolio", expanded=False):
        st.caption(
            "Trains the first selected model on stacked rows with **sym_mu** (causal target encoding) + **day_of_week** "
            "and the default feature set. Returns use **lagged** predictions vs same-day realized returns. "
            "Choose **bucket ranks** or **score-quantile** sleeves; optional **CS demean**, **L1 proximal costs**, "
            "**no-trade band**, **beta-neutral** projection, **manual or IC-driven rebalance k**, and **turnover cap** on executed weights."
        )
        panel_syms = st.multiselect(
            "Universe",
            symbols,
            default=symbols[: min(8, len(symbols))],
        )
        panel_syms = list(dict.fromkeys(panel_syms))
        p_mode = st.radio(
            "Cross-sectional book",
            ["Bucket rank (top N / bottom N)", "Score quantile L/S (weighted)"],
            horizontal=True,
        )
        p_top = st.number_input("Long top N (bucket mode)", 1, 15, 2)
        p_bot = st.number_input("Short bottom N (bucket mode)", 1, 15, 2)
        pq_hi = st.slider("Long quantile (score mode)", 0.55, 0.95, 0.80, step=0.01, disabled=p_mode.startswith("Bucket"))
        pq_lo = st.slider("Short quantile (score mode)", 0.05, 0.45, 0.20, step=0.01, disabled=p_mode.startswith("Bucket"))
        p_demean = st.checkbox("Cross-sectionally demean weights (market-neutral tilt)", value=True)
        p_turn = st.number_input("Max daily ‖Δw‖₁ turnover cap (0 = off)", 0.0, 2.0, 0.0, step=0.05)
        p_cost_prox = st.checkbox("Cost-aware trades (L1 proximal on Δw)", value=False)
        p_tc = st.number_input(
            "Per-name trade cost (L1 weight units, e.g. 0.001)",
            0.0,
            0.02,
            0.001,
            step=0.0001,
            format="%.4f",
            disabled=not p_cost_prox,
        )
        p_beta = st.checkbox("Beta-neutralize vs cross-sectional mean (rolling β)", value=False)
        p_beta_w = st.slider("β window (days)", 40, 120, 60, step=5, disabled=not p_beta)
        p_auto_ic_rebal = st.checkbox(
            "Auto rebalance k from IC half-life (k = round(half-life/2), clipped to 1–10)",
            value=False,
            disabled=p_mode.startswith("Bucket"),
        )
        p_rebal_k = st.slider(
            "Rebalance target weights every k days (1 = daily)",
            1,
            20,
            1,
            disabled=p_mode.startswith("Bucket") or p_auto_ic_rebal,
        )
        p_band = st.slider(
            "No-trade band: ignore |Δw| per name below this (hysteresis, L1 weight units)",
            0.0,
            0.02,
            0.0,
            step=0.0005,
            format="%.4f",
            disabled=p_mode.startswith("Bucket"),
        )
        p_cov_w = st.checkbox("Weight strategy returns by prediction coverage", value=False)
        p_vol_tgt = st.checkbox("Vol targeting (12% ann., causal leverage cap 2×)", value=False, disabled=p_mode.startswith("Bucket"))
        p_embargo = st.number_input(
            "Train embargo before each OOS chunk (trading days, 0 = off)",
            0,
            10,
            0,
            help="Drop training rows with trade_date in the embargo window before the test chunk (reduces label overlap).",
        )
        p_min_dates = st.number_input("Min calendar history (days) before first OOS", 60, 800, 252)
        p_chunk = st.number_input("OOS chunk (trading days)", 5, 120, 21)
        p_step = st.number_input("Roll step (trading days)", 1, 120, 21)
        need_syms = int(p_top) + int(p_bot) if p_mode.startswith("Bucket") else 3
        if len(panel_syms) >= need_syms and st.button("Run panel cross-sectional ML"):
            panel = build_ml_panel(panel_syms, fetch_stock_data)
            if panel.empty or len(panel) < 500:
                st.warning("Panel too small after stacking; pick more symbols or check data.")
            else:
                pm_name = model_names[0]
                pfn = trainers[pm_name]

                def _pmtrain(td: pd.DataFrame, fc: list[str] | tuple[str, ...], _pfn=pfn):
                    return _pfn(td, tuple(fc))

                def _pmpredict(mo, td: pd.DataFrame, fc: list[str] | tuple[str, ...]) -> pd.Series:
                    return _predict_series(mo, pm_name, td, tuple(fc))

                try:
                    oos_p, _pmods = panel_expanding_date_walk_forward(
                        panel,
                        _pmtrain,
                        _pmpredict,
                        PANEL_FEATURE_COLS,
                        min_train_dates=int(p_min_dates),
                        test_chunk_dates=int(p_chunk),
                        step_dates=int(p_step),
                        embargo_trading_days=int(p_embargo),
                    )
                except ValueError as e:
                    st.error(str(e))
                else:
                    pred_w, ret_w = pivot_predictions_and_returns(oos_p)
                    rank_bt = None
                    try:
                        if p_mode.startswith("Bucket"):
                            rank_bt = long_short_rank_portfolio_returns(
                                pred_w,
                                ret_w,
                                top_n=int(p_top),
                                bottom_n=int(p_bot),
                                demean_cs=bool(p_demean),
                            )
                        else:
                            if float(pq_lo) >= float(pq_hi):
                                st.error("Short quantile must be < long quantile.")
                            else:
                                mt = float(p_turn) if float(p_turn) > 0.0 else None
                                tcp = float(p_tc) if p_cost_prox and float(p_tc) > 0.0 else None
                                ic_daily = cross_sectional_ic_spearman_by_date(pred_w, ret_w)
                                ric_p = ic_daily.rolling(int(ic_window), min_periods=max(10, int(ic_window) // 3)).mean()
                                hl_p = ic_half_life_days(ric_p)
                                if p_auto_ic_rebal:
                                    k_use = rebalance_k_from_ic_half_life(hl_p)
                                    if hl_p == hl_p:
                                        st.caption(f"Auto rebalance: IC half-life ≈ **{hl_p:.1f}** bars → **k = {k_use}**")
                                    else:
                                        st.caption("Auto rebalance: IC half-life undefined → **k = 1**")
                                else:
                                    k_use = int(p_rebal_k)
                                nb = float(p_band) if float(p_band) > 0.0 else None
                                sq_out = score_quantile_ls_portfolio(
                                    pred_w,
                                    ret_w,
                                    long_q=float(pq_hi),
                                    short_q=float(pq_lo),
                                    demean_cs=bool(p_demean),
                                    max_turnover=mt,
                                    cost_per_trade=tcp,
                                    beta_neutral=bool(p_beta),
                                    beta_window=int(p_beta_w),
                                    rebalance_every_k=k_use,
                                    coverage_weight_returns=bool(p_cov_w),
                                    no_trade_band=nb,
                                    vol_target=bool(p_vol_tgt),
                                    target_vol_annual=0.12,
                                )
                                rank_bt = sq_out["result"]
                                if p_vol_tgt:
                                    st.caption(
                                        "Vol targeting **on**: 12% ann. target, 20-day realized vol, causal leverage (cap 2×). "
                                        "See `vol_target_leverage` and `strategy_return_pre_vol` in the result table export."
                                    )
                                if int(p_embargo) > 0:
                                    st.caption(f"Train **embargo**: last **{int(p_embargo)}** business days before each OOS chunk excluded from training.")
                                st.metric("Hit rate (stock-days, w·r>0)", f"{sq_out['hit_rate']:.3f}")
                                cov_s = sq_out["coverage"]
                                cov = pd.DataFrame({"trade_date": cov_s.index, "coverage": cov_s.to_numpy()})
                                fig_cov = px.line(cov, x="trade_date", y="coverage", title="Prediction coverage (fraction non-NaN)")
                                st.plotly_chart(fig_cov, use_container_width=True)
                                avg_ic_p = float(ic_daily.mean(skipna=True)) if len(ic_daily) else float("nan")
                                m1, m2, m3, m4, m5, m6 = st.columns(6)
                                m1.metric("Avg CS IC", f"{avg_ic_p:.4f}" if avg_ic_p == avg_ic_p else "—")
                                m2.metric("CS IC×√252", f"{avg_ic_p * math.sqrt(252):.2f}" if avg_ic_p == avg_ic_p else "—")
                                m3.metric("IC half-life (bars)", f"{hl_p:.1f}" if hl_p == hl_p else "—")
                                m4.metric("Avg coverage", f"{float(sq_out['avg_coverage']) * 100:.1f}%")
                                adt = sq_out.get("avg_dollar_turnover", float("nan"))
                                m5.metric("Avg dollar TO (½L1)", f"{adt:.4f}" if adt == adt else "—")
                                awt = sq_out.get("avg_weight_turnover", float("nan"))
                                m6.metric("Avg gross L1 ‖Δw‖₁", f"{awt:.4f}" if awt == awt else "—")
                                gexp = sq_out.get("avg_gross_exposure", float("nan"))
                                nexp = sq_out.get("avg_net_exposure", float("nan"))
                                if gexp == gexp and nexp == nexp:
                                    st.caption(f"Avg gross |w| (row L1): **{gexp:.4f}** · Avg net Σw: **{nexp:.6f}**")
                                icd = sq_out.get("implied_daily_cost_drag", float("nan"))
                                if tcp is not None and icd == icd:
                                    st.caption(f"Rough implied daily cost (½L1 TO × tc): **{icd:.5f}** (tc = {tcp:.4f})")
                                if "topk_attribution" in rank_bt.columns:
                                    st.metric("Cumulative top-20% |w| sleeve Σ", f"{rank_bt['topk_attribution'].sum():.4f}")
                                    st.metric("Cumulative rest sleeve Σ", f"{rank_bt['rest_attribution'].sum():.4f}")
                    except ValueError as e:
                        st.warning(str(e))
                    if rank_bt is not None and not rank_bt.empty:
                        pr = rank_bt["strategy_return"]
                        pm = portfolio_metrics_from_returns(pr)
                        st.metric("Panel strategy CAGR", f"{pm['cagr']:.2%}")
                        st.metric("Panel Sharpe (ann.)", f"{pm['sharpe']:.2f}")
                        st.metric("Panel max DD", f"{pm['max_dd']:.2%}")
                        if "long_attribution" in rank_bt.columns:
                            st.metric("Cumulative long sleeve Σ", f"{rank_bt['long_attribution'].sum():.4f}")
                            st.metric("Cumulative short sleeve Σ", f"{rank_bt['short_attribution'].sum():.4f}")
                        fig_pk = go.Figure()
                        fig_pk.add_trace(
                            go.Scatter(
                                x=rank_bt["trade_date"],
                                y=rank_bt["cum_strategy_return"] - 1.0,
                                name="Panel L/S",
                            )
                        )
                        fig_pk.update_layout(title="Panel ML — cumulative", yaxis_tickformat=".1%")
                        st.plotly_chart(fig_pk, use_container_width=True)

    with st.expander("Return correlation heatmap (market daily returns)", expanded=False):
        corr_syms = st.multiselect("Symbols for correlation", symbols, default=symbols[: min(6, len(symbols))])
        if len(corr_syms) >= 2:
            mrets: dict[str, pd.Series] = {}
            for cs in corr_syms:
                cdf = fetch_stock_data(cs)
                if cdf.empty:
                    continue
                cdf = cdf.copy()
                cdf["trade_date"] = pd.to_datetime(cdf["trade_date"], errors="coerce")
                cdf = cdf.dropna(subset=["trade_date"]).sort_values("trade_date")
                # One row per day before returns — duplicates break pct_change and can empty the inner join.
                cdf = cdf.drop_duplicates(subset=["trade_date"], keep="last")
                g = cdf[["trade_date", "close_price"]].copy()
                g["close_price"] = pd.to_numeric(g["close_price"], errors="coerce")
                if "daily_return" in cdf.columns:
                    g["daily_return"] = pd.to_numeric(cdf["daily_return"], errors="coerce")
                g["dr"] = _daily_return_series_prefer_varying_close(g)
                mrets[cs] = g.set_index(pd.to_datetime(g["trade_date"]))["dr"]
            try:
                corr = returns_correlation_matrix(mrets)
                fig_h = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="Pearson correlation — market daily returns (full overlapping history)",
                )
                fig_h.update_layout(height=max(420, 70 * len(corr.index)))
                st.plotly_chart(fig_h, use_container_width=True, key="ml_market_corr_heatmap")
            except ValueError as e:
                st.warning(str(e))
            except Exception as e:
                st.error(f"Correlation heatmap failed to render: {e}")

    with st.expander("Notes / limitations"):
        st.write(
            "- Use **Expanding quantile** for causal thresholds; full-window OOS quantile remains a research shortcut.\n"
            "- **Rolling max** confidence scale uses a lagged rolling max |pred| for more stable exposure.\n"
            "- **Rolling inverse-vol** uses **lagged** rolling σ per leg; global inv-vol still uses full OOS column std.\n"
            "- Panel **score-quantile** uses cross-sectional quantiles per day; **IC×√252** is a rough IR-style scale, not a realized IR.\n"
            "- **Turnover cap** scales the proposed Δw when ‖Δw‖₁ exceeds the cap (preserves direction, reduces step size).\n"
            "- Panel metrics report **dollar turnover** as ½·mean(‖Δw‖₁) (industry convention); gross L1 is shown for comparison.\n"
            "- Panel score PnL uses **w.shift(1)·r** (weights set before the return bar); optional **vol targeting** scales that return series.\n"
            "- **Train embargo** drops rows just before each OOS chunk from the training set to reduce overlapping-label leakage.\n"
            "- Sector-neutral weights are not implemented yet (only CS demean of weights)."
        )
