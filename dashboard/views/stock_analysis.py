import math
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly import io as pio

from dashboard.services.data_service import fetch_symbols, fetch_stock_data
from dashboard.services.portfolio_service import build_symbol_returns_panel, compute_portfolio

TRADING_DAYS_PER_YEAR = 252


def _pipeline_df_view(
    df: pd.DataFrame,
    start_date,
    end_date,
    sma_short_n: int,
    sma_long_n: int,
    transaction_cost_pct: float = 0.0,
    rf_annual: float = 0.0,
    include_rf: bool = False,
) -> pd.DataFrame:
    """Sort, compute SMAs on full history, slice to date range, add returns and trade markers."""
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.dropna(subset=["trade_date"]).sort_values("trade_date")
    close = pd.to_numeric(out["close_price"], errors="coerce")
    # Always compute returns from prices to avoid unit/scaling issues from upstream storage.
    out["daily_return"] = close.pct_change().fillna(0.0)
    out["sma_short"] = close.rolling(sma_short_n, min_periods=1).mean()
    out["sma_long"] = close.rolling(sma_long_n, min_periods=1).mean()

    td = out["trade_date"].dt.date
    out = out[(td >= start_date) & (td <= end_date)].copy()
    if out.empty:
        return out

    dr = pd.to_numeric(out["daily_return"], errors="coerce").fillna(0.0)
    # Signal is computed on today's close, but we can only trade starting next bar.
    signal_long = (out["sma_short"] > out["sma_long"]).fillna(False).astype(int)
    out["position"] = signal_long.shift(1).fillna(0).astype(int)
    out["trade_change"] = out["position"].diff().fillna(0).astype(int)
    out["trade"] = out["trade_change"].abs().fillna(0)
    cost_drag = out["trade"] * float(transaction_cost_pct)
    out["cost_drag"] = cost_drag
    rf_daily = float(rf_annual) / TRADING_DAYS_PER_YEAR if include_rf else 0.0
    out["rf_daily"] = rf_daily
    out["strategy_return"] = (
        (out["position"].astype(float) * dr)
        + ((1.0 - out["position"].astype(float)) * rf_daily)
        - cost_drag
    )
    out["cum_market_return"] = (1 + dr).cumprod()
    out["cum_strategy_return"] = (1 + out["strategy_return"]).cumprod()
    return out


def _benchmark_buy_hold(
    df: pd.DataFrame,
    start_date,
    end_date,
) -> Optional[dict]:
    """Buy-and-hold cumulative return and CAGR on the same calendar window (no strategy logic)."""
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.dropna(subset=["trade_date"]).sort_values("trade_date")
    close = pd.to_numeric(out["close_price"], errors="coerce")
    dr_all = close.pct_change().fillna(0.0)
    td = out["trade_date"].dt.date
    out = out[(td >= start_date) & (td <= end_date)].copy()
    if out.empty:
        return None
    dr = pd.to_numeric(dr_all.loc[out.index], errors="coerce").fillna(0.0)
    cum = (1 + dr).cumprod()
    n = len(out)
    y = n / TRADING_DAYS_PER_YEAR if n else 0.0
    cm_end = float(cum.iloc[-1])
    total = cm_end - 1.0
    cagr = cm_end ** (1.0 / y) - 1.0 if y > 0 and cm_end > 0 else float("nan")
    return {"total": total, "cagr": cagr, "n_days": n}


def _round_trip_returns(df_view: pd.DataFrame) -> list[float]:
    """Return % PnL for each completed long (buy → next sell) using close prices."""
    dv = df_view.reset_index(drop=True)
    if dv.empty or "trade_change" not in dv.columns:
        return []
    buys = dv.index[dv["trade_change"] == 1].tolist()
    sells = dv.index[dv["trade_change"] == -1].tolist()
    close = dv["close_price"].astype(float)
    rets: list[float] = []
    si = 0
    for b in buys:
        while si < len(sells) and sells[si] <= b:
            si += 1
        if si >= len(sells):
            break
        s = sells[si]
        cb = float(close.iloc[b])
        cs = float(close.iloc[s])
        if pd.notna(cb) and pd.notna(cs) and cb != 0:
            rets.append(cs / cb - 1.0)
        si += 1
    return rets


def _sharpe_daily(sr: pd.Series) -> float:
    if len(sr) <= 1:
        return float("nan")
    std = float(sr.std(ddof=1))
    if std <= 0 or math.isnan(std):
        return float("nan")
    return float(sr.mean() / std)


def show():
    st.title("📈 Stock Analysis")

    theme = str(st.session_state.get("theme", "dark")).lower()
    is_dark = theme == "dark"
    pio.templates.default = "plotly_dark" if is_dark else "plotly_white"

    symbols = fetch_symbols()
    if not symbols:
        st.warning("No symbols in gold layer. Run the ETL pipeline first.")
        return

    selected_symbol = st.selectbox("Select Stock", symbols)

    df = fetch_stock_data(selected_symbol)
    if df.empty:
        st.info("No rows for this symbol.")
        return

    df = df.copy()
    df["signal"] = df["signal"].apply(
        lambda x: "hold" if pd.isna(x) else str(x).strip().lower()
    )

    st.subheader("Strategy parameters")
    pc1, pc2 = st.columns(2)
    with pc1:
        sma_short_n = st.slider("Short SMA period", 5, 50, 20, help="Faster moving average window")
    with pc2:
        long_lo = max(20, sma_short_n + 1)
        long_default = max(50, long_lo)
        sma_long_n = st.slider(
            "Long SMA period",
            min_value=long_lo,
            max_value=200,
            value=min(long_default, 200),
            help="Slower moving average (must be greater than short SMA)",
        )

    pc3, pc4 = st.columns(2)
    with pc3:
        cost_pct = st.number_input(
            "Transaction cost per side (fraction of NAV)",
            min_value=0.0,
            max_value=0.05,
            value=0.001,
            step=0.0005,
            format="%.4f",
            help="Applied on each position change (buy or sell). Subtracted from that day’s strategy return. "
            "0.001 = 0.1% per side.",
        )
    with pc4:
        bench_options = [s for s in symbols if s != selected_symbol]
        bench_label = st.selectbox(
            "Benchmark — buy & hold (optional)",
            ["— None —"] + bench_options,
            help="Compare strategy CAGR to another symbol’s buy-and-hold on the same dates (e.g. ETF/index proxy if loaded in gold).",
        )
        benchmark_sym = None if bench_label == "— None —" else bench_label

    pc5, pc6 = st.columns(2)
    with pc5:
        include_rf = st.checkbox(
            "Include risk-free return (cash carry)",
            value=False,
            help="When flat, apply a constant daily risk-free return to capture opportunity cost.",
        )
    with pc6:
        rf_annual = st.number_input(
            "Risk-free rate (annual, fraction)",
            min_value=0.0,
            max_value=0.25,
            value=0.05,
            step=0.005,
            format="%.3f",
            disabled=not include_rf,
            help="Example: 0.05 = 5% annualized. Applied as rf_annual/252 on cash days.",
        )

    dmin = pd.to_datetime(df["trade_date"], errors="coerce").min().date()
    dmax = pd.to_datetime(df["trade_date"], errors="coerce").max().date()

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start date", value=dmin, min_value=dmin, max_value=dmax)
    with c2:
        end_date = st.date_input("End date", value=dmax, min_value=dmin, max_value=dmax)

    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return

    df_view = _pipeline_df_view(
        df,
        start_date,
        end_date,
        sma_short_n,
        sma_long_n,
        transaction_cost_pct=cost_pct,
        rf_annual=rf_annual,
        include_rf=include_rf,
    )

    if df_view.empty:
        st.info("No rows in the selected date range.")
        return

    st.subheader("📊 Price data")
    _hide_table = {"cum_market_return", "cum_strategy_return", "cost_drag", "rf_daily"}
    _cols = [c for c in df_view.columns if c not in _hide_table]
    st.dataframe(df_view[_cols], width="stretch")

    for col in ("close_price", "sma_short", "sma_long"):
        df_view[col] = pd.to_numeric(df_view[col], errors="coerce")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_view["trade_date"],
            y=df_view["close_price"],
            mode="lines",
            name="Close price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_view["trade_date"],
            y=df_view["sma_short"],
            mode="lines",
            name=f"SMA {sma_short_n}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_view["trade_date"],
            y=df_view["sma_long"],
            mode="lines",
            name=f"SMA {sma_long_n}",
        )
    )

    # Mark trade events on the day we actually change position (after lag).
    buy_df = df_view[df_view["trade_change"] == 1]
    sell_df = df_view[df_view["trade_change"] == -1]

    fig.add_trace(
        go.Scatter(
            x=buy_df["trade_date"],
            y=buy_df["close_price"],
            mode="markers",
            name="BUY (cross up)",
            marker=dict(symbol="triangle-up", size=11, color="#2ecc71"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sell_df["trade_date"],
            y=sell_df["close_price"],
            mode="markers",
            name="SELL (cross down)",
            marker=dict(symbol="triangle-down", size=11, color="#e74c3c"),
        )
    )

    fig.update_layout(
        title=f"{selected_symbol} — price, SMA({sma_short_n}/{sma_long_n}), crosses",
        hovermode="x unified",
        legend_title_text="",
        height=520,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("📊 Strategy vs market (cumulative)")
    st.caption(
        f"Market = buy-and-hold. Strategy = long only when SMA{sma_short_n} > SMA{sma_long_n} "
        f"(otherwise cash), net of {cost_pct:.2%} per-side costs on turnover. "
        "Past performance does not guarantee future results."
    )
    st.caption(
        (
            "We assume zero return on cash for baseline simplicity; a production extension would apply a daily "
            "risk-free rate to uninvested periods to capture opportunity cost."
        )
    )
    perf = df_view[["trade_date", "cum_market_return", "cum_strategy_return"]].copy()
    perf["market_return"] = perf["cum_market_return"] - 1.0
    perf["strategy_return"] = perf["cum_strategy_return"] - 1.0
    fig_perf = px.line(
        perf,
        x="trade_date",
        y=["market_return", "strategy_return"],
        title="Cumulative return",
        labels={"value": "Return", "variable": "Series"},
    )
    fig_perf.update_layout(hovermode="x unified", legend_title_text="")
    fig_perf.update_layout(yaxis_tickformat=".0%")
    _trace_names = {
        "market_return": "Market (buy & hold)",
        "strategy_return": f"Strategy (SMA {sma_short_n}/{sma_long_n})",
    }
    for tr in fig_perf.data:
        if tr.name in _trace_names:
            tr.name = _trace_names[tr.name]
    st.plotly_chart(fig_perf, width="stretch")

    # --- Metrics: returns, annualized, risk, trades ---
    n_days = len(df_view)
    years = n_days / TRADING_DAYS_PER_YEAR if n_days else 0.0

    total_return = float(df_view["cum_strategy_return"].iloc[-1] - 1.0)
    market_return = float(df_view["cum_market_return"].iloc[-1] - 1.0)

    cs_end = float(df_view["cum_strategy_return"].iloc[-1])
    cm_end = float(df_view["cum_market_return"].iloc[-1])

    if years > 0 and cs_end > 0:
        cagr_strategy = cs_end ** (1.0 / years) - 1.0
    else:
        cagr_strategy = float("nan")

    if years > 0 and cm_end > 0:
        cagr_market = cm_end ** (1.0 / years) - 1.0
    else:
        cagr_market = float("nan")

    alpha_cagr = (
        float(cagr_strategy - cagr_market)
        if cagr_strategy == cagr_strategy and cagr_market == cagr_market
        else float("nan")
    )

    exposure_pct = float(df_view["position"].mean()) if n_days else float("nan")

    sharpe_daily = _sharpe_daily(df_view["strategy_return"])
    if sharpe_daily == sharpe_daily:
        sharpe_annual = sharpe_daily * (TRADING_DAYS_PER_YEAR ** 0.5)
    else:
        sharpe_annual = float("nan")

    cum = df_view["cum_strategy_return"]
    drawdown = (cum / cum.cummax()) - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else float("nan")

    trip_rets = _round_trip_returns(df_view)
    n_trips = len(trip_rets)
    win_rate = float(sum(1 for r in trip_rets if r > 0) / n_trips) if n_trips else float("nan")
    avg_trade = float(sum(trip_rets) / n_trips) if n_trips else float("nan")

    n_buys = int((df_view["trade_change"] == 1).sum())
    n_sells = int((df_view["trade_change"] == -1).sum())

    st.subheader("📈 Performance & risk (in range)")
    r1 = st.columns(3)
    r1[0].metric("Strategy return", f"{total_return:.2%}")
    r1[1].metric("Market return", f"{market_return:.2%}")
    r1[2].metric("Excess vs market", f"{(total_return - market_return):.2%}")

    r2 = st.columns(3)
    r2[0].metric(
        "CAGR (strategy)",
        f"{cagr_strategy:.2%}" if cagr_strategy == cagr_strategy else "—",
    )
    r2[1].metric(
        "CAGR (market)",
        f"{cagr_market:.2%}" if cagr_market == cagr_market else "—",
    )
    r2[2].metric(
        "Sharpe (ann.)",
        f"{sharpe_annual:.2f}" if sharpe_annual == sharpe_annual else "—",
    )

    a1, a2 = st.columns(2)
    a1.metric(
        "Alpha vs market (ann.)",
        f"{alpha_cagr:.2%}" if alpha_cagr == alpha_cagr else "—",
    )
    a2.metric("Time in cash", f"{(1.0 - exposure_pct):.1%}" if exposure_pct == exposure_pct else "—")

    if benchmark_sym:
        braw = fetch_stock_data(benchmark_sym)
        bm = _benchmark_buy_hold(braw, start_date, end_date) if braw is not None and not braw.empty else None
        st.subheader("📎 Benchmark vs strategy")
        if bm:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric(f"Benchmark BH total ({benchmark_sym})", f"{bm['total']:.2%}")
            b2.metric(
                f"Benchmark CAGR ({benchmark_sym})",
                f"{bm['cagr']:.2%}" if bm["cagr"] == bm["cagr"] else "—",
            )
            if cagr_strategy == cagr_strategy and bm["cagr"] == bm["cagr"]:
                b3.metric(
                    "Strategy CAGR − benchmark CAGR",
                    f"{(cagr_strategy - bm['cagr']):.2%}",
                )
            else:
                b3.metric("Strategy CAGR − benchmark CAGR", "—")
            b4.metric("Benchmark days", f"{bm['n_days']}")
        else:
            st.warning("No benchmark data in the selected date range.")

    r3 = st.columns(3)
    r3[0].metric(
        "Max drawdown (strategy)",
        f"{max_dd:.2%}" if max_dd == max_dd else "—",
    )
    r3[1].metric(
        "Time in market",
        f"{exposure_pct:.1%}" if exposure_pct == exposure_pct else "—",
    )
    r3[2].metric("Trading days (range)", f"{n_days}")

    st.subheader("🔁 Trade statistics (completed round trips)")
    st.caption(
        "Round trips pair each BUY (short SMA crosses above long) with the next SELL. "
        "Win rate = share of trips with positive close-to-close return."
    )
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Round trips", str(n_trips))
    t2.metric("Win rate", f"{win_rate:.1%}" if win_rate == win_rate else "—")
    t3.metric("Avg return / trip", f"{avg_trade:.2%}" if avg_trade == avg_trade else "—")
    t4.metric("Buy / sell events", f"{n_buys} / {n_sells}")

    st.subheader("📌 Regime (in range, last bar)")
    latest = df_view.iloc[-1]
    price = latest.get("close_price", "")
    ss = latest["sma_short"]
    sl = latest["sma_long"]
    if pd.isna(ss) or pd.isna(sl):
        st.warning("Not enough bars in range to compute both SMAs.")
    elif ss > sl:
        st.success(f"Long — SMA{sma_short_n} above SMA{sma_long_n} — close {price}")
    elif ss < sl:
        st.error(f"Flat — SMA{sma_short_n} below SMA{sma_long_n} — close {price}")
    else:
        st.info(f"Neutral — SMAs equal — close {price}")

    # --- Multi-stock comparison (same window & SMA parameters) ---
    st.subheader("🧮 Cross-sectional comparison")
    st.caption("Uses the same date range and SMA settings as above.")
    compare_syms = st.multiselect(
        "Compare stocks",
        symbols,
        default=[],
        help="CAGR and cumulative strategy vs market end-values on the selected window.",
    )

    if compare_syms:
        rows = []
        for sym in compare_syms:
            raw = fetch_stock_data(sym)
            if raw.empty:
                continue
            dv = _pipeline_df_view(
                raw,
                start_date,
                end_date,
                sma_short_n,
                sma_long_n,
                transaction_cost_pct=cost_pct,
                rf_annual=rf_annual,
                include_rf=include_rf,
            )
            if dv.empty:
                rows.append(
                    {
                        "symbol": sym,
                        "strategy_total": None,
                        "market_total": None,
                        "cagr_strategy": None,
                        "cagr_market": None,
                        "n_days": 0,
                    }
                )
                continue
            n_d = len(dv)
            y = n_d / TRADING_DAYS_PER_YEAR if n_d else 0.0
            cs = float(dv["cum_strategy_return"].iloc[-1])
            cm = float(dv["cum_market_return"].iloc[-1])
            st_tot = cs - 1.0
            mk_tot = cm - 1.0
            cg_s = cs ** (1.0 / y) - 1.0 if y > 0 and cs > 0 else float("nan")
            cg_m = cm ** (1.0 / y) - 1.0 if y > 0 and cm > 0 else float("nan")
            sh_d = _sharpe_daily(dv["strategy_return"])
            sh_a = sh_d * (TRADING_DAYS_PER_YEAR ** 0.5) if sh_d == sh_d else float("nan")
            dd = (dv["cum_strategy_return"] / dv["cum_strategy_return"].cummax()) - 1.0
            mdd = float(dd.min()) if not dd.empty else float("nan")
            exp = float(dv["position"].mean()) if n_d else float("nan")
            rows.append(
                {
                    "symbol": sym,
                    "strategy_total": st_tot,
                    "market_total": mk_tot,
                    "cagr_strategy": cg_s,
                    "cagr_market": cg_m,
                    "sharpe_ann": sh_a,
                    "max_dd": mdd,
                    "exposure": exp,
                    "n_days": n_d,
                }
            )

        comp = pd.DataFrame(rows)
        if not comp.empty:
            # Ranking table (best → worst) for quick insight.
            best = comp.sort_values(["cagr_strategy", "sharpe_ann"], ascending=[False, False]).iloc[0]
            if pd.notna(best.get("symbol")):
                st.success(
                    f"🏆 Best (by strategy CAGR): {best['symbol']} "
                    f"(CAGR {best['cagr_strategy']:.2%} | Sharpe {best['sharpe_ann']:.2f} | Max DD {best['max_dd']:.2%})"
                )

            rank = comp.sort_values(["cagr_strategy", "sharpe_ann"], ascending=[False, False]).copy()
            disp = comp.copy()
            for c in (
                "strategy_total",
                "market_total",
                "cagr_strategy",
                "cagr_market",
                "sharpe_ann",
                "max_dd",
                "exposure",
            ):
                if c in disp.columns:
                    if c in {"sharpe_ann"}:
                        disp[c] = disp[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) and x == x else "—")
                    else:
                        disp[c] = disp[c].apply(lambda x: f"{x:.2%}" if pd.notna(x) and x == x else "—")
            st.dataframe(disp, width="stretch")

            cagr_df = comp.dropna(subset=["cagr_strategy"])
            if not cagr_df.empty:
                fig_c = px.bar(
                    cagr_df,
                    x="symbol",
                    y="cagr_strategy",
                    title=f"CAGR — strategy (SMA {sma_short_n}/{sma_long_n}, same window)",
                    labels={"cagr_strategy": "CAGR (strategy)", "symbol": "Symbol"},
                )
                fig_c.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig_c, width="stretch")

            tot_df = comp.dropna(subset=["strategy_total"])
            if not tot_df.empty:
                fig_e = px.bar(
                    tot_df,
                    x="symbol",
                    y="strategy_total",
                    title="Total return — strategy (not annualized)",
                    labels={"strategy_total": "Strategy total return", "symbol": "Symbol"},
                )
                fig_e.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig_e, width="stretch")

    st.subheader("📦 Portfolio simulator (equal-weight)")
    st.caption(
        "Uses the same date window as above. Returns use **`close_price` pct_change**; if that is flat in the DB, "
        "the pipeline falls back to **`daily_return`** from gold when present."
    )
    default_port = symbols[: min(3, len(symbols))]
    port_syms = st.multiselect(
        "Select stocks for portfolio",
        options=symbols,
        default=default_port,
        key="portfolio_select",
    )

    if len(port_syms) < 2:
        st.info("Select at least 2 stocks to build a portfolio.")
    else:
        df_panel = build_symbol_returns_panel(port_syms, start_date, end_date, fetch_stock_data)
        if df_panel.empty:
            st.warning("No overlapping price history for the selected portfolio symbols in this date range.")
        else:
            present = set(df_panel["symbol"].unique().tolist())
            missing_syms = [s for s in port_syms if s not in present]
            if missing_syms:
                st.warning(f"Missing data for symbols (skipped from portfolio math): {', '.join(missing_syms)}")

            usable_syms = [s for s in port_syms if s in present]
            if len(usable_syms) < 2:
                st.info("Need at least 2 symbols with data in-range to build a portfolio.")
            else:
                df_panel = build_symbol_returns_panel(usable_syms, start_date, end_date, fetch_stock_data)

                pivot = None
                pm = None
                try:
                    pivot, pm = compute_portfolio(df_panel, usable_syms)
                except ValueError as e:
                    st.warning(str(e))

                if pivot is not None and pm is not None and not pivot.empty:
                    fig_p = go.Figure()
                    fig_p.add_trace(
                        go.Scatter(
                            x=pivot.index,
                            y=pivot["total_return"] * 100.0,
                            mode="lines",
                            name="Portfolio (equal-weight)",
                        )
                    )
                    fig_p.update_layout(
                        title="Portfolio cumulative return (equal-weight, daily implicit rebalance)",
                        xaxis_title="Date",
                        yaxis_title="Return (%)",
                        hovermode="x unified",
                    )
                    pr_max = float(pivot["portfolio_return"].abs().max())
                    if pr_max < 1e-8:
                        st.warning(
                            "Portfolio daily returns are effectively **zero** (flat line). "
                            "Common causes: duplicate **`(symbol, trade_date)`** rows in gold (check "
                            "`COUNT(*) GROUP BY symbol, trade_date`), constant `close_price`, or bad ETL. "
                            "The simulator now collapses duplicate dates per symbol before returns; if this "
                            "persists, verify prices or re-run ETL."
                        )
                    _pk = f"pf_{start_date}_{end_date}_{'_'.join(sorted(usable_syms))}"
                    st.plotly_chart(fig_p, width="stretch", key=_pk)

                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Portfolio CAGR", f"{pm['cagr']:.2%}" if pm["cagr"] == pm["cagr"] else "—")
                    p2.metric("Max drawdown", f"{pm['max_dd']:.2%}" if pm["max_dd"] == pm["max_dd"] else "—")
                    p3.metric("Sharpe (ann.)", f"{pm['sharpe']:.2f}" if pm["sharpe"] == pm["sharpe"] else "—")
                    p4.metric("Trading days", f"{pm['n_days']}")

                    st.caption(
                        "Portfolio days use an **inner join**: any date missing any selected symbol is dropped "
                        "(no synthetic 0% returns). Next upgrade: forward-fill prices then recompute returns."
                    )
                    st.caption(
                        "Equal-weight here means **daily rebalanced equal-weight**: weights reset to 1/N each day via the cross-sectional mean of daily returns."
                    )

                    best_sym = (
                        df_panel.groupby("symbol", sort=False)["daily_return"]
                        .mean()
                        .sort_values(ascending=False)
                        .index[0]
                    )
                    st.caption(f"Highest average daily return in selection: **{best_sym}**")
