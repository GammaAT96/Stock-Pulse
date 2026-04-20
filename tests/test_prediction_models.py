import logging
import math
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.services.prediction_models.feature_prep import DEFAULT_FEATURE_COLS, create_features
from dashboard.services.prediction_models.model_evaluator import evaluate_model
from dashboard.services.duplicate_pivot_warnings import (
    clear_duplicate_pivot_warnings_cache,
    warn_once_duplicate_trade_symbol_rows,
)
from dashboard.services.portfolio_service import build_symbol_returns_panel, compute_portfolio
from dashboard.services.prediction_models.cross_sectional_ml import (
    cross_sectional_ic_spearman_by_date,
    long_short_rank_portfolio_returns,
    panel_expanding_date_walk_forward,
    pivot_predictions_and_returns,
    score_quantile_ls_portfolio,
)
from dashboard.services.prediction_models.ml_backtest import (
    attach_ml_strategy_returns,
    ic_half_life_days,
    rebalance_k_from_ic_half_life,
    realized_turnover,
    underwater_equity_curve,
)
from dashboard.services.prediction_models.ml_portfolio import (
    aggregate_feature_importances,
    cross_sectional_equal_weight,
    cross_sectional_inverse_vol_weights,
    cross_sectional_inverse_vol_weights_rolling,
    returns_correlation_matrix,
)
from dashboard.services.prediction_models.random_forest import predict_rf, train_rf
from dashboard.services.prediction_models.walk_forward import rolling_walk_forward_predict, walk_forward_predict


def setUpModule():
    clear_duplicate_pivot_warnings_cache()


class TestPredictionModels(unittest.TestCase):
    def test_create_features_dedupes_trade_date_before_returns(self):
        """Duplicate gold rows per date make pct_change≈0; features must collapse to one row per day."""
        n = 120
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        rows: list[dict] = []
        for i in range(n):
            rows.append({"trade_date": dates[i], "close_price": prices[i]})
            if i % 3 == 0:
                rows.append({"trade_date": dates[i], "close_price": prices[i]})
        df = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
        out = create_features(df)
        self.assertGreater(len(out), 20)
        self.assertGreater(float(out["daily_return"].std(ddof=1)), 1e-8)

    def test_create_features_drops_last_row_no_future_target(self):
        n = 200
        rng = np.random.default_rng(0)
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))
        df = pd.DataFrame(
            {
                "trade_date": pd.date_range("2020-01-01", periods=n, freq="B"),
                "close_price": prices,
                "sma_20": pd.Series(prices).rolling(20).mean(),
                "sma_50": pd.Series(prices).rolling(50).mean(),
            }
        )
        out = create_features(df)
        self.assertTrue(out["target"].notna().all())
        # last calendar row cannot have a realized next-day return in-sample
        last_date = pd.to_datetime(df["trade_date"].iloc[-1])
        self.assertFalse((pd.to_datetime(out["trade_date"]) == last_date).any())

    def test_walk_forward_predict_trains_only_on_past(self):
        n = 300
        rng = np.random.default_rng(1)
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.005, size=n)))
        df = pd.DataFrame(
            {
                "trade_date": pd.date_range("2018-01-01", periods=n, freq="B"),
                "close_price": prices,
                "sma_20": pd.Series(prices).rolling(20).mean(),
                "sma_50": pd.Series(prices).rolling(50).mean(),
            }
        )
        feats = create_features(df)
        split = int(len(feats) * 0.7)

        train_df, test_df, _model, models = walk_forward_predict(
            feats,
            train_rf,
            predict_rf,
            DEFAULT_FEATURE_COLS,
            train_size=0.7,
        )
        self.assertEqual(len(models), 1)

        self.assertEqual(len(train_df), split)
        self.assertEqual(len(test_df), len(feats) - split)
        self.assertIn("prediction", test_df.columns)

        m = evaluate_model(test_df)
        self.assertGreater(m["n"], 10.0)
        self.assertFalse(np.isnan(m["rmse"]))

    def test_rolling_walk_forward_concatenates_oos(self):
        n = 400
        rng = np.random.default_rng(2)
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.004, size=n)))
        df = pd.DataFrame(
            {
                "trade_date": pd.date_range("2017-01-01", periods=n, freq="B"),
                "close_price": prices,
                "sma_20": pd.Series(prices).rolling(20).mean(),
                "sma_50": pd.Series(prices).rolling(50).mean(),
            }
        )
        feats = create_features(df)
        _, oos, _, models = rolling_walk_forward_predict(
            feats,
            train_rf,
            predict_rf,
            DEFAULT_FEATURE_COLS,
            min_train_rows=150,
            test_chunk=20,
            step=20,
        )
        self.assertIn("prediction", oos.columns)
        self.assertGreater(len(oos), 50)
        self.assertGreater(len(models), 1)

    def test_rolling_includes_partial_tail(self):
        n = 175
        rng = np.random.default_rng(3)
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.004, size=n)))
        df = pd.DataFrame(
            {
                "trade_date": pd.date_range("2019-01-01", periods=n, freq="B"),
                "close_price": prices,
                "sma_20": pd.Series(prices).rolling(20).mean(),
                "sma_50": pd.Series(prices).rolling(50).mean(),
            }
        )
        feats = create_features(df)
        _, oos, _, _ = rolling_walk_forward_predict(
            feats,
            train_rf,
            predict_rf,
            DEFAULT_FEATURE_COLS,
            min_train_rows=100,
            test_chunk=40,
            step=40,
        )
        self.assertEqual(len(oos), len(feats) - 100)

    def test_quantile_oos_thresholding(self):
        oos = pd.DataFrame(
            {
                "daily_return": np.zeros(100),
                "prediction": np.linspace(-0.01, 0.01, 100),
            }
        )
        out = attach_ml_strategy_returns(
            oos,
            threshold_mode="quantile_oos",
            quantile_hi=0.9,
            quantile_lo=0.1,
            signal_mode="long_only",
            position_style="discrete",
        )
        self.assertGreater(int(out["signal"].sum()), 0)

    def test_realized_turnover_and_importance_agg(self):
        bt = attach_ml_strategy_returns(
            pd.DataFrame({"daily_return": [0.01, 0.0, -0.01], "prediction": [0.05, -0.02, 0.03]}),
            signal_mode="long_short_flat",
            position_style="discrete",
            prediction_threshold=0.0,
        )
        t = realized_turnover(bt)
        self.assertGreaterEqual(t, 0.0)
        rng = np.random.default_rng(4)
        n = 40
        tdf = pd.DataFrame({"prediction": rng.normal(size=n), "target": rng.normal(size=n)})
        m1 = train_rf(tdf, ["prediction"])
        m2 = train_rf(tdf, ["prediction"])
        mean_imp, std_imp = aggregate_feature_importances([m1, m2], ["prediction"])
        self.assertIsNotNone(mean_imp)
        self.assertIsNotNone(std_imp)

    def test_quantile_expanding_is_causal_length(self):
        oos = pd.DataFrame(
            {
                "daily_return": np.zeros(80),
                "prediction": np.linspace(-0.02, 0.02, 80),
            }
        )
        out = attach_ml_strategy_returns(
            oos,
            threshold_mode="quantile_expanding",
            quantile_hi=0.8,
            quantile_lo=0.2,
            quantile_expanding_min_periods=20,
            signal_mode="long_only",
            position_style="discrete",
        )
        self.assertEqual(len(out), 80)

    def test_equal_weight_concat_with_duplicate_index(self):
        d0 = pd.Timestamp("2020-01-01")
        d1 = pd.Timestamp("2020-01-02")
        a = pd.Series([0.01, 0.02, 0.03], index=pd.to_datetime([d0, d0, d1]))
        b = pd.Series([0.0, 0.04], index=pd.to_datetime([d0, d1]))
        port, panel = cross_sectional_equal_weight({"X": a, "Y": b})
        self.assertFalse(port.index.has_duplicates)
        self.assertEqual(len(port), 2)
        self.assertAlmostEqual(float(port.loc[d0]), (0.02 + 0.0) / 2.0)

    def test_correlation_matrix_duplicate_index(self):
        rng = pd.date_range("2020-01-01", periods=8, freq="B")
        a = pd.Series(np.linspace(0.01, 0.02, len(rng)), index=rng)
        a2 = pd.concat([a.iloc[:-1], pd.Series([0.09], index=[rng[-1]]), pd.Series([0.08], index=[rng[-1]])])
        b = pd.Series(np.linspace(-0.01, 0.015, len(rng)), index=rng)
        c = returns_correlation_matrix({"X": a2, "Y": b})
        self.assertEqual(c.shape, (2, 2))

    def test_underwater_and_inverse_vol(self):
        dr = pd.Series([0.01, -0.02, 0.015, -0.005])
        uw = underwater_equity_curve(dr)
        self.assertLessEqual(float(uw.max()), 0.0)
        panel = pd.DataFrame({"A": [0.01, -0.01, 0.02], "B": [-0.005, 0.01, -0.008]})
        port, w = cross_sectional_inverse_vol_weights(panel)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
        self.assertEqual(len(port), 3)

    def test_score_quantile_ls_portfolio(self):
        rng = np.random.default_rng(7)
        ix = pd.date_range("2020-01-01", periods=25, freq="B")
        pred = pd.DataFrame(rng.normal(size=(25, 5)), index=ix, columns=list("ABCDE"))
        ret = pd.DataFrame(rng.normal(0, 0.01, size=(25, 5)), index=ix, columns=list("ABCDE"))
        out = score_quantile_ls_portfolio(
            pred,
            ret,
            long_q=0.75,
            short_q=0.25,
            demean_cs=True,
            max_turnover=0.4,
        )
        self.assertGreater(len(out["result"]), 3)
        self.assertIn("coverage", out)
        self.assertIn("avg_weight_turnover", out)
        self.assertIn("avg_dollar_turnover", out)
        awt = out["avg_weight_turnover"]
        adt = out["avg_dollar_turnover"]
        if awt == awt and adt == adt:
            self.assertAlmostEqual(adt, 0.5 * awt, places=8)

    def test_rebalance_k_from_ic_half_life(self):
        self.assertEqual(rebalance_k_from_ic_half_life(float("nan")), 1)
        self.assertEqual(rebalance_k_from_ic_half_life(6.0), 3)
        self.assertEqual(rebalance_k_from_ic_half_life(2.0), 1)
        self.assertEqual(rebalance_k_from_ic_half_life(24.0), 10)

    def test_score_quantile_vol_targeting(self):
        rng = np.random.default_rng(13)
        ix = pd.date_range("2020-01-01", periods=60, freq="B")
        pred = pd.DataFrame(rng.normal(size=(60, 4)), index=ix, columns=list("ABCD"))
        ret = pd.DataFrame(rng.normal(0, 0.01, size=(60, 4)), index=ix, columns=list("ABCD"))
        out = score_quantile_ls_portfolio(pred, ret, vol_target=True, target_vol_annual=0.12)
        self.assertTrue(out["vol_target"])
        self.assertIn("vol_target_leverage", out["result"].columns)
        self.assertIn("strategy_return_pre_vol", out["result"].columns)
        self.assertFalse(np.allclose(out["result"]["strategy_return"], out["result"]["strategy_return_pre_vol"]))

    def test_panel_walk_forward_embargo_requires_history(self):
        """Embargo must leave enough training dates or panel_expanding_date_walk_forward raises."""
        rng = np.random.default_rng(14)
        dates = pd.date_range("2020-01-01", periods=120, freq="B")
        rows = []
        for d in dates:
            for sym in ("X", "Y"):
                rows.append(
                    {
                        "trade_date": d,
                        "symbol": sym,
                        "close_price": float(rng.uniform(50, 150)),
                        "sma_20": float(rng.uniform(48, 152)),
                        "sma_50": float(rng.uniform(48, 152)),
                    }
                )
        raw = pd.DataFrame(rows)
        parts = []
        for sym in ("X", "Y"):
            sub = raw[raw["symbol"] == sym].drop(columns=["symbol"])
            feats = create_features(sub)
            if feats.empty:
                continue
            g = feats.copy()
            g["symbol"] = sym
            parts.append(g)
        panel = pd.concat(parts, ignore_index=True)
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        panel = panel.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

        def _train(td, fc):
            return train_rf(td, tuple(fc))

        def _pred(mo, td, fc):
            return predict_rf(mo, td, tuple(fc))

        panel_expanding_date_walk_forward(
            panel,
            _train,
            _pred,
            DEFAULT_FEATURE_COLS,
            min_train_dates=40,
            test_chunk_dates=10,
            step_dates=10,
            embargo_trading_days=0,
        )
        with self.assertRaises(ValueError):
            panel_expanding_date_walk_forward(
                panel,
                _train,
                _pred,
                DEFAULT_FEATURE_COLS,
                min_train_dates=100,
                test_chunk_dates=10,
                step_dates=10,
                embargo_trading_days=80,
            )

    def test_score_quantile_no_trade_band_runs(self):
        rng = np.random.default_rng(12)
        ix = pd.date_range("2020-01-01", periods=35, freq="B")
        pred = pd.DataFrame(rng.normal(size=(35, 5)), index=ix, columns=list("ABCDE"))
        ret = pd.DataFrame(rng.normal(0, 0.01, size=(35, 5)), index=ix, columns=list("ABCDE"))
        out_band = score_quantile_ls_portfolio(
            pred,
            ret,
            no_trade_band=0.01,
            cost_per_trade=0.0005,
        )
        self.assertFalse(out_band["result"].empty)
        icd = out_band["implied_daily_cost_drag"]
        self.assertTrue(icd == icd)
        self.assertAlmostEqual(icd, 0.5 * out_band["avg_weight_turnover"] * 0.0005, places=10)

    def test_score_quantile_cost_proximal_smoke(self):
        rng = np.random.default_rng(11)
        ix = pd.date_range("2020-01-01", periods=30, freq="B")
        pred = pd.DataFrame(rng.normal(size=(30, 4)), index=ix, columns=list("WXYZ"))
        ret = pd.DataFrame(rng.normal(0, 0.01, size=(30, 4)), index=ix, columns=list("WXYZ"))
        out = score_quantile_ls_portfolio(
            pred,
            ret,
            cost_per_trade=0.002,
            max_turnover=0.5,
            rebalance_every_k=2,
            beta_neutral=True,
            beta_window=25,
        )
        self.assertFalse(out["result"].empty)

    def test_ic_half_life_days(self):
        self.assertTrue(math.isnan(ic_half_life_days(pd.Series([0.1, 0.2]))))
        rng = np.random.default_rng(0)
        eps = rng.normal(0, 0.05, size=200)
        y = np.zeros_like(eps)
        phi = 0.85
        for i in range(1, len(y)):
            y[i] = phi * y[i - 1] + eps[i]
        s = pd.Series(y)
        hl = ic_half_life_days(s)
        self.assertFalse(math.isnan(hl))
        self.assertGreater(hl, 1.0)

    def test_warn_once_duplicate_pivot_single_log_per_key(self):
        clear_duplicate_pivot_warnings_cache()
        d0 = pd.Timestamp("2020-01-01")
        df = pd.DataFrame(
            {
                "trade_date": [d0, d0],
                "symbol": ["X", "X"],
                "daily_return": [0.01, 0.02],
            }
        )
        log = logging.getLogger("dashboard.services.duplicate_pivot_warnings")
        with self.assertLogs("dashboard.services.duplicate_pivot_warnings", level="WARNING") as cm:
            warn_once_duplicate_trade_symbol_rows(log, "unit_test_dup", df)
            warn_once_duplicate_trade_symbol_rows(log, "unit_test_dup", df)
        self.assertEqual(len(cm.output), 1)

    def test_pivot_predictions_dedupes_duplicate_trade_symbol(self):
        d0 = pd.Timestamp("2020-01-01")
        d1 = pd.Timestamp("2020-01-02")
        oos = pd.DataFrame(
            {
                "trade_date": [d0, d0, d1, d1],
                "symbol": ["A", "A", "A", "B"],
                "prediction": [0.1, 0.9, 0.2, 0.3],
                "daily_return": [0.01, 0.02, 0.03, 0.04],
            }
        )
        pred_w, ret_w = pivot_predictions_and_returns(oos)
        self.assertAlmostEqual(float(pred_w.loc[d0, "A"]), 0.9)
        self.assertAlmostEqual(float(ret_w.loc[d0, "A"]), 0.02)

    def test_build_symbol_returns_panel_dedupes_trade_date_before_returns(self):
        """Duplicate rows per trade_date make pct_change 0 between duplicates; must collapse first."""
        d0 = pd.Timestamp("2020-01-01")
        d1 = pd.Timestamp("2020-01-02")
        raw = pd.DataFrame(
            {
                "trade_date": [d0, d0, d0, d1],
                "close_price": [100.0, 100.0, 100.0, 110.0],
            }
        )

        def fetch_fn(_sym):
            return raw

        df = build_symbol_returns_panel(["Z"], d0.date(), d1.date(), fetch_fn)
        self.assertEqual(len(df), 2)
        self.assertGreater(float(df["daily_return"].abs().max()), 1e-6)

    def test_build_symbol_returns_panel_falls_back_when_close_flat(self):
        dates = pd.date_range("2020-01-01", periods=6, freq="B")

        def fetch_flat_close(_sym):
            return pd.DataFrame(
                {
                    "trade_date": dates,
                    "close_price": [100.0] * 6,
                    "daily_return": [0.0, 0.01, -0.005, 0.002, -0.001, 0.0],
                }
            )

        df = build_symbol_returns_panel(
            ["X"],
            dates[0].date(),
            dates[-1].date(),
            fetch_flat_close,
        )
        self.assertFalse(df.empty)
        self.assertGreater(float(df["daily_return"].abs().sum()), 0.01)

    def test_compute_portfolio_with_duplicate_rows(self):
        df = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2020-01-01", "2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]
                ),
                "symbol": ["A", "A", "B", "A", "B"],
                "daily_return": [0.01, 0.02, 0.05, 0.03, 0.04],
            }
        )
        pivot, _m = compute_portfolio(df, ["A", "B"])
        self.assertEqual(len(pivot), 2)
        self.assertAlmostEqual(float(pivot["portfolio_return"].iloc[0]), (0.02 + 0.05) / 2.0)
        self.assertAlmostEqual(float(pivot["portfolio_return"].iloc[1]), (0.03 + 0.04) / 2.0)

    def test_cross_sectional_ic_spearman_by_date(self):
        ix = pd.date_range("2020-01-01", periods=12, freq="B")
        rng = np.random.default_rng(42)
        pred = pd.DataFrame(rng.normal(size=(12, 5)), index=ix, columns=list("VWXYZ"))
        ret = pd.DataFrame(rng.normal(0, 0.02, size=(12, 5)), index=ix, columns=list("VWXYZ"))
        ic = cross_sectional_ic_spearman_by_date(pred, ret)
        self.assertEqual(len(ic), 12)
        self.assertTrue(ic.iloc[1:].notna().any())

    def test_rolling_inverse_vol_weights(self):
        rng = np.random.default_rng(8)
        ix = pd.date_range("2019-01-01", periods=80, freq="B")
        panel = pd.DataFrame(rng.normal(0, 0.01, size=(80, 3)), index=ix, columns=list("XYZ"))
        port, w = cross_sectional_inverse_vol_weights_rolling(panel, vol_window=20, min_periods=10)
        self.assertEqual(len(port), 80)
        self.assertEqual(w.shape, panel.shape)

    def test_rolling_inverse_vol_ewma(self):
        rng = np.random.default_rng(9)
        ix = pd.date_range("2019-01-01", periods=80, freq="B")
        panel = pd.DataFrame(rng.normal(0, 0.01, size=(80, 3)), index=ix, columns=list("XYZ"))
        port, w = cross_sectional_inverse_vol_weights_rolling(
            panel, vol_window=20, min_periods=10, vol_method="ewma", ewma_lambda=0.94
        )
        self.assertEqual(len(port), 80)
        self.assertEqual(w.shape, panel.shape)

    def test_panel_rank_portfolio_shape(self):
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        syms = ["X", "Y", "Z"]
        rows = []
        rng = np.random.default_rng(5)
        for d in dates:
            for s in syms:
                rows.append(
                    {
                        "trade_date": d,
                        "symbol": s,
                        "prediction": float(rng.normal()),
                        "daily_return": float(rng.normal(0, 0.01)),
                    }
                )
        oos = pd.DataFrame(rows)
        pred_w, ret_w = pivot_predictions_and_returns(oos)
        out = long_short_rank_portfolio_returns(pred_w, ret_w, top_n=1, bottom_n=1)
        self.assertGreater(len(out), 5)

    def test_ml_signal_threshold_reduces_longs(self):
        oos = pd.DataFrame(
            {
                "daily_return": [0.01, -0.01, 0.02, 0.0],
                "prediction": [0.001, 0.003, -0.003, 0.004],
            }
        )
        flat = attach_ml_strategy_returns(oos.copy(), prediction_threshold=0.002, signal_mode="long_only")
        self.assertListEqual(flat["signal"].tolist(), [0, 1, 0, 1])

        ls = attach_ml_strategy_returns(oos.copy(), prediction_threshold=0.002, signal_mode="long_short_flat")
        self.assertListEqual(ls["signal"].tolist(), [0, 1, -1, 1])


if __name__ == "__main__":
    unittest.main()
