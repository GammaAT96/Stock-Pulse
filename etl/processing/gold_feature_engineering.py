import numpy as np
import pandas as pd

SMA_WINDOW = 20
SMA_50_WINDOW = 50


def create_gold_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build analytics columns from Silver OHLCV: SMAs, daily return, trend vs SMA-20,
    and buy/sell/hold signal from SMA-20 vs SMA-50.
    """
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.dropna(subset=["symbol", "trade_date", "close_price"])
    out = out.sort_values(["symbol", "trade_date"])

    chunks = []
    for _, g in out.groupby("symbol", sort=False):
        g = g.copy()
        g["daily_return"] = g["close_price"].pct_change()
        window = min(SMA_WINDOW, max(1, len(g)))
        g["sma_20"] = g["close_price"].rolling(window=window, min_periods=1).mean()
        g["sma_50"] = g["close_price"].rolling(
            window=SMA_50_WINDOW, min_periods=1
        ).mean()
        close = g["close_price"].to_numpy()
        sma = g["sma_20"].to_numpy()
        g["trend"] = np.where(close > sma, "up", np.where(close < sma, "down", "flat"))
        g["signal"] = "hold"
        g.loc[g["sma_20"] > g["sma_50"], "signal"] = "buy"
        g.loc[g["sma_20"] < g["sma_50"], "signal"] = "sell"
        chunks.append(g)

    return pd.concat(chunks, ignore_index=True)
