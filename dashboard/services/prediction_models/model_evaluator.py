from __future__ import annotations

import math

import numpy as np
import pandas as pd


def evaluate_model(df: pd.DataFrame) -> dict[str, float]:
    """
    Out-of-sample diagnostics for a return forecast.

    Note: these are *not* trading PnL metrics; always also backtest signals in the strategy engine.
    """
    if df.empty:
        return {
            "n": 0.0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "directional_accuracy": float("nan"),
            "ic_pearson": float("nan"),
            "ic_spearman": float("nan"),
            "mean_pred": float("nan"),
            "mean_target": float("nan"),
        }

    y = pd.to_numeric(df["target"], errors="coerce")
    p = pd.to_numeric(df["prediction"], errors="coerce")
    m = y.notna() & p.notna()
    y = y.loc[m]
    p = p.loc[m]

    err = p - y
    rmse = float(math.sqrt(float(np.mean(np.square(err))))) if len(err) else float("nan")
    mae = float(np.mean(np.abs(err))) if len(err) else float("nan")

    directional_accuracy = float(np.mean(np.sign(p) == np.sign(y))) if len(y) else float("nan")

    ic_pearson = float("nan")
    ic_spearman = float("nan")
    if len(y) > 2:
        ic_pearson = float(y.corr(p, method="pearson"))
        ic_spearman = float(y.corr(p, method="spearman"))

    return {
        "n": float(len(y)),
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": directional_accuracy,
        "ic_pearson": ic_pearson,
        "ic_spearman": ic_spearman,
        "mean_pred": float(p.mean()) if len(p) else float("nan"),
        "mean_target": float(y.mean()) if len(y) else float("nan"),
    }
