from __future__ import annotations

import pandas as pd

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency in some environments
    XGBRegressor = None  # type: ignore[misc, assignment]


def xgboost_available() -> bool:
    return XGBRegressor is not None


def train_xgb(df: pd.DataFrame, feature_cols: list[str] | tuple[str, ...]):
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed. Install `xgboost` to use this model.")
    X = df.loc[:, list(feature_cols)]
    y = df["target"]
    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict_xgb(model, df: pd.DataFrame, feature_cols: list[str] | tuple[str, ...]) -> pd.Series:
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed. Install `xgboost` to use this model.")
    X = df.loc[:, list(feature_cols)]
    preds = model.predict(X)
    return pd.Series(preds, index=df.index, name="prediction")
