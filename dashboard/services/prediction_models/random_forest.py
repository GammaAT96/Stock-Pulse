from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_rf(df: pd.DataFrame, feature_cols: list[str] | tuple[str, ...]) -> RandomForestRegressor:
    X = df.loc[:, list(feature_cols)]
    y = df["target"]
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict_rf(model: RandomForestRegressor, df: pd.DataFrame, feature_cols: list[str] | tuple[str, ...]) -> pd.Series:
    X = df.loc[:, list(feature_cols)]
    preds = model.predict(X)
    return pd.Series(preds, index=df.index, name="prediction")
