from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def _split_index(n: int, train_size: float) -> int:
    if not 0.0 < train_size < 1.0:
        raise ValueError("train_size must be between 0 and 1.")
    if n < 3:
        raise ValueError("Not enough rows for walk-forward split.")
    split = int(n * train_size)
    return max(1, min(split, n - 1))


def walk_forward_validation(
    df: pd.DataFrame,
    model_func: Callable[[pd.DataFrame, list[str] | tuple[str, ...]], object],
    feature_cols: list[str] | tuple[str, ...],
    train_size: float = 0.7,
) -> pd.DataFrame:
    """
    Chronological single split walk-forward: train on an early window, predict on the remainder.

    This is the minimal baseline the user requested; upgrade path is rolling/expanding windows.
    """
    n = len(df)
    split = _split_index(n, float(train_size))

    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    model = model_func(train, feature_cols)
    preds = model.predict(test.loc[:, list(feature_cols)])
    test["prediction"] = preds
    return test


def walk_forward_predict(
    df: pd.DataFrame,
    model_func: Callable[[pd.DataFrame, list[str] | tuple[str, ...]], object],
    predict_func: Callable[[object, pd.DataFrame, list[str] | tuple[str, ...]], pd.Series],
    feature_cols: list[str] | tuple[str, ...],
    train_size: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame, object, list[object]]:
    """
    Train once on the early window, return (train_df, test_df_with_prediction, model).

    `predict_func` must return a Series aligned to the provided dataframe rows.

    Returns ``(train, test, model, [model])`` so callers can treat importances like rolling (len=1).
    """
    n = len(df)
    split = _split_index(n, float(train_size))

    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    model = model_func(train, feature_cols)
    test = test.copy()
    test["prediction"] = predict_func(model, test, feature_cols)
    return train, test, model, [model]


def rolling_walk_forward_predict(
    df: pd.DataFrame,
    model_func: Callable[[pd.DataFrame, list[str] | tuple[str, ...]], object],
    predict_func: Callable[[object, pd.DataFrame, list[str] | tuple[str, ...]], pd.Series],
    feature_cols: list[str] | tuple[str, ...],
    *,
    min_train_rows: int,
    test_chunk: int,
    step: int,
) -> tuple[pd.DataFrame, pd.DataFrame, object | None, list[object]]:
    """
    Expanding-window walk-forward: repeatedly train on df[:i], predict the next chunk, advance i.

    Returns ``(last_train, oos_concat, last_model, all_models)``.
    Includes a **final partial chunk** when ``i < n`` after the main loop.
    """
    n = len(df)
    if min_train_rows < 10:
        raise ValueError("min_train_rows too small.")
    if test_chunk < 1 or step < 1:
        raise ValueError("test_chunk and step must be positive.")
    if min_train_rows >= n:
        raise ValueError("min_train_rows must be smaller than dataframe length.")

    chunks: list[pd.DataFrame] = []
    i = int(min_train_rows)
    last_model: object | None = None
    last_train: pd.DataFrame | None = None
    models: list[object] = []

    while i < n:
        end = min(i + int(test_chunk), n)
        if end <= i:
            break
        train = df.iloc[:i].copy()
        test = df.iloc[i:end].copy()
        model = model_func(train, feature_cols)
        test = test.copy()
        test["prediction"] = predict_func(model, test, feature_cols)
        chunks.append(test)
        models.append(model)
        last_model = model
        last_train = train
        i += int(step)

    if not chunks:
        raise ValueError("No walk-forward folds produced; relax min_train_rows or test_chunk.")

    oos = pd.concat(chunks, axis=0, ignore_index=True)
    if "trade_date" in oos.columns:
        oos = oos.sort_values("trade_date").reset_index(drop=True)
    return last_train if last_train is not None else df.iloc[:0].copy(), oos, last_model, models
