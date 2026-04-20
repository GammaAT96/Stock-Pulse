import logging

import pandas as pd

logger = logging.getLogger(__name__)


def standardize_bronze_data(df: pd.DataFrame):
    df = df.copy()

    logger.info("[Silver] Initial rows: %s", len(df))

    # Fix date
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date

    # Convert numeric columns
    numeric_cols = ["open_price", "high_price", "low_price", "close_price", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove bad rows
    df = df.dropna(subset=["symbol", "trade_date", "close_price"])

    # Deduplicate (keep latest); Bronze may not have an identity id column
    if "id" in df.columns:
        df = df.sort_values("id")
    df = df.drop_duplicates(
        subset=["symbol", "trade_date"],
        keep="last",
    )

    # Final sort
    df = df.sort_values(["symbol", "trade_date"])

    logger.info("[Silver] Final rows: %s", len(df))

    return df