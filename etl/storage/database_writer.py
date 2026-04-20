import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# ==============================
# LOAD ENV VARIABLES (project root, not CWD)
# ==============================

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / "configuration" / ".env"
load_dotenv(_ENV_PATH)

logger = logging.getLogger(__name__)

DB_SERVER = os.getenv("DB_SERVER")
DB_NAME = os.getenv("DB_NAME")
DB_DRIVER = os.getenv("DB_DRIVER")
DB_TRUSTED = os.getenv("DB_TRUSTED_CONNECTION")

DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ==============================
# CREATE CONNECTION
# ==============================

if DB_TRUSTED == "yes":
    connection_string = (
        f"mssql+pyodbc://@{DB_SERVER}/{DB_NAME}"
        f"?driver={DB_DRIVER}&trusted_connection=yes"
    )
else:
    connection_string = (
        f"mssql+pyodbc://{DB_USERNAME}:{DB_PASSWORD}@{DB_SERVER}/{DB_NAME}"
        f"?driver={DB_DRIVER}"
    )

engine = create_engine(connection_string)


# ==============================
# TEST CONNECTION
# ==============================

def test_connection():
    logger.info("Attempting DB connection...")

    try:
        with engine.connect() as conn:
            logger.info("Database connected successfully")
    except Exception as e:
        logger.error("Connection failed: %s", e)


# ==============================
# INSERT INTO BRONZE
# ==============================

def insert_to_bronze(df: pd.DataFrame):
    df = df.copy()

    # ==============================
    # COLUMN MAPPING (ROBUST)
    # ==============================

    # Standard mappings
    df.rename(columns={
        "Date": "trade_date",
        "Open": "open_price",
        "High": "high_price",
        "Low": "low_price",
        "Close": "close_price",
        "Volume": "volume",
    }, inplace=True)

    # Handle symbol source dynamically (match ingest: Symbol, Ticker, ticker)
    if "Symbol" in df.columns:
        df.rename(columns={"Symbol": "symbol"}, inplace=True)
    elif "Ticker" in df.columns:
        df.rename(columns={"Ticker": "symbol"}, inplace=True)
    elif "ticker" in df.columns:
        df.rename(columns={"ticker": "symbol"}, inplace=True)
    else:
        raise ValueError("No symbol column found (Symbol/Ticker/ticker missing)")

    # ==============================
    # HANDLE DATE
    # ==============================

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

    # ==============================
    # DROP INVALID ROWS (OPTIONAL BUT SAFE)
    # ==============================

    df = df.dropna(subset=["trade_date"])

    # ==============================
    # KEEP ONLY REQUIRED COLUMNS
    # ==============================

    df = df[
        [
            "symbol",
            "trade_date",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume"
        ]
    ]

    # ==============================
    # INSERT INTO DATABASE
    # ==============================

    try:
        df.to_sql(
            "stock_prices_bronze",
            engine,
            if_exists="append",
            index=False
        )

        logger.info("Inserted into Bronze table")

    except Exception as e:
        logger.exception("Bronze insert failed: %s", e)


# ==============================
# INSERT INTO SILVER
# ==============================

def insert_to_silver(df: pd.DataFrame):
    try:
        df = df.copy()

        # Remove Bronze id so Silver IDENTITY column is not fed explicit values
        if "id" in df.columns:
            df = df.drop(columns=["id"])

        df.to_sql(
            "stock_prices_silver",
            engine,
            if_exists="append",
            index=False,
        )
        logger.info("Inserted into Silver")
    except Exception as e:
        logger.exception("Silver insert failed: %s", e)


# ==============================
# INSERT INTO GOLD
# ==============================

GOLD_COLUMNS = [
    "symbol",
    "trade_date",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "sma_20",
    "sma_50",
    "daily_return",
    "trend",
    "signal",
]


def insert_to_gold(df: pd.DataFrame):
    try:
        df = df.copy()

        if "id" in df.columns:
            df = df.drop(columns=["id"])

        missing = [c for c in GOLD_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Gold insert missing columns: {missing}")

        df = df[GOLD_COLUMNS]

        df.to_sql(
            "stock_prices_gold",
            engine,
            if_exists="append",
            index=False,
        )
        logger.info("Inserted into Gold")
    except Exception as e:
        logger.exception("Gold insert failed: %s", e)