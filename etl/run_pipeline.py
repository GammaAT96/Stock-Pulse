import logging
import sys
from pathlib import Path

# Allow `python etl/run_pipeline.py`: project root must be on sys.path for `import etl.*`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from etl.logging_config import configure_logging

configure_logging()

import pandas as pd

from etl.ingestion.csv_ingestion import ingest_csv_files
from etl.processing.gold_feature_engineering import create_gold_features
from etl.processing.silver_standardization import standardize_bronze_data
from etl.storage.database_writer import (
    engine,
    insert_to_bronze,
    insert_to_gold,
    insert_to_silver,
)

logger = logging.getLogger(__name__)


def run_bronze_pipeline():
    dfs = ingest_csv_files()

    for df in dfs:
        insert_to_bronze(df)


def run_silver_layer(engine):
    logger.info("===== SILVER START =====")

    df = pd.read_sql("SELECT * FROM stock_prices_bronze", engine)
    logger.info("[Silver] Rows from Bronze: %s", len(df))

    df_clean = standardize_bronze_data(df)

    insert_to_silver(df_clean)

    logger.info("===== SILVER DONE =====")


def run_gold_layer(engine):
    logger.info("===== GOLD START =====")

    df = pd.read_sql("SELECT * FROM stock_prices_silver", engine)
    logger.info("[Gold] Rows from Silver: %s", len(df))

    df_gold = create_gold_features(df)

    insert_to_gold(df_gold)

    logger.info("===== GOLD DONE =====")


def run_pipeline():
    """Bronze → Silver → Gold (single entry for CLI and schedulers)."""
    run_bronze_pipeline()
    run_silver_layer(engine)
    run_gold_layer(engine)


if __name__ == "__main__":
    logger.info("Starting ETL pipeline...")
    run_pipeline()
    logger.info("Pipeline completed")
