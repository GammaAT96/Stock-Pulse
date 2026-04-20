import logging
import os
import shutil

import pandas as pd

logger = logging.getLogger(__name__)

# ==============================
# PATH CONFIGURATION
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

INCOMING_DIR = os.path.join(BASE_DIR, "data_sources", "incoming_csv")
ARCHIVE_DIR = os.path.join(BASE_DIR, "data_sources", "processed_archive")
REJECTED_DIR = os.path.join(BASE_DIR, "data_sources", "rejected")

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


# ==============================
# CHECK IF ALREADY PROCESSED
# ==============================

def already_processed(file_name):
    return os.path.exists(os.path.join(ARCHIVE_DIR, file_name))


# ==============================
# VALIDATION FUNCTION
# ==============================

def validate_dataframe(df):
    if df.empty:
        raise ValueError("empty_file")

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError("missing_columns")

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{col}_not_numeric")

    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except:
        raise ValueError("invalid_date")

    return df


# ==============================
# COPY FILE FUNCTION
# ==============================

def copy_file(file_path, destination, reason=None):
    filename = os.path.basename(file_path)

    if reason:
        name, ext = os.path.splitext(filename)
        filename = f"{name}_ERROR_{reason}{ext}"

    dest_path = os.path.join(destination, filename)

    shutil.copy(file_path, dest_path)


# ==============================
# MAIN INGESTION FUNCTION
# ==============================

def ingest_csv_files():
    dataframes = []

    files = os.listdir(INCOMING_DIR)

    if not files:
        logger.info("No files found in incoming_csv/")
        return dataframes

    for file in files:
        file_path = os.path.join(INCOMING_DIR, file)

        # Skip non-CSV
        if not file.endswith(".csv"):
            logger.info("Skipping non-CSV: %s", file)
            continue

        # 🔥 SKIP LOGIC
        if already_processed(file):
            logger.info("Skipping already processed file: %s", file)
            continue

        try:
            logger.info("Processing: %s", file)

            df = pd.read_csv(file_path)
            df = validate_dataframe(df)

            df.columns = [col.strip().capitalize() for col in df.columns]

            file_stem = os.path.splitext(file)[0]

            # Prefer Ticker column (e.g. HCLTECH.NS); else Symbol in CSV; else first segment of filename stem.
            if "Ticker" in df.columns:
                t = df["Ticker"]
                if t.notna().any():
                    df["Symbol"] = t.astype(str).str.strip()
                else:
                    df["Symbol"] = file_stem.split("_")[0]
                df = df.drop(columns=["Ticker"])
            elif "Symbol" in df.columns:
                pass
            else:
                df["Symbol"] = file_stem.split("_")[0]

            dataframes.append(df)

            copy_file(file_path, ARCHIVE_DIR)

            logger.info("SUCCESS: %s", file)

        except Exception as e:
            reason = str(e)
            logger.error("FAILED %s: %s", file, reason)

            copy_file(file_path, REJECTED_DIR, reason=reason)

    return dataframes


# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from etl.logging_config import configure_logging

    configure_logging()
    dfs = ingest_csv_files()
    logger.info("Total valid files: %s", len(dfs))