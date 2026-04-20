from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env (same logic as ETL)
env_path = Path(__file__).resolve().parent.parent.parent / "configuration" / ".env"
load_dotenv(env_path)

def get_engine():
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    driver = os.getenv("DB_DRIVER")

    connection_string = (
        f"mssql+pyodbc://@{server}/{database}"
        f"?driver={driver}&trusted_connection=yes"
    )

    return create_engine(connection_string)