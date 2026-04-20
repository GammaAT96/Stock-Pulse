import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / "configuration" / ".env"
    load_dotenv(env_path, override=False)


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required env var: {name}")
    return value


def _build_sqlalchemy_url() -> str:
    server = _required("DB_SERVER")
    db = _required("DB_NAME")
    driver = _required("DB_DRIVER")

    trusted = (os.getenv("DB_TRUSTED_CONNECTION") or "").strip().lower()
    if trusted in {"yes", "true", "1"}:
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={db};"
            "Trusted_Connection=yes;"
        )
        return f"mssql+pyodbc:///?odbc_connect={quote_plus(conn_str)}"

    username = _required("DB_USERNAME")
    password = _required("DB_PASSWORD")
    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={db};"
        f"UID={username};"
        f"PWD={password};"
    )
    return f"mssql+pyodbc:///?odbc_connect={quote_plus(conn_str)}"


def test_connection() -> None:
    _load_env()
    url = _build_sqlalchemy_url()

    print("Attempting DB connection...")

    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Database connected successfully")
    except Exception as e:
        print("Connection failed:", e)


if __name__ == "__main__":
    test_connection()