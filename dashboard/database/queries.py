import pandas as pd
from sqlalchemy import text

from .connection import get_engine

engine = get_engine()


def get_all_symbols():
    query = """
    SELECT DISTINCT symbol
    FROM stock_prices_gold
    ORDER BY symbol
    """
    return pd.read_sql(query, engine)


def get_stock_data(symbol):
    query = text(
        """
    SELECT *
    FROM stock_prices_gold
    WHERE symbol = :symbol
    ORDER BY trade_date
    """
    )
    return pd.read_sql(query, engine, params={"symbol": symbol})