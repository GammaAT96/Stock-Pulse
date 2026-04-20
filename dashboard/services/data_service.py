from dashboard.database.queries import get_all_symbols, get_stock_data


def fetch_symbols():
    df = get_all_symbols()
    return df["symbol"].tolist()


def fetch_stock_data(symbol):
    return get_stock_data(symbol)
