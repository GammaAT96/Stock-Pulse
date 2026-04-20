-- ============================================
-- Procedure: Load Silver → Gold
-- Purpose: Feature engineering (SMA, signals)
-- Note: Currently implemented in Python
-- ============================================

/*
-- Example SQL-based feature logic (illustrative; production uses pandas rolling)
-- Mirrors: etl/processing/gold_feature_engineering.py

WITH base AS (
    SELECT
        symbol,
        trade_date,
        open_price,
        high_price,
        low_price,
        close_price,
        volume,
        AVG(close_price) OVER (
            PARTITION BY symbol
            ORDER BY trade_date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS sma_20,
        AVG(close_price) OVER (
            PARTITION BY symbol
            ORDER BY trade_date
            ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
        ) AS sma_50
    FROM dbo.stock_prices_silver
)
SELECT
    symbol,
    trade_date,
    open_price,
    high_price,
    low_price,
    close_price,
    volume,
    sma_20,
    sma_50,
    (close_price / NULLIF(LAG(close_price) OVER (PARTITION BY symbol ORDER BY trade_date), 0)) - 1.0
        AS daily_return,
    CASE
        WHEN close_price > sma_20 THEN N'up'
        WHEN close_price < sma_20 THEN N'down'
        ELSE N'flat'
    END AS trend,
    CASE
        WHEN sma_20 > sma_50 THEN N'buy'
        WHEN sma_20 < sma_50 THEN N'sell'
        ELSE N'hold'
    END AS signal
FROM base;
*/
