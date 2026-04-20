-- ============================================
-- Procedure: Load Bronze → Silver
-- Purpose: Canonical daily bars — **one row per (symbol, trade_date)**
-- Note: Production ETL may still run in Python; this file documents the **desired** SQL shape.
-- ============================================
--
-- Why dedupe: duplicate calendar days in bronze (re-loads, overlapping files) inflate row counts
-- and break pct_change / ML / portfolio math. Silver should collapse to a single EOD row per key.
-- Convention: keep the **latest** ingested row per (symbol, trade_date) — same as pandas
-- `drop_duplicates(subset=['trade_date'], keep='last')` when `id` increases with inserts.
--
-- Prerequisites:
--   - dbo.stock_prices_bronze(symbol, trade_date, ..., id IDENTITY)
--   - dbo.stock_prices_silver(same price columns; id may re-seed on truncate/insert)
--
-- Pick ONE strategy below (all illustrative; entire script stays commented out).

/*
-- ---------------------------------------------------------------------------
-- Option A — Full refresh: truncate silver, insert deduped bronze (simplest)
-- ---------------------------------------------------------------------------
TRUNCATE TABLE dbo.stock_prices_silver;

INSERT INTO dbo.stock_prices_silver (
    symbol,
    trade_date,
    open_price,
    high_price,
    low_price,
    close_price,
    volume
)
SELECT
    symbol,
    trade_date,
    open_price,
    high_price,
    low_price,
    close_price,
    volume
FROM (
    SELECT
        b.*,
        ROW_NUMBER() OVER (
            PARTITION BY b.symbol, b.trade_date
            ORDER BY b.id DESC          -- latest ingest wins (EOD refresh)
        ) AS rn
    FROM dbo.stock_prices_bronze AS b
) AS d
WHERE d.rn = 1;

-- ---------------------------------------------------------------------------
-- Option B — MERGE upsert (if you keep silver persistent and only patch changes)
-- ---------------------------------------------------------------------------
-- MERGE dbo.stock_prices_silver AS tgt
-- USING (
--     SELECT
--         symbol,
--         trade_date,
--         open_price,
--         high_price,
--         low_price,
--         close_price,
--         volume
--     FROM (
--         SELECT
--             b.*,
--             ROW_NUMBER() OVER (
--                 PARTITION BY b.symbol, b.trade_date
--                 ORDER BY b.id DESC
--             ) AS rn
--         FROM dbo.stock_prices_bronze AS b
--     ) x
--     WHERE x.rn = 1
-- ) AS src
--   ON tgt.symbol = src.symbol AND tgt.trade_date = src.trade_date
-- WHEN MATCHED THEN UPDATE SET
--     open_price = src.open_price,
--     high_price = src.high_price,
--     low_price = src.low_price,
--     close_price = src.close_price,
--     volume = src.volume
-- WHEN NOT MATCHED BY TARGET THEN INSERT (
--     symbol, trade_date, open_price, high_price, low_price, close_price, volume
-- ) VALUES (
--     src.symbol, src.trade_date, src.open_price, src.high_price, src.low_price, src.close_price, src.volume
-- );

-- ---------------------------------------------------------------------------
-- Option C — Validate: list symbols with duplicate (symbol, trade_date) in bronze
-- ---------------------------------------------------------------------------
-- SELECT symbol, trade_date, COUNT(*) AS cnt
-- FROM dbo.stock_prices_bronze
-- GROUP BY symbol, trade_date
-- HAVING COUNT(*) > 1
-- ORDER BY cnt DESC, symbol, trade_date;

-- ---------------------------------------------------------------------------
-- NOT recommended: SELECT DISTINCT on all columns — wrong if two rows differ slightly
-- on the same day, and does not define which duplicate to keep.
-- ---------------------------------------------------------------------------
*/
