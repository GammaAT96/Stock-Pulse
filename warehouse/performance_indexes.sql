-- ============================================
-- Performance Indexes for StockPulse
-- Improves query speed on Gold layer
-- ============================================

-- Index on symbol (for filtering by stock)
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes 
    WHERE name = 'idx_gold_symbol'
)
BEGIN
    CREATE NONCLUSTERED INDEX idx_gold_symbol
    ON dbo.stock_prices_gold(symbol);
END;

-- Index on trade_date (for time-based queries)
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes 
    WHERE name = 'idx_gold_trade_date'
)
BEGIN
    CREATE NONCLUSTERED INDEX idx_gold_trade_date
    ON dbo.stock_prices_gold(trade_date);
END;

-- Composite index (best for dashboard queries)
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes 
    WHERE name = 'idx_gold_symbol_date'
)
BEGIN
    CREATE NONCLUSTERED INDEX idx_gold_symbol_date
    ON dbo.stock_prices_gold(symbol, trade_date);
END;