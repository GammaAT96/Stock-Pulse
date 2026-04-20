-- ============================================
-- Sector Mapping (Reference Data)
-- Maps stock symbol → sector
-- ============================================

IF OBJECT_ID(N'dbo.sector_mapping', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.sector_mapping (
        symbol VARCHAR(50) PRIMARY KEY,
        sector VARCHAR(50)
    );
END;

-- Insert only if not already present
IF NOT EXISTS (SELECT 1 FROM sector_mapping)
BEGIN
    INSERT INTO sector_mapping (symbol, sector) VALUES
    ('AAPL', 'Technology'),
    ('MSFT', 'Technology'),
    ('GOOGL', 'Technology'),
    ('AMZN', 'E-Commerce'),
    ('TSLA', 'Automobile'),
    ('JPM', 'Finance'),
    ('BAC', 'Finance'),
    ('WMT', 'Retail'),
    ('NFLX', 'Entertainment'),
    ('NVDA', 'Technology'),
    ('META', 'Technology');
END;