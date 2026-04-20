-- Gold layer: curated, analytics-ready stock prices (SQL Server)
-- Run against your target database (e.g. StockPulse) in SSMS or sqlcmd.

IF OBJECT_ID(N'dbo.stock_prices_gold', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.stock_prices_gold
    (
        id             INT IDENTITY(1, 1) NOT NULL,
        symbol         NVARCHAR(32)       NOT NULL,
        trade_date     DATE               NOT NULL,
        open_price     DECIMAL(18, 6)     NOT NULL,
        high_price     DECIMAL(18, 6)     NOT NULL,
        low_price      DECIMAL(18, 6)     NOT NULL,
        close_price    DECIMAL(18, 6)     NOT NULL,
        volume         BIGINT             NOT NULL,
        sma_20         DECIMAL(18, 6)     NULL,
        sma_50         DECIMAL(18, 6)     NULL,
        daily_return   DECIMAL(18, 8)     NULL,
        trend          NVARCHAR(16)       NULL,
        signal         NVARCHAR(10)       NULL,
        loaded_at_utc  DATETIME2(7)       NOT NULL
            CONSTRAINT DF_stock_prices_gold_loaded_at_utc DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_stock_prices_gold PRIMARY KEY CLUSTERED (id),
        CONSTRAINT UQ_stock_prices_gold_symbol_trade_date UNIQUE (symbol, trade_date)
    );

    CREATE NONCLUSTERED INDEX IX_stock_prices_gold_symbol
        ON dbo.stock_prices_gold (symbol);

    CREATE NONCLUSTERED INDEX IX_stock_prices_gold_trade_date
        ON dbo.stock_prices_gold (trade_date);
END;
GO

-- If stock_prices_gold already existed without feature columns, add them:
IF OBJECT_ID(N'dbo.stock_prices_gold', N'U') IS NOT NULL
BEGIN
    IF COL_LENGTH('dbo.stock_prices_gold', 'sma_20') IS NULL
        ALTER TABLE dbo.stock_prices_gold ADD sma_20 DECIMAL(18, 6) NULL;
    IF COL_LENGTH('dbo.stock_prices_gold', 'daily_return') IS NULL
        ALTER TABLE dbo.stock_prices_gold ADD daily_return DECIMAL(18, 8) NULL;
    IF COL_LENGTH('dbo.stock_prices_gold', 'trend') IS NULL
        ALTER TABLE dbo.stock_prices_gold ADD trend NVARCHAR(16) NULL;
    IF COL_LENGTH('dbo.stock_prices_gold', 'sma_50') IS NULL
        ALTER TABLE dbo.stock_prices_gold ADD sma_50 DECIMAL(18, 6) NULL;
    IF COL_LENGTH('dbo.stock_prices_gold', 'signal') IS NULL
        ALTER TABLE dbo.stock_prices_gold ADD signal NVARCHAR(10) NULL;
END;
GO
