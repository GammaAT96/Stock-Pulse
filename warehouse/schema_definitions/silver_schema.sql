IF OBJECT_ID(N'dbo.stock_prices_silver', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.stock_prices_silver (
        id INT IDENTITY(1,1) PRIMARY KEY,
        symbol VARCHAR(50),
        trade_date DATE,
        open_price FLOAT,
        high_price FLOAT,
        low_price FLOAT,
        close_price FLOAT,
        volume BIGINT
    );
END;
GO