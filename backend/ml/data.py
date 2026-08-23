import yfinance as yf
import pandas as pd


def get_historical_data(symbol: str, period: str = "10y"):
    symbol = symbol.upper()

    data = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(f"No historical data found for {symbol}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[
        ["Open", "High", "Low", "Close", "Volume"]
    ].copy()

    data.dropna(inplace=True)

    return data


if __name__ == "__main__":
    symbol = "RELIANCE.NS"

    df = get_historical_data(symbol)

    print(f"\nStock: {symbol}")
    print(f"Rows: {len(df)}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())