import pandas as pd
import yfinance as yf


def download_stock_data(
    tickers: list[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    if not tickers:
        raise ValueError("The ticker list cannot be empty.")
    if not start_date or not end_date:
        raise ValueError("Both start_date and end_date are required.")
    cleaned_tickers = [ticker.upper().strip() for ticker in tickers]
    data = yf.download(
        tickers=cleaned_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )
    if data.empty:
        raise ValueError(
            "No stock data was downloaded. "
            "Check the ticker symbols and date range."
        )
    return data

def get_closing_prices(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise ValueError("The downloaded data is empty.")
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            closing_prices = data["Adj Close"].copy()
        elif "Close" in data.columns.get_level_values(0):
            closing_prices = data["Close"].copy()
        else:
            raise ValueError("No 'Adj Close' or 'Close' data was found.")
    else:
        if "Adj Close" in data.columns:
            closing_prices = data[["Adj Close"]].copy()
        elif "Close" in data.columns:
            closing_prices = data[["Close"]].copy()
        else:
            raise ValueError("No 'Adj Close' or 'Close' column was found.")
    closing_prices = closing_prices.sort_index()
    closing_prices = closing_prices.dropna(how="any")
    if closing_prices.empty:
        raise ValueError("No valid closing-price data remains.")

    return closing_prices

if __name__ == "__main__":
    stock_data = download_stock_data(
        tickers=["AAPL", "MSFT"],
        start_date="2023-01-01",
        end_date="2024-01-01"
    )

    print(stock_data.head())
    print(stock_data.columns)