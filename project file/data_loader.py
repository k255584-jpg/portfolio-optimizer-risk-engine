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
if __name__ == "__main__":
    stock_data = download_stock_data(
        tickers=["AAPL", "MSFT"],
        start_date="2023-01-01",
        end_date="2024-01-01"
    )

    print(stock_data.head())
    print(stock_data.columns)