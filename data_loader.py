import numpy as np
import pandas as pd
import yfinance as yf


def download_stock_data(
    tickers: list[str],
    start_date: str,
    end_date: str
):
    if not tickers:
        raise ValueError("The ticker list cannot be empty.")
    if not start_date or not end_date:
        raise ValueError("Both start_date and end_date are required.")
    if not all(isinstance(ticker, str) for ticker in tickers):
        raise TypeError("Every ticker must be a string.")

    cleaned_tickers = []
    for ticker in tickers:
        cleaned_ticker = ticker.strip().upper()
        if not cleaned_ticker:
            raise ValueError("Ticker symbols cannot be empty.")
        if cleaned_ticker not in cleaned_tickers:
            cleaned_tickers.append(cleaned_ticker)

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
    data.attrs["requested_tickers"] = cleaned_tickers
    return data

def get_closing_prices(data: pd.DataFrame):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a Pandas DataFrame.")

    if data.empty:
        raise ValueError("The downloaded data is empty.")

    if isinstance(data.columns, pd.MultiIndex):
        price_field = None
        price_level = None

        for level in range(data.columns.nlevels):
            level_values = data.columns.get_level_values(level)
            if "Adj Close" in level_values:
                price_field = "Adj Close"
                price_level = level
                break
        if price_field is None:
            raise ValueError(
                "Adjusted closing prices were not returned by yfinance."
            )

        closing_prices = data.xs(
            price_field,
            level=price_level,
            axis=1
        ).copy()

        if isinstance(closing_prices, pd.Series):
            closing_prices = closing_prices.to_frame()
    else:
        if "Adj Close" in data.columns:
            closing_prices = data[["Adj Close"]].copy()
        else:
            raise ValueError(
                "Adjusted closing prices were not returned by yfinance."
            )

        if len(closing_prices.columns) == 1:
            ticker = data.attrs.get("ticker")
            if isinstance(ticker, str) and ticker.strip():
                closing_prices.columns = [ticker.strip().upper()]

    requested_tickers = data.attrs.get("requested_tickers")
    if isinstance(requested_tickers, list):
        requested_tickers = [
            ticker.strip().upper()
            for ticker in requested_tickers
            if isinstance(ticker, str) and ticker.strip()
        ]
        if (
            len(requested_tickers) == 1
            and len(closing_prices.columns) == 1
        ):
            closing_prices.columns = requested_tickers
        available_tickers = [
            ticker for ticker in requested_tickers
            if ticker in closing_prices.columns
        ]
        remaining_tickers = [
            ticker for ticker in closing_prices.columns
            if ticker not in available_tickers
        ]
        closing_prices = closing_prices.loc[
            :,
            available_tickers + remaining_tickers
        ]

    closing_prices = closing_prices.sort_index()

    try:
        price_values = closing_prices.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Closing prices must contain only numeric values."
        ) from error

    valid_values = price_values[~np.isnan(price_values)]

    if not np.isfinite(valid_values).all():
        raise ValueError(
            "Closing prices must contain only finite values."
        )

    if (valid_values <= 0).any():
        raise ValueError("Closing prices must be greater than zero.")

    original_row_count = len(closing_prices)
    entirely_missing_tickers = [
        str(ticker)
        for ticker in closing_prices.columns
        if closing_prices[ticker].isna().all()
    ]
    closing_prices = closing_prices.dropna(how="any")

    if closing_prices.empty:
        missing_text = ""
        if entirely_missing_tickers:
            missing_text = (
                " Completely unavailable ticker(s): "
                f"{', '.join(entirely_missing_tickers)}."
            )
        raise ValueError(
            "No valid closing-price data remains." + missing_text
        )

    removed_rows = original_row_count - len(closing_prices)
    closing_prices.attrs["total_rows_before_cleaning"] = (
        original_row_count
    )
    closing_prices.attrs["total_rows_after_cleaning"] = len(
        closing_prices
    )
    closing_prices.attrs["removed_rows"] = removed_rows
    closing_prices.attrs["removed_rows_percentage"] = (
        removed_rows / original_row_count
        if original_row_count
        else 0.0
    )
    closing_prices.attrs["entirely_missing_tickers"] = (
        entirely_missing_tickers
    )
    closing_prices.attrs["actual_start_date"] = str(
        closing_prices.index.min()
    )
    closing_prices.attrs["actual_end_date"] = str(
        closing_prices.index.max()
    )

    return closing_prices