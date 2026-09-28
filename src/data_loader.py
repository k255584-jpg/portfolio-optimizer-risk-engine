from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class ReturnStatistics:
    daily_log_returns: pd.DataFrame
    annual_returns: pd.Series
    annual_covariance: pd.DataFrame


def fetch_adjusted_close_prices(
    tickers: Sequence[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if not tickers:
        raise ValueError("At least one ticker must be provided.")

    raw = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        raise ValueError("No market data returned for the selected inputs.")

    adj_close = raw["Adj Close"] if "Adj Close" in raw else raw
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(name=tickers[0])

    prices = adj_close.dropna(how="any")
    if prices.empty:
        raise ValueError("Adjusted close data is empty after dropping missing values.")

    return prices


def compute_return_statistics(prices: pd.DataFrame, trading_days: int = 252) -> ReturnStatistics:
    if prices.shape[0] < 2:
        raise ValueError("At least two rows of price data are required.")

    daily_log_returns = np.log(prices / prices.shift(1)).dropna(how="any")
    annual_returns = daily_log_returns.mean() * trading_days
    annual_covariance = daily_log_returns.cov() * trading_days

    return ReturnStatistics(
        daily_log_returns=daily_log_returns,
        annual_returns=annual_returns,
        annual_covariance=annual_covariance,
    )
