import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def calculate_log_returns(prices: pd.DataFrame):

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a Pandas DataFrame.")

    if prices.empty:
        raise ValueError("The prices DataFrame cannot be empty.")

    try:
        price_values = prices.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Prices must contain only numeric values.") from error

    if not np.isfinite(price_values).all():
        raise ValueError("Prices must contain only finite values.")

    if (price_values <= 0).any():
        raise ValueError("All prices must be greater than zero.")

    log_returns = np.log(prices / prices.shift(1))

    log_returns = log_returns.dropna()

    if log_returns.empty:
        raise ValueError(
            "At least two price rows are required to calculate returns."
        )

    return log_returns


def calculate_simple_returns(prices: pd.DataFrame):
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a Pandas DataFrame.")

    if prices.empty:
        raise ValueError("The prices DataFrame cannot be empty.")

    try:
        price_values = prices.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Prices must contain only numeric values.") from error

    if not np.isfinite(price_values).all():
        raise ValueError("Prices must contain only finite values.")

    if (price_values <= 0).any():
        raise ValueError("All prices must be greater than zero.")

    if len(prices) < 2:
        raise ValueError(
            "At least two price rows are required to calculate returns."
        )

    simple_returns = prices.pct_change().dropna()

    if simple_returns.empty:
        raise ValueError("No simple returns could be calculated.")

    return simple_returns


def calculate_annual_returns(
    returns: pd.DataFrame
):

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a Pandas DataFrame.")

    if returns.empty:
        raise ValueError("The daily simple returns DataFrame cannot be empty.")

    try:
        return_values = returns.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Daily simple returns must contain only numeric values."
        ) from error

    if not np.isfinite(return_values).all():
        raise ValueError(
            "Daily simple returns must contain only finite values."
        )

    average_daily_returns = returns.mean()

    annual_returns = average_daily_returns * TRADING_DAYS_PER_YEAR

    return annual_returns


def calculate_covariance_matrix(
    returns: pd.DataFrame
):
    """
    Calculate the annualized covariance matrix of stock returns.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a Pandas DataFrame.")

    if returns.empty:
        raise ValueError("The daily simple returns DataFrame cannot be empty.")

    if len(returns) < 2:
        raise ValueError(
            "At least two return observations are required "
            "to calculate covariance."
        )

    try:
        return_values = returns.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Daily simple returns must contain only numeric values."
        ) from error

    if not np.isfinite(return_values).all():
        raise ValueError(
            "Daily simple returns must contain only finite values."
        )

    daily_covariance = returns.cov()

    annualized_covariance = (
        daily_covariance * TRADING_DAYS_PER_YEAR
    )

    return annualized_covariance


def portfolio_return(
    weights: np.ndarray,
    expected_returns: pd.Series
):
    """
    Calculate the expected annual return of a portfolio.
    """

    weights = np.asarray(weights, dtype=float)
    expected_returns = np.asarray(expected_returns, dtype=float)

    if len(weights) != len(expected_returns):
        raise ValueError(
            "The number of weights must match the number of returns."
        )

    if not np.isfinite(weights).all():
        raise ValueError("weights must contain only finite values.")

    if not np.isfinite(expected_returns).all():
        raise ValueError(
            "expected_returns must contain only finite values."
        )

    return float(np.dot(weights, expected_returns))


def portfolio_volatility(
    weights: np.ndarray,
    covariance_matrix: pd.DataFrame
):

    weights = np.asarray(weights, dtype=float)
    covariance_values = np.asarray(covariance_matrix, dtype=float)

    if covariance_values.ndim != 2:
        raise ValueError("covariance_matrix must be two-dimensional.")

    if covariance_values.shape[0] != covariance_values.shape[1]:
        raise ValueError("covariance_matrix must be square.")

    if len(weights) != covariance_values.shape[0]:
        raise ValueError(
            "The number of weights must match the covariance matrix."
        )

    if not np.isfinite(weights).all():
        raise ValueError("weights must contain only finite values.")

    if not np.isfinite(covariance_values).all():
        raise ValueError(
            "covariance_matrix must contain only finite values."
        )

    portfolio_variance = (
        weights.T @ covariance_values @ weights
    )

    if portfolio_variance < -1e-12:
        raise ValueError(
            "The covariance matrix produced a negative portfolio variance."
        )

    return float(np.sqrt(portfolio_variance))


def sharpe_ratio(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float
):

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")

    expected_portfolio_return = portfolio_return(
        weights,
        expected_returns
    )

    expected_portfolio_volatility = portfolio_volatility(
        weights,
        covariance_matrix
    )

    if expected_portfolio_volatility == 0:
        raise ValueError(
            "Sharpe ratio is undefined because portfolio volatility is "
            "zero. Use portfolio return instead."
        )

    return float(
        (expected_portfolio_return - risk_free_rate)
        / expected_portfolio_volatility
    )


def negative_sharpe(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float
):
    """
    Return the negative Sharpe ratio for use by an optimizer.

    Optimization libraries minimize functions. Returning the negative
    Sharpe ratio allows the optimizer to maximize the actual Sharpe ratio.
    """

    return -sharpe_ratio(
        weights,
        expected_returns,
        covariance_matrix,
        risk_free_rate
    )
