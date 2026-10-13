import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numbers import Real

from portfolio_math import (
    negative_sharpe,
    portfolio_volatility
)
def optimize_portfolio(
    expected_returns:pd.Series,
    covariance_matrix:pd.DataFrame,
    max_weight: float,
    goal: str,
    risk_free_rate: float = 0.04
):
    if not isinstance(expected_returns, pd.Series):
        raise TypeError("expected_returns must be a Pandas Series.")
    if not isinstance(covariance_matrix, pd.DataFrame):
        raise TypeError(
            "covariance_matrix must be a Pandas DataFrame."
        )
    number_of_stocks = len(expected_returns)
    if number_of_stocks == 0:
        raise ValueError("At least one stock is required.")
    if not expected_returns.index.is_unique:
        raise ValueError(
            "expected_returns must have unique ticker labels."
        )
    if covariance_matrix.shape != (
        number_of_stocks,
        number_of_stocks
    ):
        raise ValueError(
            "The covariance matrix must have one row and one column "
            "for every stock."
        )
    if not covariance_matrix.index.is_unique:
        raise ValueError(
            "The covariance matrix index must be unique."
        )
    if not covariance_matrix.columns.is_unique:
        raise ValueError(
            "The covariance matrix columns must be unique."
        )
    expected_tickers = list(expected_returns.index)
    if set(covariance_matrix.index) != set(expected_tickers):
        raise ValueError(
            "Covariance-matrix row labels must match expected-return labels."
        )
    if set(covariance_matrix.columns) != set(expected_tickers):
        raise ValueError(
            "Covariance-matrix column labels must match expected-return labels."
        )
    covariance_matrix = covariance_matrix.loc[
        expected_tickers,
        expected_tickers
    ]
    expected_return_values = expected_returns.to_numpy(dtype=float)
    covariance_values = covariance_matrix.to_numpy(dtype=float)
    if not np.isfinite(expected_return_values).all():
        raise ValueError(
            "expected_returns must contain only finite values."
        )
    if not np.isfinite(covariance_values).all():
        raise ValueError(
            "covariance_matrix must contain only finite values."
        )
    if isinstance(max_weight, bool) or not isinstance(max_weight, Real):
        raise TypeError("max_weight must be a real number.")

    if not np.isfinite(max_weight):
        raise ValueError("max_weight must be finite.")
    if not 0 < max_weight <= 1:
        raise ValueError(
            "max_weight must be greater than 0 and no greater than 1."
        )
    if isinstance(risk_free_rate, bool) or not isinstance(
        risk_free_rate,
        Real
    ):
        raise TypeError("risk_free_rate must be a real number.")

    if not np.isfinite(risk_free_rate):
        raise ValueError(
            "risk_free_rate must be finite."
        )
    if number_of_stocks * max_weight < 1 - 1e-12:
        raise ValueError(
            "The maximum weight is too small to allocate 100% "
            "of the portfolio."
        )
    if goal not in {"max_sharpe", "min_vol"}:
        raise ValueError(
            "goal must be either 'max_sharpe' or 'min_vol'."
        )
    initial_weights = np.ones(number_of_stocks) / number_of_stocks
    bounds = [(0.0, max_weight)] * number_of_stocks
    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1.0
    }
    if goal == "max_sharpe":
        objective_function = lambda weights: negative_sharpe(
            weights=weights,
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            risk_free_rate=risk_free_rate
        )
    else:
        objective_function = lambda weights: portfolio_volatility(
            weights=weights,
            covariance_matrix=covariance_matrix
        )
    result = minimize(
        fun=objective_function,
        x0=initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    if not result.success:
        raise ValueError(
            f"Portfolio optimization failed: {result.message}"
        )
    optimized_weights = np.asarray(result.x, dtype=float)
    if not np.isfinite(optimized_weights).all():
        raise ValueError(
            "The optimizer returned non-finite weights."
        )
    tolerance = 1e-8
    weights_are_valid = (
        (optimized_weights >= -tolerance).all()
        and (optimized_weights <= max_weight + tolerance).all()
        and np.isclose(
            optimized_weights.sum(),
            1.0,
            atol=tolerance
        )
    )
    if not weights_are_valid:
        raise ValueError(
            "The optimizer returned weights that violate the "
            "portfolio constraints."
        )
    return pd.Series(
        optimized_weights,
        index=expected_returns.index
    )

