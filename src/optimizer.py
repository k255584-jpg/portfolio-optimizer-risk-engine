from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class OptimizationResult:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    success: bool
    message: str


def portfolio_performance(
    weights: np.ndarray,
    annual_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float = 0.0,
) -> tuple[float, float, float]:
    expected_return = float(weights @ annual_returns)
    volatility = float(np.sqrt(weights.T @ covariance @ weights))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else np.nan
    return expected_return, volatility, float(sharpe_ratio)


def optimize_portfolio(
    annual_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    bounds: tuple[float, float],
    objective: str,
    risk_free_rate: float = 0.0,
) -> OptimizationResult:
    n_assets = len(annual_returns)
    x0 = np.full(n_assets, 1.0 / n_assets)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    weight_bounds = tuple(bounds for _ in range(n_assets))

    mean_arr = annual_returns.to_numpy()
    cov_arr = annual_covariance.to_numpy()

    def neg_sharpe(weights: np.ndarray) -> float:
        _, vol, sr = portfolio_performance(weights, mean_arr, cov_arr, risk_free_rate)
        return -sr if vol > 0 else 1e6

    def volatility(weights: np.ndarray) -> float:
        return portfolio_performance(weights, mean_arr, cov_arr, risk_free_rate)[1]

    objective_fn = neg_sharpe if objective == "max_sharpe" else volatility

    result = minimize(
        objective_fn,
        x0=x0,
        method="SLSQP",
        bounds=weight_bounds,
        constraints=constraints,
    )

    weights = result.x if result.success else x0
    expected_return, vol, sharpe = portfolio_performance(weights, mean_arr, cov_arr, risk_free_rate)

    return OptimizationResult(
        weights=weights,
        expected_return=expected_return,
        volatility=vol,
        sharpe_ratio=sharpe,
        success=bool(result.success),
        message=str(result.message),
    )


def generate_efficient_frontier(
    annual_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    bounds: tuple[float, float],
    points: int = 40,
) -> pd.DataFrame:
    n_assets = len(annual_returns)
    x0 = np.full(n_assets, 1.0 / n_assets)
    mean_arr = annual_returns.to_numpy()
    cov_arr = annual_covariance.to_numpy()

    target_returns = np.linspace(float(mean_arr.min()), float(mean_arr.max()), points)
    frontier = []

    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: float(w @ mean_arr) - t},
        )
        result = minimize(
            lambda w: float(np.sqrt(w.T @ cov_arr @ w)),
            x0=x0,
            method="SLSQP",
            bounds=tuple(bounds for _ in range(n_assets)),
            constraints=constraints,
        )

        if result.success:
            frontier.append(
                {
                    "return": float(result.x @ mean_arr),
                    "volatility": float(np.sqrt(result.x.T @ cov_arr @ result.x)),
                    "sharpe": float((result.x @ mean_arr) / np.sqrt(result.x.T @ cov_arr @ result.x)),
                }
            )

    return pd.DataFrame(frontier)
