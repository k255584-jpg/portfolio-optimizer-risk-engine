from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_var_metrics(
    daily_log_returns: pd.DataFrame,
    weights: np.ndarray,
    initial_capital: float,
    confidence_level: float = 0.95,
    n_paths: int = 10_000,
    horizon_days: int = 1,
    random_seed: int = 42,
) -> dict[str, float | np.ndarray]:
    mean_vector = daily_log_returns.mean().to_numpy() * horizon_days
    covariance = daily_log_returns.cov().to_numpy() * horizon_days

    rng = np.random.default_rng(random_seed)
    simulated_asset_log_returns = rng.multivariate_normal(mean_vector, covariance, size=n_paths)
    simulated_portfolio_log_returns = simulated_asset_log_returns @ weights
    simulated_portfolio_simple_returns = np.exp(simulated_portfolio_log_returns) - 1.0
    simulated_pnl = initial_capital * simulated_portfolio_simple_returns
    simulated_losses = -simulated_pnl

    alpha = 1.0 - confidence_level
    z = norm.ppf(alpha)
    portfolio_mean = float(weights @ mean_vector)
    portfolio_std = float(np.sqrt(weights.T @ covariance @ weights))
    parametric_var = initial_capital * max(0.0, -(portfolio_mean + portfolio_std * z))

    historical_portfolio_returns = np.exp(daily_log_returns.to_numpy() @ weights) - 1.0
    historical_losses = -initial_capital * historical_portfolio_returns
    historical_var = float(np.percentile(historical_losses, confidence_level * 100))

    monte_carlo_var = float(np.percentile(simulated_losses, confidence_level * 100))
    tail_losses = simulated_losses[simulated_losses >= monte_carlo_var]
    cvar = float(tail_losses.mean()) if tail_losses.size else monte_carlo_var

    return {
        "parametric_var_95": float(parametric_var),
        "historical_var_95": historical_var,
        "cvar_95": cvar,
        "monte_carlo_var_95": monte_carlo_var,
        "simulated_pnl": simulated_pnl,
        "pnl_var_threshold": float(np.percentile(simulated_pnl, (1 - confidence_level) * 100)),
    }
