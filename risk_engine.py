import math

import numpy as np
import pandas as pd


SIMULATION_BATCH_SIZE = 1000
MAX_SIMULATIONS = 1_000_000
MAX_OUTPUT_BYTES = 512 * 1024 * 1024
MAX_BATCH_BYTES = 512 * 1024 * 1024


def _validate_simulation_inputs(
    log_returns,
    weights,
    initial_investment,
    num_simulations,
    num_days,
    random_seed
):
    if not isinstance(log_returns, pd.DataFrame):
        raise TypeError("log_returns must be a Pandas DataFrame.")

    if log_returns.empty:
        raise ValueError("log_returns cannot be empty.")

    if len(log_returns) < 2:
        raise ValueError(
            "At least two historical return observations are required."
        )

    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a Pandas Series.")

    if weights.empty:
        raise ValueError("weights cannot be empty.")

    if not weights.index.is_unique:
        raise ValueError("weights must have unique ticker labels.")

    if not log_returns.columns.is_unique:
        raise ValueError(
            "log_returns must have unique ticker labels."
        )

    if set(weights.index) != set(log_returns.columns):
        raise ValueError(
            "Ticker labels in weights and log_returns must match."
        )

    log_returns = log_returns.loc[:, weights.index]

    try:
        return_values = log_returns.to_numpy(dtype=float)
        weight_values = weights.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "log_returns and weights must contain numeric values."
        ) from error

    if not np.isfinite(return_values).all():
        raise ValueError(
            "log_returns must contain only finite values."
        )

    if not np.isfinite(weight_values).all():
        raise ValueError("weights must contain only finite values.")

    if (weight_values < 0).any():
        raise ValueError("weights cannot be negative.")

    if not np.isclose(weight_values.sum(), 1.0, atol=1e-8):
        raise ValueError("weights must add up to 1.0.")

    if isinstance(initial_investment, bool) or not isinstance(
        initial_investment,
        (int, float, np.integer, np.floating)
    ):
        raise TypeError("initial_investment must be a real number.")

    if not math.isfinite(float(initial_investment)):
        raise ValueError(
            "initial_investment must be finite."
        )

    if initial_investment <= 0:
        raise ValueError(
            "initial_investment must be greater than zero."
        )

    for value, name in (
        (num_simulations, "num_simulations"),
        (num_days, "num_days")
    ):
        if isinstance(value, bool) or not isinstance(
            value,
            (int, np.integer)
        ):
            raise TypeError(f"{name} must be a positive integer.")

        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    if num_simulations > MAX_SIMULATIONS:
        raise ValueError(
            f"num_simulations cannot exceed {MAX_SIMULATIONS:,}."
        )

    estimated_output_bytes = (
        (num_days + 1) * num_simulations * np.dtype(float).itemsize
    )
    if estimated_output_bytes > MAX_OUTPUT_BYTES:
        raise ValueError(
            "The requested simulation output is too large. "
            "Reduce num_simulations or num_days."
        )

    estimated_batch_bytes = (
        min(SIMULATION_BATCH_SIZE, num_simulations)
        * num_days
        * log_returns.shape[1]
        * np.dtype(float).itemsize
        * 3
    )
    if estimated_batch_bytes > MAX_BATCH_BYTES:
        raise ValueError(
            "The requested simulation batch is too large for safe "
            "memory use. Reduce num_days or the number of assets."
        )

    if random_seed is not None:
        if isinstance(random_seed, bool) or not isinstance(
            random_seed,
            (int, np.integer)
        ):
            raise TypeError(
                "random_seed must be None or an integer."
            )

    return (
        log_returns,
        weight_values,
        float(initial_investment)
    )


def run_monte_carlo(
    log_returns,
    weights,
    initial_investment=100000,
    num_simulations=10000,
    num_days=252,
    random_seed=42
):
    """
    Simulate hypothetical portfolio values using historical daily returns.

    The simulation estimates possible outcomes from historical daily
    log-return statistics and a multivariate normal distribution assumption.
    It does not guarantee or predict exact future results.
    """

    (
        log_returns,
        weight_values,
        initial_investment
    ) = _validate_simulation_inputs(
        log_returns,
        weights,
        initial_investment,
        num_simulations,
        num_days,
        random_seed
    )

    mean_returns = log_returns.mean().to_numpy(dtype=float)
    daily_covariance = log_returns.cov().to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)

    portfolio_values = np.empty(
        (num_days + 1, num_simulations),
        dtype=float
    )
    portfolio_values[0, :] = initial_investment
    batch_size = max(1, int(SIMULATION_BATCH_SIZE))

    for start in range(0, num_simulations, batch_size):
        end = min(start + batch_size, num_simulations)
        simulated_log_returns = rng.multivariate_normal(
            mean=mean_returns,
            cov=daily_covariance,
            size=(end - start, num_days)
        )
        simulated_simple_returns = np.expm1(simulated_log_returns)
        portfolio_daily_returns = np.sum(
            simulated_simple_returns * weight_values,
            axis=2
        )
        growth_factors = 1.0 + portfolio_daily_returns
        portfolio_values[1:, start:end] = (
            initial_investment
            * np.cumprod(growth_factors, axis=1).T
        )

    return pd.DataFrame(
        portfolio_values,
        index=range(num_days + 1),
        columns=range(1, num_simulations + 1)
    )


def _validate_risk_inputs(
    simulated_values,
    initial_investment,
    confidence_level
):
    if not isinstance(simulated_values, pd.DataFrame):
        raise TypeError(
            "simulated_values must be a Pandas DataFrame."
        )

    if simulated_values.empty:
        raise ValueError(
            "simulated_values cannot be empty."
        )

    try:
        values = simulated_values.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "simulated_values must contain numeric values."
        ) from error

    if not np.isfinite(values).all():
        raise ValueError(
            "simulated_values must contain only finite values."
        )

    if isinstance(initial_investment, bool) or not isinstance(
        initial_investment,
        (int, float, np.integer, np.floating)
    ):
        raise TypeError("initial_investment must be a real number.")

    if not math.isfinite(float(initial_investment)):
        raise ValueError(
            "initial_investment must be finite."
        )

    if initial_investment <= 0:
        raise ValueError(
            "initial_investment must be greater than zero."
        )

    if isinstance(confidence_level, bool) or not isinstance(
        confidence_level,
        (int, float, np.integer, np.floating)
    ):
        raise TypeError("confidence_level must be a real number.")

    if not math.isfinite(float(confidence_level)):
        raise ValueError(
            "confidence_level must be finite."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    return values, float(initial_investment), float(confidence_level)


def _calculate_losses(
    simulated_values,
    initial_investment,
    confidence_level
):
    values, initial_investment, confidence_level = (
        _validate_risk_inputs(
            simulated_values,
            initial_investment,
            confidence_level
        )
    )

    final_values = values[-1, :]
    losses = initial_investment - final_values

    return losses, confidence_level


def calculate_var(
    simulated_values,
    initial_investment,
    confidence_level=0.95
):
    """
    Calculate model-based Value at Risk as a monetary loss threshold.

    VaR is an estimated threshold from the simulation model, not a
    guaranteed maximum loss.
    """

    losses, confidence_level = _calculate_losses(
        simulated_values,
        initial_investment,
        confidence_level
    )

    return float(np.quantile(losses, confidence_level, method="higher"))


def calculate_cvar(
    simulated_values,
    initial_investment,
    confidence_level=0.95
):
    """
    Calculate Conditional Value at Risk, also called Expected Shortfall.

    CVaR is the average simulated loss in the worst tail of outcomes.
    """

    losses, confidence_level = _calculate_losses(
        simulated_values,
        initial_investment,
        confidence_level
    )

    sorted_losses = np.sort(losses)[::-1]
    tail_size = (1.0 - confidence_level) * len(sorted_losses)
    full_values = int(np.floor(tail_size))
    remaining_fraction = tail_size - full_values

    tail_total = sorted_losses[:full_values].sum()

    if remaining_fraction > 0 and full_values < len(sorted_losses):
        tail_total += (
            remaining_fraction * sorted_losses[full_values]
        )

    return float(tail_total / tail_size)


def summarize_simulation(
    simulated_values,
    initial_investment,
    confidence_level=0.95
):
    """Return common summary statistics for simulated portfolio values."""

    losses, confidence_level = _calculate_losses(
        simulated_values,
        initial_investment,
        confidence_level
    )

    final_values = simulated_values.iloc[-1].to_numpy(dtype=float)

    return {
        "initial_investment": float(initial_investment),
        "mean_final_value": float(np.mean(final_values)),
        "median_final_value": float(np.median(final_values)),
        "minimum_final_value": float(np.min(final_values)),
        "maximum_final_value": float(np.max(final_values)),
        "probability_of_loss": float(np.mean(losses > 0)),
        "confidence_level": confidence_level,
        "var": calculate_var(
            simulated_values,
            initial_investment,
            confidence_level
        ),
        "cvar": calculate_cvar(
            simulated_values,
            initial_investment,
            confidence_level
        )
    }
