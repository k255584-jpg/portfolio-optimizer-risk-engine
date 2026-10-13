from datetime import date, datetime
import math

import numpy as np
import pandas as pd

from data_loader import download_stock_data, get_closing_prices
from mandate_parser import parse_investor_mandate
from optimizer import optimize_portfolio
from portfolio_math import (
    calculate_annual_returns,
    calculate_covariance_matrix,
    calculate_log_returns,
    calculate_simple_returns,
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio
)
from risk_engine import run_monte_carlo, summarize_simulation


def _validate_positive_number(value, name):
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a number.")

    value = float(value)

    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name} must be finite and greater than zero."
        )

    return value


def _validate_finite_number(value, name):
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a number.")

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return value


def _validate_positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(
        value,
        (int, np.integer)
    ):
        raise TypeError(f"{name} must be a positive integer.")

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")

    return int(value)


def _validate_date_range(start_date, end_date):
    if not isinstance(start_date, str) or not start_date.strip():
        raise ValueError("start_date must be a nonempty string.")

    if not isinstance(end_date, str) or not end_date.strip():
        raise ValueError("end_date must be a nonempty string.")

    try:
        start = datetime.strptime(
            start_date,
            "%Y-%m-%d"
        ).date()
        end = datetime.strptime(
            end_date,
            "%Y-%m-%d"
        ).date()
    except ValueError as error:
        raise ValueError(
            "Dates must use YYYY-MM-DD format."
        ) from error

    if end <= start:
        raise ValueError("end_date must be after start_date.")


def _format_data_date(value):
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return str(value)


def _validate_requested_columns(closing_prices, tickers):
    if not isinstance(closing_prices, pd.DataFrame):
        raise TypeError(
            "get_closing_prices() must return a Pandas DataFrame."
        )

    if len(tickers) == 1 and list(closing_prices.columns) == ["Adj Close"]:
        closing_prices = closing_prices.copy()
        closing_prices.columns = tickers

    actual_tickers = list(closing_prices.columns)
    requested_tickers = list(tickers)

    missing_tickers = [
        ticker for ticker in requested_tickers
        if ticker not in actual_tickers
    ]
    unexpected_tickers = [
        ticker for ticker in actual_tickers
        if ticker not in requested_tickers
    ]

    if missing_tickers:
        raise ValueError(
            f"Historical data is missing ticker(s): "
            f"{', '.join(missing_tickers)}."
        )

    if unexpected_tickers:
        raise ValueError(
            f"Historical data contains unexpected ticker(s): "
            f"{', '.join(unexpected_tickers)}."
        )

    selected_prices = closing_prices.loc[:, requested_tickers].copy()
    selected_prices.attrs = closing_prices.attrs.copy()
    return selected_prices


def _require_adjusted_prices(raw_data):
    if not isinstance(raw_data, pd.DataFrame):
        raise TypeError(
            "download_stock_data() must return a Pandas DataFrame."
        )

    if isinstance(raw_data.columns, pd.MultiIndex):
        has_adjusted_close = any(
            "Adj Close" in raw_data.columns.get_level_values(level)
            for level in range(raw_data.columns.nlevels)
        )
    else:
        has_adjusted_close = "Adj Close" in raw_data.columns

    if not has_adjusted_close:
        raise ValueError(
            "Adjusted closing prices were not returned by the data source. "
            "The analysis will not silently use unadjusted closing prices."
        )


def _validate_optimized_weights(weights, tickers):
    if not isinstance(weights, pd.Series):
        raise TypeError(
            "optimize_portfolio() must return a Pandas Series."
        )

    if list(weights.index) != list(tickers):
        raise ValueError(
            "Optimized weights must use the requested ticker order."
        )

    try:
        values = weights.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Optimized weights must be numeric."
        ) from error

    if not np.isfinite(values).all():
        raise ValueError(
            "Optimized weights must contain only finite values."
        )

    if (values < -1e-8).any() or not np.isclose(
        values.sum(),
        1.0,
        atol=1e-8
    ):
        raise ValueError(
            "Optimized weights must be nonnegative and sum to 1.0."
        )


def analyze_portfolio(
    user_prompt,
    initial_investment=100000,
    start_date="2016-01-01",
    end_date=None,
    risk_free_rate=0.04,
    num_simulations=10000,
    num_days=252,
    random_seed=42,
    progress_callback=None
):
    """
    Run the complete portfolio analysis pipeline.

    The returned statistics are estimates based on historical data,
    annualized simple-return calculations, and hypothetical simulations.
    They are not guarantees of future performance.
    """

    initial_investment = _validate_positive_number(
        initial_investment,
        "initial_investment"
    )
    risk_free_rate = _validate_finite_number(
        risk_free_rate,
        "risk_free_rate"
    )
    num_simulations = _validate_positive_integer(
        num_simulations,
        "num_simulations"
    )
    num_days = _validate_positive_integer(num_days, "num_days")

    if end_date is None:
        end_date = date.today().isoformat()

    _validate_date_range(start_date, end_date)

    if random_seed is not None:
        if isinstance(random_seed, bool) or not isinstance(
            random_seed,
            (int, np.integer)
        ):
            raise TypeError(
                "random_seed must be None or an integer."
            )
        random_seed = int(random_seed)

    def report_progress(message):
        if progress_callback is not None:
            progress_callback(message)

    report_progress("Parsing investment instructions...")
    mandate = parse_investor_mandate(user_prompt)
    tickers = mandate["tickers"]
    max_weight = mandate["max_weight"]
    goal = mandate["goal"]

    report_progress("Downloading historical stock data...")
    raw_data = download_stock_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )
    _require_adjusted_prices(raw_data)

    closing_prices = get_closing_prices(raw_data)
    closing_prices = _validate_requested_columns(
        closing_prices,
        tickers
    )

    data_quality = {
        "total_rows_before_cleaning": closing_prices.attrs.get(
            "total_rows_before_cleaning",
            len(closing_prices)
        ),
        "total_rows_after_cleaning": closing_prices.attrs.get(
            "total_rows_after_cleaning",
            len(closing_prices)
        ),
        "removed_rows": closing_prices.attrs.get("removed_rows", 0),
        "removed_rows_percentage": closing_prices.attrs.get(
            "removed_rows_percentage",
            0.0
        ),
        "entirely_missing_tickers": closing_prices.attrs.get(
            "entirely_missing_tickers",
            []
        )
    }

    removed_rows = data_quality["removed_rows"]

    if len(closing_prices) < 253:
        raise ValueError(
            "At least 253 complete daily price rows are required "
            "to produce 252 daily return observations."
        )

    report_progress("Calculating financial statistics...")
    log_returns = calculate_log_returns(closing_prices)
    simple_returns = calculate_simple_returns(closing_prices)

    if len(log_returns) < 252:
        raise ValueError(
            "At least 252 daily return observations are required "
            "for meaningful analysis."
        )

    annual_returns = calculate_annual_returns(simple_returns)
    covariance_matrix = calculate_covariance_matrix(simple_returns)

    report_progress("Optimizing portfolio...")
    optimized_weights = optimize_portfolio(
        expected_returns=annual_returns,
        covariance_matrix=covariance_matrix,
        max_weight=max_weight,
        goal=goal,
        risk_free_rate=risk_free_rate
    )
    _validate_optimized_weights(optimized_weights, tickers)

    weights_array = optimized_weights.to_numpy()
    expected_portfolio_return = portfolio_return(
        weights_array,
        annual_returns
    )
    volatility = portfolio_volatility(
        weights_array,
        covariance_matrix
    )
    sharpe = sharpe_ratio(
        weights_array,
        annual_returns,
        covariance_matrix,
        risk_free_rate
    )

    report_progress("Running Monte Carlo simulations...")
    simulated_values = run_monte_carlo(
        log_returns=log_returns,
        weights=optimized_weights,
        initial_investment=initial_investment,
        num_simulations=num_simulations,
        num_days=num_days,
        random_seed=random_seed
    )

    risk_summary = summarize_simulation(
        simulated_values,
        initial_investment,
        confidence_level=0.95
    )

    report_progress("Generating analysis report...")
    return {
        "tickers": tickers,
        "goal": goal,
        "max_weight": max_weight,
        "initial_investment": initial_investment,
        "start_date": start_date,
        "end_date": end_date,
        "actual_start_date": _format_data_date(
            closing_prices.index.min()
        ),
        "actual_end_date": _format_data_date(
            closing_prices.index.max()
        ),
        "removed_rows": removed_rows,
        "data_quality": data_quality,
        "risk_free_rate": risk_free_rate,
        "optimized_weights": optimized_weights,
        "expected_return": expected_portfolio_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "num_simulations": num_simulations,
        "num_days": num_days,
        "risk_summary": risk_summary,
        "simulated_values": simulated_values
    }


def _goal_label(goal):
    if goal == "max_sharpe":
        return "Maximum Sharpe Ratio"
    return "Minimum Volatility"


def print_analysis_report(results, currency_label="PKR"):
    """Print a readable command-line report from analyze_portfolio()."""

    tickers = results["tickers"]
    weights = results["optimized_weights"]
    investment = results["initial_investment"]
    risk_summary = results["risk_summary"]
    currency = currency_label.strip().upper() or "PKR"

    print("=" * 40)
    print(" AI PORTFOLIO OPTIMIZER & RISK ENGINE")
    print("=" * 40)
    print("\nINVESTMENT MANDATE")
    print(f"Selected Stocks: {', '.join(tickers)}")
    print(f"Optimization Goal: {_goal_label(results['goal'])}")
    print(
        "Maximum Allocation Per Stock: "
        f"{results['max_weight']:.2%}"
    )
    print(f"Initial Investment: {currency} {investment:,.2f}")
    print(
        "Historical Data Period: "
        f"{results['actual_start_date']} to "
        f"{results['actual_end_date']}"
    )
    print(f"Removed incomplete rows: {results['removed_rows']}")
    data_quality = results.get("data_quality", {})
    removed_percentage = data_quality.get(
        "removed_rows_percentage",
        0.0
    )
    print(
        "Rows retained: "
        f"{data_quality.get('total_rows_after_cleaning', 'unknown')} "
        f"of {data_quality.get('total_rows_before_cleaning', 'unknown')}"
    )
    print(f"Rows removed: {removed_percentage:.2%}")
    if removed_percentage > 0.10:
        print(
            "Warning: More than 10% of historical rows were removed "
            "because of incomplete price data."
        )

    print("\n" + "-" * 40)
    print("OPTIMIZED PORTFOLIO ALLOCATION")
    print("-" * 40)

    for ticker, weight in weights.items():
        allocation = investment * weight
        print(
            f"{ticker}: {weight:.2%} | "
            f"{currency} {allocation:,.2f}"
        )

    print(f"Total Allocation: {weights.sum():.2%}")

    print("\n" + "-" * 40)
    print("PORTFOLIO PERFORMANCE ESTIMATES")
    print("-" * 40)
    print(
        "Estimated Annualized Arithmetic Return: "
        f"{results['expected_return']:.2%}"
    )
    print(f"Annualized Volatility: {results['volatility']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(
        "Note: Return and Sharpe estimates use historical daily "
        "simple returns annualized over 252 trading days."
    )
    print(
        f"Risk-Free Rate: {results['risk_free_rate']:.2%} "
        "(illustrative input)"
    )

    print("\n" + "-" * 40)
    print("MONTE CARLO RISK ANALYSIS")
    print("-" * 40)
    print(
        f"Number of Simulations: "
        f"{results['num_simulations']:,}"
    )
    print(
        f"Simulation Horizon: {results['num_days']} Trading Days"
    )
    print(
        "Mean Final Portfolio Value: "
        f"{currency} {risk_summary['mean_final_value']:,.2f}"
    )
    print(
        "Median Final Portfolio Value: "
        f"{currency} {risk_summary['median_final_value']:,.2f}"
    )
    print(
        f"Probability of Loss: "
        f"{risk_summary['probability_of_loss']:.2%}"
    )
    print(
        f"95% Value at Risk: "
        f"{currency} {risk_summary['var']:,.2f}"
    )
    print(
        f"95% Conditional Value at Risk: "
        f"{currency} {risk_summary['cvar']:,.2f}"
    )

    print("\n" + "-" * 40)
    print("DISCLAIMER")
    print("-" * 40)
    print(
        "Results are based on historical data and statistical "
        "assumptions. Simulated outcomes are hypothetical and are "
        "not guarantees of future performance."
    )
    print(
        "The currency label is used only to display hypothetical "
        "portfolio values. No PKR-to-USD or other currency conversion "
        "is performed, and exchange-rate movements are not modeled."
    )
    print(
        "Transaction fees, brokerage commissions, and taxes are excluded. "
        "The allocation is hypothetical and does not represent an "
        "executable stock purchase."
    )
    if currency != "USD":
        print(
            "Currency warning: Because the selected currency is not USD, "
            "this analysis does not include currency conversion or FX risk."
        )


def _read_investment_amount():
    value = input(
        "Initial investment amount "
        "(press Enter for 100000): "
    ).strip()

    if not value:
        return 100000.0

    try:
        amount = float(value)
    except ValueError as error:
        raise ValueError(
            "Investment amount must be a valid number."
        ) from error

    return _validate_positive_number(
        amount,
        "investment amount"
    )


def _read_risk_free_rate():
    value = input(
        "Annual risk-free rate in % (Enter for 4): "
    ).strip()

    if not value:
        return 0.04

    try:
        percentage = float(value)
    except ValueError as error:
        raise ValueError(
            "Risk-free rate must be a valid number."
        ) from error

    if not math.isfinite(percentage):
        raise ValueError(
            "Risk-free rate must be finite."
        )

    return percentage / 100


if __name__ == "__main__":
    try:
        user_prompt = input(
            "Describe your investment preferences: "
        ).strip()
        initial_investment = _read_investment_amount()
        currency_label = input(
            "Currency label (press Enter for PKR): "
        ).strip() or "PKR"

        risk_free_rate = _read_risk_free_rate()

        results = analyze_portfolio(
            user_prompt=user_prompt,
            initial_investment=initial_investment,
            risk_free_rate=risk_free_rate,
            progress_callback=print
        )

        print_analysis_report(
            results,
            currency_label=currency_label
        )
    except (RuntimeError, TypeError, ValueError, KeyError) as error:
        print(f"\nError: {error}")
