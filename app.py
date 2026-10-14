from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import compute_return_statistics, fetch_adjusted_close_prices
from src.optimizer import generate_efficient_frontier, optimize_portfolio
from src.risk_engine import compute_var_metrics
from src.visualizer import (
    plot_allocation_donut,
    plot_efficient_frontier,
    plot_monte_carlo_distribution,
)

st.set_page_config(page_title="Portfolio Optimizer & Risk Engine", layout="wide")
st.title("Quantitative Portfolio Optimization & Risk Engine")

st.sidebar.header("Portfolio Inputs")
available_tickers = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
    "XOM",
    "JNJ",
]
selected_tickers = st.sidebar.multiselect("Stock Tickers", options=available_tickers, default=available_tickers[:4])
lookback_days = st.sidebar.slider("Lookback (days)", min_value=90, max_value=2520, value=756, step=21)
initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1_000.0, value=1_000_000.0, step=10_000.0)
max_weight = st.sidebar.slider("Maximum Asset Weight", min_value=0.10, max_value=1.00, value=0.50, step=0.05)
risk_free_rate = st.sidebar.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.15, value=0.02, step=0.005)

end_date = date.today()
start_date = end_date - timedelta(days=int(lookback_days))

if not selected_tickers:
    st.info("Select at least one ticker to begin.")
    st.stop()

if max_weight * len(selected_tickers) < 1.0:
    st.error("Maximum asset weight cap is infeasible for the number of selected assets.")
    st.stop()

try:
    prices = fetch_adjusted_close_prices(selected_tickers, start_date, end_date)
    stats = compute_return_statistics(prices)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

max_sharpe = optimize_portfolio(
    annual_returns=stats.annual_returns,
    annual_covariance=stats.annual_covariance,
    bounds=(0.0, max_weight),
    objective="max_sharpe",
    risk_free_rate=risk_free_rate,
)

min_volatility = optimize_portfolio(
    annual_returns=stats.annual_returns,
    annual_covariance=stats.annual_covariance,
    bounds=(0.0, max_weight),
    objective="min_volatility",
    risk_free_rate=risk_free_rate,
)

if not max_sharpe.success:
    st.warning(f"Max Sharpe optimization warning: {max_sharpe.message}")
if not min_volatility.success:
    st.warning(f"Min volatility optimization warning: {min_volatility.message}")

frontier = generate_efficient_frontier(stats.annual_returns, stats.annual_covariance, bounds=(0.0, max_weight), points=60)

risk_metrics = compute_var_metrics(
    daily_log_returns=stats.daily_log_returns,
    weights=max_sharpe.weights,
    initial_capital=initial_capital,
    confidence_level=0.95,
    n_paths=10_000,
)

summary = pd.DataFrame(
    {
        "Portfolio": ["Max Sharpe", "Min Volatility"],
        "Expected Return": [max_sharpe.expected_return, min_volatility.expected_return],
        "Volatility": [max_sharpe.volatility, min_volatility.volatility],
        "Sharpe Ratio": [max_sharpe.sharpe_ratio, min_volatility.sharpe_ratio],
    }
)

st.subheader("Optimization Summary")
st.dataframe(summary.style.format({"Expected Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe Ratio": "{:.2f}"}), use_container_width=True)

w_df = pd.DataFrame(
    {
        "Ticker": selected_tickers,
        "Max Sharpe Weight": max_sharpe.weights,
        "Min Vol Weight": min_volatility.weights,
    }
)
st.subheader("Portfolio Weights")
st.dataframe(w_df.style.format({"Max Sharpe Weight": "{:.2%}", "Min Vol Weight": "{:.2%}"}), use_container_width=True)

st.subheader("Risk Metrics (95%)")
col1, col2, col3 = st.columns(3)
col1.metric("Parametric VaR", f"${risk_metrics['parametric_var_95']:,.0f}")
col2.metric("Historical VaR", f"${risk_metrics['historical_var_95']:,.0f}")
col3.metric("CVaR / Expected Shortfall", f"${risk_metrics['cvar_95']:,.0f}")

chart_col1, chart_col2 = st.columns(2)
chart_col1.plotly_chart(
    plot_allocation_donut(max_sharpe.weights, selected_tickers, "Max Sharpe Allocation"),
    use_container_width=True,
)
chart_col2.plotly_chart(
    plot_allocation_donut(min_volatility.weights, selected_tickers, "Minimum Volatility Allocation"),
    use_container_width=True,
)

st.plotly_chart(
    plot_efficient_frontier(
        frontier,
        max_sharpe_point=(max_sharpe.volatility, max_sharpe.expected_return),
        min_vol_point=(min_volatility.volatility, min_volatility.expected_return),
    ),
    use_container_width=True,
)

st.plotly_chart(
    plot_monte_carlo_distribution(
        simulated_pnl=np.asarray(risk_metrics["simulated_pnl"]),
        pnl_var_threshold=float(risk_metrics["pnl_var_threshold"]),
    ),
    use_container_width=True,
)
