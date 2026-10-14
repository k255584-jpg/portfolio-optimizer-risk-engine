from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    max_sharpe_point: tuple[float, float],
    min_vol_point: tuple[float, float],
) -> go.Figure:
    fig = go.Figure()

    if not frontier.empty:
        fig.add_trace(
            go.Scatter(
                x=frontier["volatility"],
                y=frontier["return"],
                mode="markers",
                name="Efficient Frontier",
                marker={"color": frontier.get("sharpe", pd.Series(np.zeros(len(frontier)))), "colorscale": "Viridis", "showscale": True},
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[max_sharpe_point[0]],
            y=[max_sharpe_point[1]],
            mode="markers",
            name="Max Sharpe",
            marker={"size": 12, "color": "green", "symbol": "star"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_vol_point[0]],
            y=[min_vol_point[1]],
            mode="markers",
            name="Min Volatility",
            marker={"size": 12, "color": "orange", "symbol": "diamond"},
        )
    )

    fig.update_layout(
        title="Markowitz Efficient Frontier",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        template="plotly_white",
    )
    return fig


def plot_allocation_donut(weights: np.ndarray, labels: list[str], title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=weights,
                hole=0.55,
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(title=title)
    return fig


def plot_monte_carlo_distribution(simulated_pnl: np.ndarray, pnl_var_threshold: float) -> go.Figure:
    fig = go.Figure(data=[go.Histogram(x=simulated_pnl, nbinsx=60, name="Simulated P&L")])
    fig.add_vline(
        x=pnl_var_threshold,
        line_color="red",
        line_width=2,
        line_dash="dash",
        annotation_text="95% VaR Threshold",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Monte Carlo P&L Distribution",
        xaxis_title="Portfolio P&L",
        yaxis_title="Frequency",
        template="plotly_white",
    )
    return fig
