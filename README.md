# portfolio-optimizer-risk-engine

Institutional-grade quantitative analytics dashboard implementing Modern Portfolio Theory (MPT), SciPy SLSQP optimization, and Monte Carlo VaR/CVaR risk modeling in Streamlit.

## Architecture

```text
portfolio-optimizer-risk-engine/
├── app.py
├── requirements.txt
└── src/
    ├── data_loader.py
    ├── optimizer.py
    ├── risk_engine.py
    └── visualizer.py
```

### Component Responsibilities

- **`app.py`**: Streamlit dashboard and user inputs (tickers, lookback, initial capital, max weight cap, risk-free rate).
- **`src/data_loader.py`**: Fetches adjusted close prices from Yahoo Finance and computes daily log returns, annualized expected returns, and annualized covariance matrix.
- **`src/optimizer.py`**: Markowitz optimizer using `scipy.optimize.minimize(..., method="SLSQP")` for:
  - Maximum Sharpe Ratio portfolio
  - Minimum Volatility portfolio
  - Efficient frontier points
- **`src/risk_engine.py`**: 10,000-path multivariate normal Monte Carlo simulation for risk analytics:
  - 95% Parametric VaR
  - 95% Historical VaR
  - 95% CVaR (Expected Shortfall)
- **`src/visualizer.py`**: Plotly visuals:
  - Efficient frontier scatter
  - Allocation donut charts
  - Monte Carlo P&L histogram with red 95% VaR threshold line

## Mathematical Formulation

Let:
- \(\mu \in \mathbb{R}^n\): annualized expected returns
- \(\Sigma \in \mathbb{R}^{n\times n}\): annualized covariance matrix
- \(w \in \mathbb{R}^n\): portfolio weights

### Constraints

- Budget: \(\sum_{i=1}^{n} w_i = 1\)
- Bounds: \(0 \le w_i \le w_{\max}\)

### Portfolio Return and Volatility

- Expected return: \(\mu_p = w^\top \mu\)
- Volatility: \(\sigma_p = \sqrt{w^\top \Sigma w}\)

### Objectives

- **Max Sharpe**:
  \[
  \max_w \frac{\mu_p - r_f}{\sigma_p}
  \]
- **Min Volatility**:
  \[
  \min_w \sigma_p
  \]

### Risk Metrics (95%)

- **Parametric VaR**:
  \[
  \text{VaR}_{0.95}^{\text{param}} = V_0\,\max\left(0, -\left(\mu_p + z_{0.05}\sigma_p\right)\right)
  \]
- **Historical VaR**: empirical 95th percentile of historical loss distribution.
- **CVaR / Expected Shortfall**: expected loss conditional on losses exceeding (or equaling) VaR.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then configure in the sidebar:
1. Stock tickers
2. Lookback horizon
3. Initial capital
4. Maximum per-asset weight cap
5. Risk-free rate

## Notes

- Data source: Yahoo Finance via `yfinance`.
- The engine uses daily log returns and annualizes with 252 trading days.
- Optimization feasibility requires `max_weight * number_of_assets >= 1`.
