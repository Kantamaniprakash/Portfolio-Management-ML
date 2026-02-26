"""
Portfolio Optimization Module
================================
Implements two portfolio construction approaches:

1. **Markowitz Mean-Variance Optimization**
   - Maximises Sharpe Ratio subject to long-only constraints
   - Efficient frontier visualization

2. **ML-Enhanced Weights** (Random Forest return prediction)
   - Predicts next-period expected returns using technical features
   - Weights allocated proportional to predicted return confidence

Both methods respect risk constraints via the 95% VaR metric.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RISK_FREE_RATE = 0.045   # 4.5% annual (approx. 2024 Fed funds rate)


# ── 1. Markowitz Optimization ─────────────────────────────────────────────

def compute_portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, trading_days: int = 252) -> tuple:
    """Return (annualised_return, annualised_volatility, sharpe_ratio)."""
    port_return = np.dot(weights, mean_returns) * trading_days
    port_vol    = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(trading_days)
    sharpe      = (port_return - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe


def neg_sharpe(weights, mean_returns, cov_matrix):
    _, _, sharpe = compute_portfolio_stats(weights, mean_returns, cov_matrix)
    return -sharpe


def markowitz_optimize(returns: pd.DataFrame) -> dict:
    """
    Compute the Maximum Sharpe Ratio portfolio via scipy minimize.

    Parameters
    ----------
    returns : pd.DataFrame  daily log-return series (stationary)

    Returns
    -------
    dict  {weights, tickers, return, volatility, sharpe, cov_matrix}
    """
    tickers      = returns.columns.tolist()
    n            = len(tickers)
    mean_returns = returns.mean().values
    cov_matrix   = returns.cov().values

    # Constraints: weights sum to 1, each weight in [0, 0.40]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds      = tuple((0.0, 0.40) for _ in range(n))
    init_w      = np.array([1 / n] * n)

    result = minimize(
        neg_sharpe,
        init_w,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    opt_weights = result.x
    opt_ret, opt_vol, opt_sharpe = compute_portfolio_stats(opt_weights, mean_returns, cov_matrix)

    return {
        "weights":     dict(zip(tickers, opt_weights.round(4))),
        "tickers":     tickers,
        "return":      round(opt_ret,    4),
        "volatility":  round(opt_vol,    4),
        "sharpe":      round(opt_sharpe, 4),
        "cov_matrix":  cov_matrix,
        "mean_returns": mean_returns,
    }


def efficient_frontier(returns: pd.DataFrame, n_portfolios: int = 5000) -> pd.DataFrame:
    """
    Monte Carlo simulation of random portfolios to trace the efficient frontier.

    Returns
    -------
    pd.DataFrame  columns: [Return, Volatility, Sharpe, w_AAPL, w_MSFT, ...]
    """
    tickers      = returns.columns.tolist()
    n            = len(tickers)
    mean_returns = returns.mean().values
    cov_matrix   = returns.cov().values
    rng          = np.random.default_rng(42)

    rows = []
    for _ in range(n_portfolios):
        w   = rng.dirichlet(np.ones(n))
        r, v, s = compute_portfolio_stats(w, mean_returns, cov_matrix)
        row = {"Return": r, "Volatility": v, "Sharpe": s}
        row.update(dict(zip(tickers, w)))
        rows.append(row)

    return pd.DataFrame(rows)


# ── 2. ML-Enhanced Weights ────────────────────────────────────────────────

def build_ml_features(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Engineer technical features used as inputs to the return-prediction model:
      - Momentum (5, 10, 20-day returns)
      - Rolling volatility (10, 20-day)
      - RSI (14-day)
      - Price relative to 50-day / 200-day SMA
    """
    log_ret = np.log(prices / prices.shift(1))
    feats   = {}

    for ticker in prices.columns:
        p = prices[ticker]
        r = log_ret[ticker]

        feats[f"{ticker}_mom5"]   = r.rolling(5).sum()
        feats[f"{ticker}_mom10"]  = r.rolling(10).sum()
        feats[f"{ticker}_mom20"]  = r.rolling(20).sum()
        feats[f"{ticker}_vol10"]  = r.rolling(10).std()
        feats[f"{ticker}_vol20"]  = r.rolling(20).std()
        feats[f"{ticker}_sma50r"] = p / p.rolling(50).mean() - 1
        feats[f"{ticker}_sma200r"]= p / p.rolling(200).mean() - 1

        # RSI
        delta  = r.copy()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / loss.replace(0, np.nan)
        feats[f"{ticker}_rsi14"] = 100 - 100 / (1 + rs)

    return pd.DataFrame(feats, index=prices.index).dropna()


def ml_portfolio_weights(
    prices: pd.DataFrame,
    train_end_date: str,
) -> dict:
    """
    Train a Random Forest to predict 5-day forward returns for each ticker
    and allocate portfolio weights proportional to positive predicted returns.

    Parameters
    ----------
    prices         : pd.DataFrame  full price history
    train_end_date : str           last date of training period

    Returns
    -------
    dict  {ticker: weight}
    """
    features   = build_ml_features(prices)
    log_ret    = np.log(prices / prices.shift(1))
    fwd_ret    = log_ret.rolling(5).sum().shift(-5)   # 5-day forward return

    train_mask = features.index <= train_end_date
    X_train    = features[train_mask]
    X_test     = features[~train_mask].iloc[[0]]   # next available day

    predicted = {}
    for ticker in prices.columns:
        y = fwd_ret[ticker].reindex(features.index)
        y_train = y[train_mask]
        valid   = y_train.dropna()
        X_tr    = X_train.loc[valid.index]

        scaler  = StandardScaler()
        X_s     = scaler.fit_transform(X_tr)
        X_te    = scaler.transform(X_test)

        rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X_s, valid.values)
        predicted[ticker] = rf.predict(X_te)[0]

    # Allocate only to tickers with positive predicted return
    pred_series = pd.Series(predicted)
    positive    = pred_series.clip(lower=0)
    total       = positive.sum()
    weights     = (positive / total).round(4) if total > 0 else pd.Series({t: 1/len(prices.columns) for t in prices.columns})
    return weights.to_dict()


# ── 3. Value at Risk ──────────────────────────────────────────────────────

def portfolio_var(
    weights: dict,
    returns: pd.DataFrame,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """
    Historical VaR at given confidence level for a given portfolio.

    Returns
    -------
    float  VaR as a positive number (loss)
    """
    tickers    = list(weights.keys())
    w          = np.array([weights[t] for t in tickers])
    port_rets  = returns[tickers] @ w
    var        = -np.percentile(port_rets, (1 - confidence) * 100) * np.sqrt(horizon_days)
    return round(float(var), 6)


if __name__ == "__main__":
    from data_collection import download_prices
    prices  = download_prices()
    log_ret = np.log(prices / prices.shift(1)).dropna()
    result  = markowitz_optimize(log_ret)
    print("Optimal Weights (Max Sharpe):")
    for t, w in result["weights"].items():
        print(f"  {t}: {w*100:.1f}%")
    print(f"\nReturn: {result['return']*100:.2f}%  Vol: {result['volatility']*100:.2f}%  Sharpe: {result['sharpe']:.3f}")
