"""
Backtesting Engine
===================
Simulates the combined ARIMA + NLP trading strategy on historical data
and computes all standard performance metrics.

Strategy logic (per ticker, per day):
  combined_signal = α * arima_signal + (1 - α) * nlp_signal
  if combined_signal > 0.3  → BUY  (long position)
  if combined_signal < -0.3 → SELL (exit / short)
  else                       → HOLD (maintain position)

Portfolio is rebalanced daily using Markowitz optimal weights.
Performance compared against an equal-weight buy-and-hold benchmark.
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

TRANSACTION_COST = 0.001   # 0.1% per trade (realistic for retail)
INITIAL_CAPITAL  = 100_000


def combined_signal(
    arima_signals:  pd.DataFrame,
    nlp_signals:    pd.DataFrame,
    arima_weight:   float = 0.55,
    signal_threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Blend ARIMA and NLP signals into a single composite signal.

    Parameters
    ----------
    arima_signals    : pd.DataFrame  values in {-1, 0, +1}
    nlp_signals      : pd.DataFrame  values in {-1, 0, +1}
    arima_weight     : float         weight on ARIMA (1 - arima_weight on NLP)
    signal_threshold : float         threshold to trigger BUY/SELL

    Returns
    -------
    pd.DataFrame  composite signal in {-1, 0, +1}
    """
    # Align indices
    common_idx = arima_signals.index.intersection(nlp_signals.index)
    common_col = [c for c in arima_signals.columns if c in nlp_signals.columns]

    a = arima_signals.loc[common_idx, common_col]
    n = nlp_signals.loc[common_idx, common_col]

    blended = arima_weight * a + (1 - arima_weight) * n

    signal_df = pd.DataFrame(0, index=blended.index, columns=blended.columns)
    signal_df[blended >  signal_threshold] =  1
    signal_df[blended < -signal_threshold] = -1

    return signal_df.astype(int)


def backtest(
    prices:      pd.DataFrame,
    signals:     pd.DataFrame,
    opt_weights: dict,
    initial_capital: float = INITIAL_CAPITAL,
    tc: float = TRANSACTION_COST,
) -> dict:
    """
    Simulate daily portfolio performance under the combined trading strategy.

    Parameters
    ----------
    prices          : pd.DataFrame  daily closing prices
    signals         : pd.DataFrame  composite signals {-1,0,+1}
    opt_weights     : dict          Markowitz weights per ticker
    initial_capital : float
    tc              : float         transaction cost per trade (fraction)

    Returns
    -------
    dict  with equity curve, daily returns, metrics, trade log
    """
    tickers = [t for t in signals.columns if t in prices.columns]
    common  = signals.index.intersection(prices.index)
    sig     = signals.loc[common, tickers]
    px      = prices.loc[common, tickers]

    n_assets     = len(tickers)
    base_weights = np.array([opt_weights.get(t, 1/n_assets) for t in tickers])
    base_weights = base_weights / base_weights.sum()

    portfolio_value = initial_capital
    prev_position   = np.zeros(n_assets)   # 0 = no position
    equity_curve    = []
    trade_log       = []
    daily_returns   = []

    for i in range(1, len(common)):
        date      = common[i]
        prev_date = common[i - 1]
        price_now  = px.loc[date].values
        price_prev = px.loc[prev_date].values

        # Today's signal determines tomorrow's position
        today_sig = sig.loc[date].values    # {-1, 0, 1}

        # Effective position: 1 = long, 0 = flat, -1 = short
        position = np.where(today_sig == 1, 1,
                   np.where(today_sig == -1, 0, prev_position))

        # Adjust weights by signal strength
        active_weights = base_weights * (position > 0)
        w_sum = active_weights.sum()
        active_weights = active_weights / w_sum if w_sum > 0 else np.ones(n_assets) / n_assets

        # Daily return of the portfolio
        asset_rets = (price_now / price_prev) - 1
        port_ret   = np.dot(active_weights, asset_rets)

        # Transaction cost on trades
        trades        = np.sum(position != prev_position)
        cost          = trades * tc
        net_ret       = port_ret - cost

        portfolio_value *= (1 + net_ret)
        equity_curve.append(portfolio_value)
        daily_returns.append(net_ret)

        if trades > 0:
            trade_log.append({
                "Date":           date,
                "Trades":         int(trades),
                "Portfolio_Value":round(portfolio_value, 2),
                "Daily_Return":   round(net_ret, 4),
            })

        prev_position = position

    equity_series  = pd.Series(equity_curve, index=common[1:], name="Portfolio_Value")
    returns_series = pd.Series(daily_returns, index=common[1:], name="Daily_Return")
    trade_df       = pd.DataFrame(trade_log)

    metrics = compute_metrics(equity_series, returns_series, initial_capital)
    return {
        "equity_curve":  equity_series,
        "daily_returns": returns_series,
        "metrics":       metrics,
        "trade_log":     trade_df,
    }


def compute_metrics(equity: pd.Series, returns: pd.Series, initial_capital: float = INITIAL_CAPITAL) -> dict:
    """Compute standard portfolio performance metrics."""
    total_return   = (equity.iloc[-1] / initial_capital - 1) * 100
    n_years        = len(returns) / 252
    cagr           = ((equity.iloc[-1] / initial_capital) ** (1 / n_years) - 1) * 100
    ann_vol        = returns.std() * np.sqrt(252) * 100
    rf_daily       = 0.045 / 252
    excess         = returns - rf_daily
    sharpe         = (excess.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    # Max drawdown
    cum_max        = equity.cummax()
    drawdown       = (equity - cum_max) / cum_max
    max_drawdown   = drawdown.min() * 100

    # Calmar ratio
    calmar         = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate
    win_rate       = (returns > 0).mean() * 100

    return {
        "Total_Return_%":     round(total_return, 2),
        "CAGR_%":             round(cagr,          2),
        "Ann_Volatility_%":   round(ann_vol,        2),
        "Sharpe_Ratio":       round(sharpe,         4),
        "Max_Drawdown_%":     round(max_drawdown,   2),
        "Calmar_Ratio":       round(calmar,         4),
        "Win_Rate_%":         round(win_rate,       2),
        "Total_Trades":       "-",
    }


def benchmark_buy_hold(prices: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> pd.Series:
    """Equal-weight buy-and-hold benchmark equity curve."""
    n     = prices.shape[1]
    alloc = initial_capital / n
    units = alloc / prices.iloc[0]
    portfolio = (prices * units).sum(axis=1)
    return portfolio.rename("Benchmark_Value")


def directional_accuracy(signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overall directional prediction accuracy of combined signals.
    Correct = signal direction matches actual next-day price direction.
    """
    log_ret = np.log(prices / prices.shift(1))
    rows    = []
    for ticker in signals.columns:
        if ticker not in log_ret.columns:
            continue
        sig  = signals[ticker]
        rets = log_ret[ticker].reindex(sig.index).shift(-1)
        mask = sig != 0
        correct = (np.sign(sig[mask]) == np.sign(rets[mask])).sum()
        total   = mask.sum()
        rows.append({
            "Ticker":   ticker,
            "Signals":  int(total),
            "Correct":  int(correct),
            "Accuracy": round(correct / total, 4) if total > 0 else 0,
        })
    overall = pd.DataFrame(rows)
    overall.loc[len(overall)] = {
        "Ticker":   "OVERALL",
        "Signals":  overall["Signals"].sum(),
        "Correct":  overall["Correct"].sum(),
        "Accuracy": round(overall["Correct"].sum() / overall["Signals"].sum(), 4),
    }
    return overall.set_index("Ticker")
