"""
ARIMA Trading Signal Module
=============================
Fits an ARIMA model per ticker on a rolling basis to generate
forward-looking price forecasts. The forecast vs. current price
determines the optimal entry/exit signal to maximise return and
mitigate downside risk.

Key concepts:
  - Rolling window ARIMA (re-fitted every `refit_freq` days) to stay adaptive
  - 95% confidence interval used as a risk filter — only trade when the
    lower CI bound still implies a positive expected return
  - Direction accuracy tracked against realised next-day returns
"""

import warnings
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def fit_arima_single(series: pd.Series) -> tuple:
    """
    Fit ARIMA(p,d,q) to a log-price series using auto_arima order selection.

    Returns (fitted_statsmodels_model, (p,d,q))
    """
    log_s = np.log(series.dropna())
    auto  = pm.auto_arima(
        log_s,
        start_p=0, start_q=0, max_p=4, max_q=4,
        d=None, seasonal=False,
        information_criterion="aic",
        stepwise=True, suppress_warnings=True,
        error_action="ignore", trace=False,
    )
    order   = auto.order
    fitted  = ARIMA(log_s, order=order).fit()
    return fitted, order


def arima_forecast_signal(
    fitted_model,
    current_price: float,
    alpha: float = 0.05,
    min_return_threshold: float = 0.003,
) -> dict:
    """
    Generate a single-step trade signal from a fitted ARIMA model.

    Signal logic
    ------------
    - Forecast next log-price and back-transform to price level
    - Compute expected return = (forecast - current) / current
    - BUY  (+1) if expected_return > threshold AND lower_CI > current_price * 0.98
    - SELL (-1) if expected_return < -threshold
    - HOLD ( 0) otherwise

    Returns
    -------
    dict {signal, forecast_price, lower_ci, upper_ci, expected_return}
    """
    fc_obj    = fitted_model.get_forecast(steps=1)
    log_fc    = fc_obj.predicted_mean.iloc[0]
    log_ci    = fc_obj.conf_int(alpha=alpha).iloc[0]

    fc_price  = np.exp(log_fc)
    lower_ci  = np.exp(log_ci.iloc[0])
    upper_ci  = np.exp(log_ci.iloc[1])
    exp_ret   = (fc_price - current_price) / current_price

    if exp_ret > min_return_threshold and lower_ci > current_price * 0.98:
        signal = 1    # BUY
    elif exp_ret < -min_return_threshold:
        signal = -1   # SELL / SHORT
    else:
        signal = 0    # HOLD

    return {
        "signal":          signal,
        "forecast_price":  round(fc_price,  4),
        "lower_ci":        round(lower_ci,  4),
        "upper_ci":        round(upper_ci,  4),
        "expected_return": round(exp_ret,   6),
    }


def rolling_arima_signals(
    prices: pd.DataFrame,
    train_window: int = 252,   # 1 year of trading days
    refit_freq:   int = 21,    # refit monthly
) -> pd.DataFrame:
    """
    Generate daily ARIMA trading signals for every ticker using a
    rolling training window.

    Parameters
    ----------
    prices       : pd.DataFrame  daily closing prices
    train_window : int           number of days used for ARIMA training
    refit_freq   : int           how often (in days) to re-fit the model

    Returns
    -------
    pd.DataFrame  pivot: rows=Date, cols=Ticker, values=signal {-1,0,+1}
    """
    all_signals = {}

    for ticker in prices.columns:
        print(f"    Rolling ARIMA signals for {ticker} ...")
        series  = prices[ticker].dropna()
        signals = []

        fitted_model = None
        for i in range(train_window, len(series)):
            # Re-fit ARIMA periodically
            if fitted_model is None or (i - train_window) % refit_freq == 0:
                train_slice  = series.iloc[i - train_window: i]
                try:
                    fitted_model, order = fit_arima_single(train_slice)
                except Exception:
                    signals.append((series.index[i], 0))
                    continue

            current_price = series.iloc[i]
            try:
                result = arima_forecast_signal(fitted_model, current_price)
                signals.append((series.index[i], result["signal"]))
            except Exception:
                signals.append((series.index[i], 0))

        sig_df = pd.DataFrame(signals, columns=["Date", ticker]).set_index("Date")
        all_signals[ticker] = sig_df[ticker]

    pivot = pd.concat(all_signals.values(), axis=1)
    pivot.index = pd.to_datetime(pivot.index)
    return pivot


def arima_signal_accuracy(signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute directional accuracy of ARIMA signals vs. realised next-day returns.

    A prediction is correct if:
      signal == +1 and next_day_return > 0
      signal == -1 and next_day_return < 0

    Returns
    -------
    pd.DataFrame  per-ticker accuracy statistics
    """
    log_returns = np.log(prices / prices.shift(1))
    rows = []
    for ticker in signals.columns:
        if ticker not in log_returns.columns:
            continue
        sig  = signals[ticker].dropna()
        rets = log_returns[ticker].reindex(sig.index).shift(-1)   # next-day return
        mask = sig != 0   # only evaluate non-HOLD signals
        sig_active  = sig[mask]
        rets_active = rets[mask]
        actual_dir  = np.sign(rets_active)
        correct     = (sig_active == actual_dir).sum()
        total       = mask.sum()
        accuracy    = correct / total if total > 0 else 0
        rows.append({
            "Ticker":         ticker,
            "Total_Signals":  int(total),
            "Correct":        int(correct),
            "Accuracy":       round(accuracy, 4),
            "Buy_Signals":    int((sig == 1).sum()),
            "Sell_Signals":   int((sig == -1).sum()),
            "Hold_Signals":   int((sig == 0).sum()),
        })
    return pd.DataFrame(rows).set_index("Ticker")


if __name__ == "__main__":
    from data_collection import download_prices
    prices  = download_prices()
    signals = rolling_arima_signals(prices[["AAPL", "MSFT"]], train_window=252, refit_freq=21)
    acc     = arima_signal_accuracy(signals, prices)
    print(acc)
