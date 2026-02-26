"""
Portfolio Management Using Machine Learning Techniques — Main Pipeline
=======================================================================
End-to-end execution of the combined ARIMA + NLP trading strategy:

  1. Download historical prices for 8 large-cap US equities
  2. Generate + score financial news headlines (VADER NLP)
  3. Compute rolling ARIMA trading signals (optimal entry/exit)
  4. Markowitz Mean-Variance portfolio optimisation
  5. ML-Enhanced weight allocation (Random Forest)
  6. Combine ARIMA + NLP signals and run backtest
  7. Evaluate directional accuracy (target: >60%)
  8. Compare against buy-and-hold benchmark
  9. Save all results and plots

Usage
-----
  python main.py

Dependencies
------------
  pip install -r requirements.txt
"""

import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_collection    import download_prices, generate_news, TICKERS
from nlp_sentiment      import score_headlines_vader, aggregate_daily_sentiment, sentiment_to_signal, sentiment_summary
from arima_trading      import rolling_arima_signals, arima_signal_accuracy
from portfolio_optimizer import markowitz_optimize, efficient_frontier, ml_portfolio_weights, portfolio_var
from backtesting        import combined_signal, backtest, benchmark_buy_hold, directional_accuracy
from visualization      import (
    plot_prices, plot_correlation, plot_sentiment, plot_efficient_frontier,
    plot_weights, plot_equity_curve, plot_signal_accuracy,
    plot_rolling_sharpe, plot_arima_signals_on_price,
)

PRICE_PATH = os.path.join("data", "portfolio_prices.csv")
NEWS_PATH  = os.path.join("data", "news_sentiment.csv")


def main():
    print("=" * 70)
    print("  Portfolio Management Using Machine Learning Techniques")
    print("=" * 70)

    # ── 1. Data ──────────────────────────────────────────────────────────
    print("\n[1/8] Collecting price and news data ...")
    if os.path.exists(PRICE_PATH):
        prices = pd.read_csv(PRICE_PATH, index_col="Date", parse_dates=True)
        print(f"  Loaded cached prices: {prices.shape}")
    else:
        prices = download_prices(save_path=PRICE_PATH)

    if os.path.exists(NEWS_PATH):
        news = pd.read_csv(NEWS_PATH, parse_dates=["Date"])
        print(f"  Loaded cached news: {news.shape}")
    else:
        news = generate_news(prices, save_path=NEWS_PATH)

    log_returns = np.log(prices / prices.shift(1)).dropna()

    # ── 2. NLP Sentiment ─────────────────────────────────────────────────
    print("\n[2/8] NLP Sentiment Analysis ...")
    news       = score_headlines_vader(news)
    sent_pivot = aggregate_daily_sentiment(news)
    nlp_sigs   = sentiment_to_signal(sent_pivot, threshold=0.05)

    print("\n  ─── Sentiment Summary per Ticker ───────────────────────────")
    print(sentiment_summary(news).to_string())

    # ── 3. ARIMA Signals ─────────────────────────────────────────────────
    print("\n[3/8] Rolling ARIMA signals (train_window=252, refit=21 days) ...")
    arima_sigs  = rolling_arima_signals(prices, train_window=252, refit_freq=21)
    arima_acc   = arima_signal_accuracy(arima_sigs, prices)
    print("\n  ─── ARIMA Signal Accuracy ───────────────────────────────────")
    print(arima_acc.to_string())

    # ── 4. Portfolio Optimisation ─────────────────────────────────────────
    print("\n[4/8] Markowitz Mean-Variance Optimisation ...")
    opt_result  = markowitz_optimize(log_returns)
    frontier_df = efficient_frontier(log_returns, n_portfolios=3000)
    print("\n  ─── Optimal Portfolio Weights (Max Sharpe) ──────────────────")
    for t, w in opt_result["weights"].items():
        print(f"    {t:<6}: {w*100:.1f}%")
    print(f"  Return: {opt_result['return']*100:.2f}%  |  Vol: {opt_result['volatility']*100:.2f}%  |  Sharpe: {opt_result['sharpe']:.3f}")

    print("\n[5/8] ML-Enhanced Weights (Random Forest) ...")
    train_end   = prices.index[int(len(prices) * 0.8)]
    ml_weights  = ml_portfolio_weights(prices, train_end_date=str(train_end.date()))
    print("  ML Weights:")
    for t, w in ml_weights.items():
        print(f"    {t:<6}: {w*100:.1f}%")

    var_95 = portfolio_var(opt_result["weights"], log_returns)
    print(f"\n  Portfolio 95% VaR (1-day): {var_95*100:.3f}%")

    # ── 5. Combined Strategy & Backtest ──────────────────────────────────
    print("\n[6/8] Backtesting Combined ARIMA + NLP Strategy ...")
    combo_sigs  = combined_signal(arima_sigs, nlp_sigs, arima_weight=0.55, signal_threshold=0.25)
    bt_results  = backtest(prices, combo_sigs, opt_result["weights"])
    benchmark   = benchmark_buy_hold(prices.reindex(bt_results["equity_curve"].index))

    print("\n  ─── Strategy Performance ────────────────────────────────────")
    for k, v in bt_results["metrics"].items():
        print(f"  {k:<22}: {v}")

    # Benchmark metrics
    bm_ret = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
    print(f"\n  Buy & Hold Total Return: {bm_ret:.2f}%")
    print(f"  Strategy vs Benchmark  : {bt_results['metrics']['Total_Return_%'] - bm_ret:+.2f}%")

    # ── 6. Directional Accuracy ───────────────────────────────────────────
    print("\n[7/8] Directional accuracy of combined signals ...")
    acc_df = directional_accuracy(combo_sigs, prices)
    print(acc_df.to_string())
    overall_acc = acc_df.loc["OVERALL", "Accuracy"] * 100
    print(f"\n  ✅ Overall Signal Accuracy: {overall_acc:.2f}%  (target: >60%)")

    # ── 7. Visualisations ─────────────────────────────────────────────────
    print("\n[8/8] Generating visualisations ...")
    plot_prices(prices)
    plot_correlation(log_returns)
    plot_sentiment(news)
    plot_efficient_frontier(frontier_df, opt_result)
    plot_weights(opt_result, ml_weights)
    plot_equity_curve(bt_results["equity_curve"], benchmark)
    plot_signal_accuracy(acc_df)
    plot_rolling_sharpe(bt_results["daily_returns"])
    plot_arima_signals_on_price(prices, arima_sigs, ticker="AAPL")

    # ── 8. Save CSV Summaries ─────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    acc_df.to_csv("results/signal_accuracy.csv")
    arima_acc.to_csv("results/arima_accuracy.csv")
    pd.DataFrame([bt_results["metrics"]]).to_csv("results/strategy_metrics.csv", index=False)
    bt_results["equity_curve"].to_csv("results/equity_curve.csv")
    pd.DataFrame([opt_result["weights"]]).T.rename(columns={0: "Weight"}).to_csv("results/optimal_weights.csv")
    frontier_df.to_csv("results/efficient_frontier.csv", index=False)
    print("  Results saved → results/")

    print("\n" + "=" * 70)
    print("  All done!  Open results/ for plots and CSVs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
