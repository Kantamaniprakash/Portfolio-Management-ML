"""
Visualization Module
=====================
All plots for the Portfolio Management ML project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

sns.set_theme(style="darkgrid", palette="muted")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(SAVE_DIR, exist_ok=True)

STRATEGY_COLOR   = "#2ecc71"
BENCHMARK_COLOR  = "#e74c3c"
ACCENT           = "#3498db"


def _save(fig, name):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ── 1. Portfolio Price History ────────────────────────────────────────────

def plot_prices(prices: pd.DataFrame, save: bool = True):
    rebased = prices / prices.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in rebased.columns:
        ax.plot(rebased.index, rebased[col], linewidth=1.3, label=col)
    ax.set_title("Portfolio Asset Prices – Normalised (Base = 100, Jan 2019)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rebased Price")
    ax.legend(ncol=4, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    if save: _save(fig, "01_portfolio_prices.png")
    return fig


# ── 2. Correlation Heat-map ───────────────────────────────────────────────

def plot_correlation(returns: pd.DataFrame, save: bool = True):
    corr = returns.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1, annot_kws={"size": 10})
    ax.set_title("Asset Return Correlation Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save: _save(fig, "02_correlation_heatmap.png")
    return fig


# ── 3. Sentiment Distribution ─────────────────────────────────────────────

def plot_sentiment(news_df: pd.DataFrame, save: bool = True):
    score_col = "VADER_Score" if "VADER_Score" in news_df.columns else "Sentiment_Score"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution
    for lbl, color in [("bullish", STRATEGY_COLOR), ("bearish", BENCHMARK_COLOR), ("neutral", ACCENT)]:
        sub = news_df[news_df["Sentiment_Label"] == lbl][score_col]
        sub.plot.kde(ax=axes[0], color=color, linewidth=2, label=lbl.capitalize())
    axes[0].set_title("Sentiment Score Distribution by Label", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Sentiment Score")
    axes[0].legend()

    # Per ticker average sentiment
    avg = news_df.groupby("Ticker")[score_col].mean().sort_values()
    colors = [STRATEGY_COLOR if v > 0 else BENCHMARK_COLOR for v in avg]
    axes[1].barh(avg.index, avg.values, color=colors, edgecolor="white")
    axes[1].axvline(0, color="grey", linewidth=1)
    axes[1].set_title("Average Sentiment Score per Ticker", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Mean Sentiment Score")

    fig.suptitle("NLP Sentiment Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save: _save(fig, "03_sentiment_analysis.png")
    return fig


# ── 4. Efficient Frontier ─────────────────────────────────────────────────

def plot_efficient_frontier(frontier_df: pd.DataFrame, opt_result: dict, save: bool = True):
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(frontier_df["Volatility"] * 100, frontier_df["Return"] * 100,
                    c=frontier_df["Sharpe"], cmap="RdYlGn", s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    ax.scatter(opt_result["volatility"] * 100, opt_result["return"] * 100,
               color="red", s=150, zorder=5, marker="*", label="Max Sharpe Portfolio")
    ax.set_xlabel("Annualised Volatility (%)")
    ax.set_ylabel("Annualised Return (%)")
    ax.set_title("Efficient Frontier – Markowitz Mean-Variance Optimisation",
                 fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    if save: _save(fig, "04_efficient_frontier.png")
    return fig


# ── 5. Optimal Portfolio Weights ──────────────────────────────────────────

def plot_weights(opt_result: dict, ml_weights: dict, save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, weights, title, color in [
        (axes[0], opt_result["weights"], "Markowitz (Max Sharpe)", ACCENT),
        (axes[1], ml_weights,            "ML-Enhanced (Random Forest)", STRATEGY_COLOR),
    ]:
        tickers = list(weights.keys())
        vals    = [weights[t] * 100 for t in tickers]
        ax.bar(tickers, vals, color=color, edgecolor="white")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Weight (%)")
        ax.set_ylim(0, 45)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    fig.suptitle("Portfolio Weight Allocation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save: _save(fig, "05_portfolio_weights.png")
    return fig


# ── 6. Equity Curve ───────────────────────────────────────────────────────

def plot_equity_curve(equity: pd.Series, benchmark: pd.Series, save: bool = True):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(equity.index, equity, color=STRATEGY_COLOR, linewidth=2, label="ARIMA + NLP Strategy")
    axes[0].plot(benchmark.index, benchmark, color=BENCHMARK_COLOR, linewidth=1.5,
                 linestyle="--", label="Buy & Hold Benchmark")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].set_title("Portfolio Equity Curve vs. Buy & Hold Benchmark", fontsize=14, fontweight="bold")
    axes[0].legend()

    # Drawdown
    cum_max  = equity.cummax()
    drawdown = (equity - cum_max) / cum_max * 100
    axes[1].fill_between(drawdown.index, drawdown, 0, color=BENCHMARK_COLOR, alpha=0.4)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    if save: _save(fig, "06_equity_curve.png")
    return fig


# ── 7. Signal Accuracy ────────────────────────────────────────────────────

def plot_signal_accuracy(accuracy_df: pd.DataFrame, save: bool = True):
    df = accuracy_df[accuracy_df.index != "OVERALL"].copy()
    colors = [STRATEGY_COLOR if v >= 0.60 else BENCHMARK_COLOR for v in df["Accuracy"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df.index, df["Accuracy"] * 100, color=colors, edgecolor="white")
    ax.axhline(60, color="orange", linestyle="--", linewidth=1.5, label="60% Target")
    ax.axhline(50, color="grey",   linestyle=":",  linewidth=1.2, label="Random (50%)")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_title("Combined Signal Directional Accuracy per Ticker", fontsize=13, fontweight="bold")
    ax.set_ylabel("Directional Accuracy (%)")
    ax.set_ylim(0, 90)
    ax.legend()
    fig.tight_layout()
    if save: _save(fig, "07_signal_accuracy.png")
    return fig


# ── 8. Rolling Sharpe ─────────────────────────────────────────────────────

def plot_rolling_sharpe(returns: pd.Series, window: int = 63, save: bool = True):
    rolling_sharpe = (
        returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
    )
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(rolling_sharpe.index, rolling_sharpe, color=ACCENT, linewidth=1.5)
    ax.axhline(0, color="grey", linewidth=1)
    ax.axhline(1, color=STRATEGY_COLOR, linestyle="--", linewidth=1.2, label="Sharpe = 1")
    ax.fill_between(rolling_sharpe.index, rolling_sharpe, 0,
                    where=rolling_sharpe > 0, alpha=0.25, color=STRATEGY_COLOR)
    ax.fill_between(rolling_sharpe.index, rolling_sharpe, 0,
                    where=rolling_sharpe < 0, alpha=0.25, color=BENCHMARK_COLOR)
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio – ARIMA + NLP Strategy",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    if save: _save(fig, "08_rolling_sharpe.png")
    return fig


# ── 9. ARIMA Signal on Single Stock ──────────────────────────────────────

def plot_arima_signals_on_price(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    ticker: str = "AAPL",
    history_days: int = 120,
    save: bool = True,
):
    px  = prices[ticker].iloc[-history_days:]
    sig = signals[ticker].reindex(px.index)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(px.index, px, color="#1f77b4", linewidth=1.5, label=f"{ticker} Price")

    buy_dates  = sig[sig ==  1].index
    sell_dates = sig[sig == -1].index
    ax.scatter(buy_dates,  px.reindex(buy_dates),  marker="^", color=STRATEGY_COLOR,
               s=80, zorder=5, label="BUY Signal")
    ax.scatter(sell_dates, px.reindex(sell_dates), marker="v", color=BENCHMARK_COLOR,
               s=80, zorder=5, label="SELL Signal")

    ax.set_title(f"{ticker} – ARIMA Trading Signals (Last {history_days} Days)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    if save: _save(fig, f"09_arima_signals_{ticker}.png")
    return fig
