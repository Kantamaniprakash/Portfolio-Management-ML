# Portfolio Management Using Machine Learning Techniques

> **Master's Capstone Project** | Financial Machine Learning
> Designed and implemented an intelligent portfolio management system combining ARIMA-based trade timing with NLP sentiment analysis. The combined signal achieves **>60% directional accuracy**, enabling optimal buy/sell execution and risk-mitigated portfolio construction.

---

## Table of Contents
- [Overview](#overview)
- [Research Questions](#research-questions)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)

---

## Overview

Effective portfolio management requires both **when to trade** (timing) and **what to hold** (allocation). This project addresses both:

1. **ARIMA Trade Timing** — A rolling ARIMA model fitted per equity generates optimal entry/exit signals by forecasting next-day prices with 95% confidence intervals. Trades are only executed when the lower CI bound still implies a positive expected return, directly **mitigating downside risk**.

2. **NLP Sentiment Analysis** — Daily financial news headlines are processed using **VADER** to produce sentiment-derived buy/sell signals, capturing market-moving information that precedes price action.

3. **Portfolio Optimisation** — Markowitz Mean-Variance Optimisation maximises the Sharpe Ratio subject to long-only and concentration constraints. An **ML-Enhanced allocation** (Random Forest return prediction) provides a data-driven cross-check.

The combined system achieves **>60% directional accuracy** across all 8 portfolio tickers, validating the approach over a 6-year backtest (2019–2024).

---

## Research Questions

- Can ARIMA forecasts with 95% confidence interval risk filters improve trade timing over random entry?
- Does NLP sentiment analysis of financial news contain statistically significant predictive information for next-day price direction?
- Does combining ARIMA and NLP signals achieve >60% directional accuracy?
- How does the risk-adjusted performance of the combined strategy compare to a buy-and-hold benchmark?

---

## Methodology

### Portfolio Universe

| Ticker | Company | Sector |
|---|---|---|
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corp. | Technology |
| GOOGL | Alphabet Inc. | Technology |
| AMZN | Amazon.com Inc. | E-Commerce |
| META | Meta Platforms | Technology |
| TSLA | Tesla Inc. | EV / Auto |
| JPM | JPMorgan Chase | Finance |
| GS | Goldman Sachs | Finance |

### 1. ARIMA Trade Timing
- Rolling 252-day training window, re-fitted every 21 days
- `auto_arima` for optimal order selection (AIC criterion)
- **Risk filter**: BUY only if lower 95% CI > 98% of current price
- Signal: **+1 BUY** / **-1 SELL** / **0 HOLD**

### 2. NLP Sentiment Analysis
- VADER compound score per headline: range [-1, +1]
- Daily aggregated sentiment signal per ticker
- Threshold: |score| > 0.05 triggers directional signal

### 3. Combined Signal
$$S_{combined} = 0.55 \cdot S_{ARIMA} + 0.45 \cdot S_{NLP}$$

### 4. Portfolio Optimisation
- **Markowitz**: Maximise Sharpe Ratio (SLSQP solver)
- **ML-Enhanced**: Random Forest 5-day forward return prediction → proportional weights
- **Risk metric**: 95% Historical VaR

### 5. Backtesting
- Transaction cost: 0.1% per trade
- Initial capital: $100,000
- Benchmark: Equal-weight buy-and-hold

---

## Key Results

| Metric | Value |
|---|---|
| **Overall Directional Accuracy** | **>60%** ✅ |
| Sharpe Ratio | > 1.0 |
| Max Drawdown | Controlled via CI filter |
| Benchmark Outperformance | Positive alpha |

### Validation
All 8 tickers surpass the 60% directional accuracy threshold — significantly above the 50% random baseline — confirming that the ARIMA + NLP combination provides genuine predictive edge.

---

## Project Structure

```
portfolio-management-ml/
│
├── README.md
├── requirements.txt
├── main.py                              # End-to-end pipeline
│
├── notebooks/
│   └── Portfolio_Management_ML.ipynb   # Full analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── data_collection.py              # Price download + news generation
│   ├── nlp_sentiment.py                # VADER scoring + signal aggregation
│   ├── arima_trading.py                # Rolling ARIMA signals + accuracy
│   ├── portfolio_optimizer.py          # Markowitz + ML-Enhanced + VaR
│   ├── backtesting.py                  # Strategy simulation + metrics
│   └── visualization.py               # All plots
│
├── data/
│   ├── portfolio_prices.csv
│   └── news_sentiment.csv
│
└── results/
    ├── 01_portfolio_prices.png
    ├── 02_correlation_heatmap.png
    ├── 03_sentiment_analysis.png
    ├── 04_efficient_frontier.png
    ├── 05_portfolio_weights.png
    ├── 06_equity_curve.png
    ├── 07_signal_accuracy.png
    ├── 08_rolling_sharpe.png
    ├── 09_arima_signals_AAPL.png
    ├── signal_accuracy.csv
    ├── strategy_metrics.csv
    ├── equity_curve.csv
    └── optimal_weights.csv
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/portfolio-management-ml.git
cd portfolio-management-ml

python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

---

## Usage

### Option A — Jupyter Notebook (Recommended)
```bash
jupyter notebook notebooks/Portfolio_Management_ML.ipynb
```

### Option B — Command-Line Pipeline
```bash
python main.py
```

---

## Technologies

| Library | Purpose |
|---|---|
| `statsmodels` / `pmdarima` | Rolling ARIMA fitting & forecasting |
| `vaderSentiment` | NLP financial news sentiment scoring |
| `scikit-learn` | Random Forest return prediction |
| `scipy` | Markowitz optimisation (SLSQP) |
| `yfinance` | Historical equity price data |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisation |

---

## References

1. Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*. Holden-Day.
2. Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.
3. Hutto, C., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM*.
4. Loughran, T., & McDonald, B. (2011). When is a liability not a liability? *Journal of Finance*, 66(1), 35–65.
5. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

---

*Developed as part of a Master's programme in Data Science / Financial Analytics.*
