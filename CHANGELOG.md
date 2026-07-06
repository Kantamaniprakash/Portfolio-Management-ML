# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-05

### Added
- Rolling ARIMA trade-timing engine with `auto_arima` order selection and a 95% confidence-interval risk filter that only issues BUY signals when the lower CI bound still implies a positive expected return.
- VADER-based NLP sentiment analysis over daily financial news headlines, producing per-ticker directional buy/sell signals aggregated to a daily score.
- Combined ARIMA + NLP signal (0.55/0.45 weighting) reported at >60% directional accuracy across all 8 portfolio tickers on the synthetic seeded news corpus.
- Markowitz mean-variance portfolio optimisation (Sharpe-maximising SLSQP solver) plus an ML-Enhanced Random Forest allocation and 95% Historical VaR risk metric.
- Event-driven backtesting harness (0.1% transaction costs, $100k initial capital) with equity-curve, rolling-Sharpe, and strategy-metrics outputs benchmarked against equal-weight buy-and-hold.
- End-to-end `main.py` pipeline and a full Jupyter analysis notebook, with a modular `src/` package covering data collection, sentiment, ARIMA, optimisation, backtesting, and visualisation.
- CI across Python 3.10–3.12, smoke tests for source parsing and backtesting math, MIT license, and a synthetic-data disclosure documenting that the news corpus is generated, not real.

[0.1.0]: https://github.com/Kantamaniprakash/Portfolio-Management-ML/releases/tag/v0.1.0
