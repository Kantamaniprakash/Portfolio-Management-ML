"""
Data Collection Module
========================
Downloads historical OHLCV price data and generates synthetic financial
news headlines with sentiment labels for a diversified equity portfolio.

Portfolio tickers (8 large-cap US equities across 3 sectors):
  Technology : AAPL, MSFT, GOOGL, META
  E-Commerce : AMZN
  Finance    : JPM, GS
  EV / Auto  : TSLA
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

# ── Portfolio universe ─────────────────────────────────────────────────────
TICKERS = {
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",
    "AMZN":  "Amazon.com Inc.",
    "META":  "Meta Platforms Inc.",
    "TSLA":  "Tesla Inc.",
    "JPM":   "JPMorgan Chase & Co.",
    "GS":    "Goldman Sachs Group Inc.",
}

START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"

# ── Synthetic news templates (seeded deterministically) ───────────────────
_BULLISH_TEMPLATES = [
    "{ticker} reports record quarterly earnings, beating analyst estimates",
    "{ticker} announces major product launch driving strong revenue growth",
    "{ticker} shares surge after positive guidance upgrade from management",
    "{ticker} secures landmark partnership, expanding market reach significantly",
    "{ticker} benefits from macro tailwinds; analysts raise price targets",
    "{ticker} posts better-than-expected profit margins amid cost efficiencies",
    "{ticker} CEO unveils bold strategy, investors respond positively",
    "{ticker} outperforms sector peers with robust free cash flow generation",
]

_BEARISH_TEMPLATES = [
    "{ticker} misses earnings expectations; stock falls on weak guidance",
    "{ticker} faces regulatory scrutiny raising concerns over future growth",
    "{ticker} reports disappointing revenue amid slowing consumer demand",
    "{ticker} cuts full-year outlook; management cites macro headwinds",
    "{ticker} shares decline on supply chain disruption news",
    "{ticker} under pressure as competition intensifies in core markets",
    "{ticker} CFO departure sparks investor concern over financial strategy",
    "{ticker} warns of margin compression in upcoming quarters",
]

_NEUTRAL_TEMPLATES = [
    "{ticker} quarterly results in line with consensus expectations",
    "{ticker} maintains annual guidance; no changes to outlook",
    "{ticker} announces routine share buyback program continuation",
    "{ticker} completes previously announced acquisition on schedule",
]


def download_prices(
    tickers: dict = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Download daily adjusted closing prices for all portfolio tickers.

    Returns
    -------
    pd.DataFrame  shape (trading_days, n_tickers)
    """
    frames = {}
    for symbol in tickers:
        print(f"  Downloading {symbol} ...")
        raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            print(f"    WARNING: no data for {symbol}")
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frames[symbol] = raw["Close"].rename(symbol)

    df = pd.concat(frames.values(), axis=1)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.ffill().dropna()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"  Prices saved → {save_path}")

    print(f"  Price data: {df.shape}  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def generate_news(
    prices: pd.DataFrame,
    tickers: dict = TICKERS,
    save_path: str = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily financial news headlines with sentiment labels.

    The sentiment of each headline is correlated with the next-day price
    direction, simulating how real news often precedes price moves.

    Returns
    -------
    pd.DataFrame  columns: [Date, Ticker, Headline, Sentiment_Label, Sentiment_Score]
    """
    rng     = np.random.default_rng(random_seed)
    records = []

    log_returns = np.log(prices / prices.shift(1)).dropna()

    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        for i, date in enumerate(log_returns.index):
            ret = log_returns.loc[date, ticker]

            # Bias headline sentiment toward actual next-day price direction
            # (adds NLP predictive signal)
            if ret > 0.005:          # bullish day
                prob = [0.65, 0.20, 0.15]   # bullish / bearish / neutral
            elif ret < -0.005:       # bearish day
                prob = [0.15, 0.65, 0.20]
            else:                    # flat
                prob = [0.25, 0.25, 0.50]

            sentiment_type = rng.choice(["bullish", "bearish", "neutral"], p=prob)

            if sentiment_type == "bullish":
                template = rng.choice(_BULLISH_TEMPLATES)
                score    = float(rng.uniform(0.05, 0.95))
            elif sentiment_type == "bearish":
                template = rng.choice(_BEARISH_TEMPLATES)
                score    = float(rng.uniform(-0.95, -0.05))
            else:
                template = rng.choice(_NEUTRAL_TEMPLATES)
                score    = float(rng.uniform(-0.05, 0.05))

            records.append({
                "Date":             date,
                "Ticker":           ticker,
                "Headline":         template.format(ticker=ticker),
                "Sentiment_Label":  sentiment_type,
                "Sentiment_Score":  round(score, 4),
            })

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"  News data saved → {save_path}")

    print(f"  News records: {len(df)}  |  Tickers: {df['Ticker'].nunique()}")
    return df


if __name__ == "__main__":
    prices = download_prices(save_path="../data/portfolio_prices.csv")
    news   = generate_news(prices, save_path="../data/news_sentiment.csv")
    print(news.head())
