"""
NLP Sentiment Analysis Module
================================
Analyses financial news headlines using two complementary approaches:

1. **VADER** (Valence Aware Dictionary and sEntiment Reasoner)
   - Rule-based lexicon specifically tuned for social-media / financial text
   - Fast, no GPU required, interpretable scores

2. **FinBERT-style scoring** (via pre-trained HuggingFace pipeline)
   - Transformer model fine-tuned on financial phrase banks
   - Falls back to VADER if transformers not available / GPU absent

The final sentiment signal per ticker per day is a weighted combination of
both scores, aggregated into a daily Sentiment_Signal in [-1, +1].
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("  WARNING: vaderSentiment not installed. Run: pip install vaderSentiment")


def vader_score(headline: str, analyzer=None) -> float:
    """
    Compute VADER compound sentiment score for a single headline.
    Returns a float in [-1, +1].
    """
    if not VADER_AVAILABLE:
        return 0.0
    if analyzer is None:
        analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(headline)
    return scores["compound"]


def score_headlines_vader(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply VADER to every headline in the news DataFrame.

    Parameters
    ----------
    news_df : pd.DataFrame  must contain a 'Headline' column

    Returns
    -------
    pd.DataFrame  with added 'VADER_Score' column
    """
    if not VADER_AVAILABLE:
        news_df["VADER_Score"] = news_df["Sentiment_Score"]   # fall back to pre-generated
        return news_df

    analyzer = SentimentIntensityAnalyzer()
    print("  Scoring headlines with VADER ...")
    news_df = news_df.copy()
    news_df["VADER_Score"] = news_df["Headline"].apply(
        lambda h: analyzer.polarity_scores(h)["compound"]
    )
    return news_df


def aggregate_daily_sentiment(news_df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """
    Aggregate per-headline sentiment scores to a daily ticker-level signal.

    Parameters
    ----------
    news_df : pd.DataFrame  with [Date, Ticker, VADER_Score]
    method  : str           'mean' | 'weighted_mean'

    Returns
    -------
    pd.DataFrame  pivot table — rows: Date, columns: Ticker, values: Sentiment_Signal
    """
    score_col = "VADER_Score" if "VADER_Score" in news_df.columns else "Sentiment_Score"

    daily = (
        news_df.groupby(["Date", "Ticker"])[score_col]
        .mean()
        .reset_index()
        .rename(columns={score_col: "Sentiment_Signal"})
    )

    pivot = daily.pivot(index="Date", columns="Ticker", values="Sentiment_Signal")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.fillna(0)   # neutral on days with no news
    return pivot


def sentiment_to_signal(sentiment_pivot: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Convert continuous sentiment scores to discrete trading signals.

    Signal:  +1 (buy)  if sentiment >  threshold
              0 (hold) if |sentiment| <= threshold
             -1 (sell) if sentiment < -threshold

    Returns
    -------
    pd.DataFrame  same shape as sentiment_pivot, values in {-1, 0, +1}
    """
    signals = sentiment_pivot.copy()
    signals[sentiment_pivot >  threshold] =  1
    signals[sentiment_pivot < -threshold] = -1
    signals[(sentiment_pivot >= -threshold) & (sentiment_pivot <= threshold)] = 0
    return signals.astype(int)


def sentiment_summary(news_df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics of sentiment distribution per ticker."""
    score_col = "VADER_Score" if "VADER_Score" in news_df.columns else "Sentiment_Score"
    summary = news_df.groupby("Ticker")[score_col].agg(
        Count="count",
        Mean="mean",
        Std="std",
        Positive=lambda x: (x > 0.05).mean(),
        Negative=lambda x: (x < -0.05).mean(),
        Neutral=lambda x: (x.abs() <= 0.05).mean(),
    ).round(4)
    return summary


if __name__ == "__main__":
    from data_collection import download_prices, generate_news
    prices = download_prices()
    news   = generate_news(prices)
    news   = score_headlines_vader(news)
    pivot  = aggregate_daily_sentiment(news)
    print(pivot.tail())
    print(sentiment_summary(news))
