"""
Smoke tests
===========
Fast, dependency-light checks that keep CI green:

1. Every source module under ``src/`` parses cleanly (``ast.parse``) — this
   catches syntax errors without importing heavy optional dependencies
   (statsmodels, pmdarima, sklearn, matplotlib, ...).
2. The pure-numpy/pandas backtesting math functions produce correct,
   deterministic results on tiny hand-built inputs.

No network, API keys, or GPU are required.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

# Make ``src`` importable as a package root for the modules we can import.
sys.path.insert(0, str(REPO_ROOT))


# ── 1. Every source file parses ────────────────────────────────────────────
@pytest.mark.parametrize(
    "py_file",
    sorted(SRC_DIR.glob("*.py")),
    ids=lambda p: p.name,
)
def test_source_files_parse(py_file):
    """Each module in src/ is syntactically valid Python."""
    source = py_file.read_text(encoding="utf-8")
    ast.parse(source, filename=str(py_file))


def test_key_files_exist():
    """Project scaffolding is present."""
    for rel in ("main.py", "requirements.txt", "README.md", "src/__init__.py"):
        assert (REPO_ROOT / rel).is_file(), f"missing {rel}"


def test_removed_heavy_deps_not_declared():
    """nltk / torch / transformers were intentionally removed (security + weight)."""
    reqs = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for pkg in ("nltk", "torch", "transformers"):
        assert pkg not in reqs, f"{pkg} should not be in requirements.txt"


# ── 2. Backtesting math (pure numpy/pandas — safe to import) ────────────────
from src import backtesting as bt  # noqa: E402


def test_combined_signal_thresholding():
    """Blended ARIMA+NLP signal discretises to {-1, 0, +1} correctly."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    arima = pd.DataFrame({"AAPL": [1, -1, 0], "MSFT": [1, 0, 0]}, index=idx)
    nlp = pd.DataFrame({"AAPL": [1, -1, 0], "MSFT": [-1, 0, 0]}, index=idx)

    out = bt.combined_signal(arima, nlp, arima_weight=0.55, signal_threshold=0.25)

    # Both agree strongly bullish -> +1
    assert out.loc[idx[0], "AAPL"] == 1
    # Both agree strongly bearish -> -1
    assert out.loc[idx[1], "AAPL"] == -1
    # Everyone flat -> 0
    assert out.loc[idx[2], "AAPL"] == 0
    # Values are always in the valid set
    assert set(np.unique(out.values)).issubset({-1, 0, 1})


def test_compute_metrics_on_flat_curve():
    """A perfectly flat equity curve has zero return, drawdown and sharpe."""
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    equity = pd.Series(100_000.0, index=idx)
    returns = pd.Series(0.0, index=idx)

    m = bt.compute_metrics(equity, returns, initial_capital=100_000.0)

    assert m["Total_Return_%"] == 0.0
    assert m["Max_Drawdown_%"] == 0.0
    assert m["Sharpe_Ratio"] == 0  # std == 0 guard path


def test_benchmark_buy_hold_grows_with_prices():
    """Equal-weight buy-and-hold curve starts at capital and tracks prices."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    prices = pd.DataFrame(
        {"AAPL": [100, 110, 121, 133.1], "MSFT": [50, 55, 60.5, 66.55]},
        index=idx,
    )
    curve = bt.benchmark_buy_hold(prices, initial_capital=100_000.0)

    assert curve.iloc[0] == pytest.approx(100_000.0)
    # Both assets rose ~10%/day, so the portfolio must be strictly increasing.
    assert (curve.diff().dropna() > 0).all()


def test_directional_accuracy_perfect_signals():
    """Signals that exactly match next-day direction score 100% accuracy."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    # Price rises, rises, falls -> next-day directions: +, +, -
    prices = pd.DataFrame({"AAPL": [100, 101, 102, 101]}, index=idx)
    # Signal on day t predicts direction of t+1.
    signals = pd.DataFrame({"AAPL": [1, 1, -1, 0]}, index=idx)

    acc = bt.directional_accuracy(signals, prices)

    assert acc.loc["AAPL", "Accuracy"] == 1.0
    assert acc.loc["OVERALL", "Accuracy"] == 1.0
