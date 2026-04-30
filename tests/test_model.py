"""Unit tests for the sentiment analysis pipeline."""

import sys
sys.path.insert(0, "src")

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from data_preprocessing import clean_text, preprocess_dataframe
import pandas as pd


class TestCleanText:
    def test_lowercase(self):
        assert clean_text("EARNINGS BEAT") == "earnings beat"

    def test_removes_url(self):
        assert "http" not in clean_text("See https://reuters.com for details")

    def test_collapses_whitespace(self):
        result = clean_text("profit  grew   strongly")
        assert "  " not in result

    def test_preserves_financial_tokens(self):
        text = "Q3 revenue up 2.5% to $1.2B"
        result = clean_text(text)
        assert "q3" in result
        assert "2.5" in result

    def test_empty_string(self):
        assert clean_text("") == ""


class TestPreprocessDataframe:
    def _make_df(self):
        return pd.DataFrame({
            "sentence": [
                "Profits surged in Q3.",
                "Company filed for bankruptcy.",
                "Revenue remained flat.",
            ],
            "sentiment": ["positive", "negative", "neutral"],
        })

    def test_label_encoding(self):
        df = preprocess_dataframe(self._make_df())
        assert set(df["label"].unique()).issubset({-1, 0, 1})

    def test_clean_text_column_created(self):
        df = preprocess_dataframe(self._make_df())
        assert "clean_text" in df.columns

    def test_no_nulls_after_processing(self):
        df = preprocess_dataframe(self._make_df())
        assert df["label"].isna().sum() == 0

    def test_positive_maps_to_1(self):
        df = preprocess_dataframe(self._make_df())
        assert df.loc[df["sentiment"] == "positive", "label"].iloc[0] == 1

    def test_negative_maps_to_minus_1(self):
        df = preprocess_dataframe(self._make_df())
        assert df.loc[df["sentiment"] == "negative", "label"].iloc[0] == -1


class TestBacktester:
    def test_simulate_signals_length(self):
        from backtester import simulate_signals
        s = simulate_signals(n_days=100)
        assert len(s) == 100

    def test_simulate_signals_values(self):
        from backtester import simulate_signals
        s = simulate_signals(n_days=1000)
        assert set(s.unique()).issubset({-1, 0, 1})

    def test_run_backtest_returns_equity(self):
        from backtester import simulate_signals, run_backtest
        import pandas as pd

        dates = pd.date_range("2022-01-01", periods=100, freq="B")
        returns = np.random.normal(0.0005, 0.01, 100)
        price_df = pd.DataFrame({
            "close": (1 + returns).cumprod() * 100,
            "return": returns,
        }, index=dates)

        signals = simulate_signals(n_days=100)
        bt = run_backtest(price_df, signals)
        assert "equity_strategy" in bt.columns
        assert "equity_bh" in bt.columns
        assert bt["equity_strategy"].iloc[0] > 0

    def test_metrics_keys(self):
        from backtester import simulate_signals, run_backtest, compute_metrics
        import pandas as pd

        dates = pd.date_range("2022-01-01", periods=50, freq="B")
        returns = np.random.normal(0.0005, 0.01, 50)
        price_df = pd.DataFrame({
            "close": (1 + returns).cumprod() * 100,
            "return": returns,
        }, index=dates)

        bt = run_backtest(price_df, simulate_signals(50))
        m = compute_metrics(bt)
        assert "sharpe_strategy" in m
        assert "max_drawdown_strategy" in m
        assert "total_return_strategy" in m
