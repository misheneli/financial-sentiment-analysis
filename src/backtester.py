"""
Sentiment-driven backtester.

Strategy:
  - positive sentiment → Long SPY next day
  - negative sentiment → Short SPY (or cash)
  - neutral            → Cash

Computes: Cumulative Return, Sharpe Ratio, Max Drawdown, Win Rate.
Plots: Equity curve vs Buy-and-Hold.

NOTE: This is a simplified illustrative backtest using *in-sample* signals
      mapped to SPY price history. For a real backtest, use actual news dates
      aligned with price data.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import yaml

logger = logging.getLogger(__name__)


def load_spy_data(ticker: str = "SPY",
                  start: str = "2020-01-01",
                  end: str = "2023-12-31") -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        df = df[["Close"]].rename(columns={"Close": "close"})
        df["return"] = df["close"].pct_change()
        return df.dropna()
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")


def simulate_signals(n_days: int, random_state: int = 42) -> pd.Series:
    """
    Simulate sentiment signal distribution matching Financial PhraseBank:
      Positive: 28%, Negative: 12%, Neutral: 60%
    In a real pipeline, replace this with model predictions on dated news.
    """
    rng = np.random.default_rng(random_state)
    signals = rng.choice(
        a=[1, -1, 0],
        size=n_days,
        p=[0.281, 0.125, 0.594],
    )
    return pd.Series(signals, name="signal")


def predict_signals_from_model(model_path: str,
                               vectorizer_path: str,
                               texts: list) -> np.ndarray:
    """Run trained model on a list of news texts → integer labels."""
    model_meta = joblib.load(model_path)
    model = model_meta["model"]
    import joblib as jl
    vectorizer = jl.load(vectorizer_path)
    X = vectorizer.transform(texts)
    return model.predict(X)


def run_backtest(price_df: pd.DataFrame,
                 signals: pd.Series,
                 transaction_cost: float = 0.001,
                 allow_short: bool = True) -> pd.DataFrame:
    """
    Vectorized backtest.

    Signal mapping:
        +1  → Long  (buy close, sell next close)
        -1  → Short (if allow_short) or Cash
         0  → Cash

    Returns DataFrame with daily strategy and BH returns + equity curves.
    """
    n = min(len(price_df), len(signals))
    df = price_df.iloc[:n].copy()
    df["signal"] = signals.values[:n]

    if not allow_short:
        df["signal"] = df["signal"].clip(lower=0)

    # Strategy return = signal * next day return - |Δsignal| * cost
    df["position_change"] = df["signal"].diff().abs().fillna(0)
    df["strategy_return"] = (df["signal"].shift(1) * df["return"]
                             - df["position_change"] * transaction_cost)

    # Equity curves (starting from 1.0)
    df["equity_strategy"] = (1 + df["strategy_return"].fillna(0)).cumprod()
    df["equity_bh"] = (1 + df["return"].fillna(0)).cumprod()

    return df


def compute_metrics(bt: pd.DataFrame, trading_days: int = 252) -> dict:
    """Compute key performance metrics."""
    strat = bt["strategy_return"].dropna()
    bh = bt["return"].dropna()

    def sharpe(rets):
        if rets.std() == 0:
            return 0.0
        return (rets.mean() / rets.std()) * np.sqrt(trading_days)

    def max_drawdown(equity):
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max
        return dd.min()

    total_strat = bt["equity_strategy"].iloc[-1] - 1
    total_bh = bt["equity_bh"].iloc[-1] - 1
    n_trades = (bt["signal"] != 0).sum()
    win_rate = ((strat > 0).sum() / len(strat)) if len(strat) > 0 else 0

    return {
        "total_return_strategy": total_strat,
        "total_return_bh": total_bh,
        "sharpe_strategy": sharpe(strat),
        "sharpe_bh": sharpe(bh),
        "max_drawdown_strategy": max_drawdown(bt["equity_strategy"]),
        "max_drawdown_bh": max_drawdown(bt["equity_bh"]),
        "n_trades": int(n_trades),
        "win_rate": win_rate,
    }


def plot_equity_curve(bt: pd.DataFrame,
                      metrics: dict,
                      results_dir: str = "analysis_results/"):
    """Plot strategy equity curve vs Buy-and-Hold."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8),
                             gridspec_kw={"height_ratios": [3, 1]})

    # --- Equity curve ---
    ax = axes[0]
    ax.plot(bt.index, bt["equity_strategy"], label="Sentiment Strategy",
            color="#2c3e50", lw=2)
    ax.plot(bt.index, bt["equity_bh"], label="Buy & Hold SPY",
            color="#e74c3c", lw=1.5, alpha=0.85, linestyle="--")
    ax.set_title(
        f"Sentiment-Driven Strategy vs Buy & Hold\n"
        f"Strategy: {metrics['total_return_strategy']*100:.1f}% "
        f"| Sharpe: {metrics['sharpe_strategy']:.2f} "
        f"| MaxDD: {metrics['max_drawdown_strategy']*100:.1f}%\n"
        f"Buy&Hold: {metrics['total_return_bh']*100:.1f}% "
        f"| Sharpe: {metrics['sharpe_bh']:.2f} "
        f"| MaxDD: {metrics['max_drawdown_bh']*100:.1f}%",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("Portfolio Value (normalized)")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Drawdown ---
    ax2 = axes[1]
    dd = (bt["equity_strategy"] - bt["equity_strategy"].cummax()) / bt["equity_strategy"].cummax()
    ax2.fill_between(bt.index, dd * 100, 0, alpha=0.4, color="#e74c3c")
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(results_dir) / "backtest_equity_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Equity curve saved to {out}")


def run(config_path: str = "configs/config.yaml"):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    bt_cfg = config["backtest"]
    paths = config["paths"]

    price_df = load_spy_data(
        ticker=bt_cfg["ticker"],
        start=bt_cfg["start_date"],
        end=bt_cfg["end_date"],
    )
    logger.info(f"Loaded {len(price_df)} trading days for {bt_cfg['ticker']}")

    signals = simulate_signals(n_days=len(price_df))

    bt = run_backtest(
        price_df, signals,
        transaction_cost=bt_cfg["transaction_cost"],
        allow_short=True,
    )
    metrics = compute_metrics(bt)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.4f}")
        else:
            print(f"  {k:<35} {v}")
    print("=" * 60)

    bt.to_csv(f"{paths['results_dir']}/backtest_results.csv")
    plot_equity_curve(bt, metrics, paths["results_dir"])

    return bt, metrics


if __name__ == "__main__":
    run()
