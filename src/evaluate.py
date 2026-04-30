import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sentiment_to_signal(sentiment: str) -> int:
    return {"positive": 1, "neutral": 0, "negative": -1}[sentiment]

def backtest(df: pd.DataFrame, price_col: str = "close", sentiment_col: str = "sentiment") -> dict:
    """
    df должен содержать: date, close, predicted_sentiment
    """
    df = df.sort_values("date").copy()
    df["signal"] = df[sentiment_col].map(sentiment_to_signal)
    df["return"] = df[price_col].pct_change()
    df["strategy_return"] = df["signal"].shift(1) * df["return"]  # lag 1 день
    
    cumulative = (1 + df["strategy_return"].dropna()).cumprod()
    buy_hold = (1 + df["return"].dropna()).cumprod()
    
    sharpe = (
        df["strategy_return"].mean() / df["strategy_return"].std() * np.sqrt(252)
    )
    
    fig, ax = plt.subplots(figsize=(10, 4))
    cumulative.plot(ax=ax, label="Sentiment Strategy")
    buy_hold.plot(ax=ax, label="Buy & Hold")
    ax.set_title("Sentiment Signal Backtest")
    ax.legend()
    fig.savefig("models/backtest.png", dpi=150)
    
    return {
        "total_return": float(cumulative.iloc[-1] - 1),
        "sharpe_ratio": float(sharpe),
        "buy_hold_return": float(buy_hold.iloc[-1] - 1),
    }
