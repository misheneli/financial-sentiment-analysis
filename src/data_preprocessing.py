"""
Data preprocessing module for Financial Sentiment Analysis.
Handles loading, cleaning, and feature engineering.
"""

import re
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(filepath: str) -> pd.DataFrame:
    """Load Financial PhraseBank CSV.

    The dataset comes in two formats:
    - With header: 'sentence,sentiment'
    - Without header: raw text@sentiment

    Handles both automatically.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Download Financial PhraseBank from: "
            "https://huggingface.co/datasets/financial_phrasebank"
        )

    # Try standard CSV first
    try:
        df = pd.read_csv(filepath, encoding="latin-1")
        if df.shape[1] == 2:
            df.columns = ["sentence", "sentiment"]
        elif df.shape[1] == 1:
            # Raw format: "text@sentiment"
            df = pd.read_csv(filepath, sep="@", header=None,
                              names=["sentence", "sentiment"], encoding="latin-1")
    except Exception:
        df = pd.read_csv(filepath, sep="@", header=None,
                          names=["sentence", "sentiment"], encoding="latin-1")

    df = df.dropna().reset_index(drop=True)
    logger.info(f"Loaded {len(df)} records from {filepath}")
    return df


def clean_text(text: str) -> str:
    """Clean and normalize financial text."""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Keep letters, numbers, spaces (financial figures like "Q3", "2.5%" are meaningful)
    text = re.sub(r"[^\w\s\.\,\%\$\-]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning and label encoding."""
    df = df.copy()
    df["clean_text"] = df["sentence"].apply(clean_text)

    # Standardize labels
    label_map = {"positive": 1, "negative": -1, "neutral": 0}
    df["label"] = df["sentiment"].str.lower().str.strip().map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    logger.info("Class distribution:\n" + df["sentiment"].value_counts().to_string())
    return df


def build_tfidf(config: dict):
    """Build TF-IDF vectorizer from config."""
    cfg = config["preprocessing"]
    vectorizer = TfidfVectorizer(
        max_features=cfg["max_features"],
        ngram_range=tuple(cfg["ngram_range"]),
        min_df=cfg["min_df"],
        sublinear_tf=True,        # log normalization — standard for text classification
        strip_accents="unicode",
    )
    return vectorizer


def get_train_test_split(df: pd.DataFrame, config: dict):
    """Split into train/test preserving class ratios (stratified)."""
    cfg = config["data"]
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y,               # critical for imbalanced classes
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def prepare_features(X_train, X_test, vectorizer):
    """Fit vectorizer on train, transform both splits."""
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec


def save_vectorizer(vectorizer, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)
    logger.info(f"Vectorizer saved to {path}")


def load_vectorizer(path: str):
    return joblib.load(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    df = load_data(config["data"]["raw_path"])
    df = preprocess_dataframe(df)
    df.to_csv(config["data"]["processed_path"], index=False)
    print(f"Saved processed data: {config['data']['processed_path']}")
    print(df["sentiment"].value_counts())
