# Financial Sentiment Analysis

[![CI](https://github.com/misheneli/financial-sentiment-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/misheneli/financial-sentiment-analysis/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

NLP pipeline for 3-class sentiment classification of financial news (positive / neutral / negative), with a simple sentiment-driven backtest against SPY.

---

## Results

| Model | Macro-F1 | Accuracy | Notes |
|---|---|---|---|
| Logistic Regression + TF-IDF | **0.872** | **0.881** | Best CV score |
| LinearSVC + TF-IDF | 0.858 | 0.869 | Slightly lower F1 on Negative |
| Random Forest + TF-IDF | 0.821 | 0.836 | Overfits more |

Per-class F1 (Logistic Regression, test set):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative (12.5%) | 0.81 | 0.78 | 0.79 |
| Neutral (59.4%) | 0.91 | 0.93 | 0.92 |
| Positive (28.1%) | 0.87 | 0.85 | 0.86 |

> Class imbalance handled via `class_weight='balanced'` and stratified split.

### Backtest (Sentiment → SPY signal, 2020–2023)

| Metric | Sentiment Strategy | Buy & Hold |
|---|---|---|
| Total Return | ~18% | ~45% |
| Sharpe Ratio | ~0.62 | ~0.71 |
| Max Drawdown | -14% | -24% |
| Win Rate | ~51% | — |

> The strategy reduces drawdown significantly at the cost of upside capture. For production, align news timestamps with pre-market price data.

---

## Project Structure

```
financial-sentiment-analysis/
├── configs/
│   └── config.yaml          # All hyperparameters and paths
├── data/
│   ├── raw/                 # Financial PhraseBank (not tracked by Git)
│   └── processed/           # Cleaned CSV (not tracked by Git)
├── models/                  # Saved model + vectorizer (not tracked by Git)
├── notebooks/
│   └── data_analysis.ipynb  # EDA, model comparison, SHAP analysis
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train_model.py       # Trains LogReg / SVM / RF, CV, saves best
│   ├── evaluate.py          # Metrics, confusion matrix, ROC, SHAP
│   ├── backtester.py        # Sentiment → trading signal → equity curve
│   └── predict.py           # Inference + interactive CLI demo
├── tests/
│   └── test_model.py
├── analysis_results/        # Output plots (confusion matrix, ROC, equity curve)
├── .github/workflows/ci.yml
├── .gitignore
├── Makefile
├── requirements.txt
└── README.md
```

---

## Dataset

[Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank) — 4,846 English financial news sentences annotated by finance professionals.

| Sentiment | Count | % |
|---|---|---|
| Neutral | 2,879 | 59.4% |
| Positive | 1,363 | 28.1% |
| Negative | 604 | 12.5% |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/misheneli/financial-sentiment-analysis.git
cd financial-sentiment-analysis
pip install -r requirements.txt

# 2. Download dataset → place CSV at data/raw/financial_phrasebank.csv

# 3. Preprocess → Train → Backtest
make all

# Or step by step:
make preprocess
make train
make backtest

# 4. Interactive demo
make predict

# 5. Run tests
make test
```

---

## Methodology

**Text preprocessing:** lowercasing, URL removal, whitespace normalization. Financial tokens (Q3, 2.5%, $1.2B) are preserved intentionally — they carry predictive signal.

**Feature engineering:** TF-IDF with bigrams (`ngram_range=(1,2)`), `sublinear_tf=True` (log normalization), `max_features=10,000`. Bigrams capture phrases like "net loss", "revenue growth".

**Class imbalance:** Negative class is only 12.5% of data. Handled via `class_weight='balanced'` (weights inversely proportional to class frequency) and stratified train/test split.

**Model selection:** 5-fold stratified cross-validation, optimizing macro-F1 (treats all classes equally regardless of frequency).

**Backtesting:** Sentiment signal (positive→long, negative→short, neutral→cash) applied to SPY daily returns with 10 bps transaction cost per trade.

---

## Key Files to Read

- `src/train_model.py` — full ML pipeline with CV and model comparison  
- `src/evaluate.py` — metrics, SHAP feature importance  
- `src/backtester.py` — vectorized backtest with Sharpe/drawdown metrics  
- `notebooks/data_analysis.ipynb` — EDA and results visualization  


