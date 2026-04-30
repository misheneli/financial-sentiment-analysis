"""
Inference module.
Loads the trained model and vectorizer and predicts
sentiment for arbitrary financial text.
"""

import joblib
import logging
from pathlib import Path
from data_preprocessing import clean_text, load_config

logger = logging.getLogger(__name__)

LABEL_MAP = {-1: "🔴 NEGATIVE", 0: "⚪ NEUTRAL", 1: "🟢 POSITIVE"}


class SentimentPredictor:
    def __init__(self, model_path: str, vectorizer_path: str):
        meta = joblib.load(model_path)
        self.model = meta["model"]
        self.model_name = meta.get("model_name", "unknown")
        self.vectorizer = joblib.load(vectorizer_path)
        logger.info(f"Loaded model: {self.model_name}")

    def predict(self, texts: list[str]) -> list[dict]:
        """Predict sentiment for a list of texts.

        Returns list of dicts with keys: text, label, label_name, probabilities.
        """
        cleaned = [clean_text(t) for t in texts]
        X = self.vectorizer.transform(cleaned)
        labels = self.model.predict(X)

        results = []
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            classes = self.model.classes_
            for text, label, prob in zip(texts, labels, probs):
                prob_dict = {LABEL_MAP[c]: round(float(p), 4)
                             for c, p in zip(classes, prob)}
                results.append({
                    "text": text[:80] + "…" if len(text) > 80 else text,
                    "label": int(label),
                    "label_name": LABEL_MAP[label],
                    "probabilities": prob_dict,
                })
        else:
            for text, label in zip(texts, labels):
                results.append({
                    "text": text[:80] + "…" if len(text) > 80 else text,
                    "label": int(label),
                    "label_name": LABEL_MAP[label],
                    "probabilities": None,
                })
        return results

    def predict_one(self, text: str) -> dict:
        return self.predict([text])[0]


def interactive_demo(config_path: str = "configs/config.yaml"):
    """Command-line interactive demo."""
    logging.basicConfig(level=logging.WARNING)
    config = load_config(config_path)
    paths = config["paths"]

    if not Path(paths["model_file"]).exists():
        print("Model not found. Run: python src/train_model.py")
        return

    predictor = SentimentPredictor(paths["model_file"], paths["vectorizer_file"])

    # Sample predictions shown on startup
    samples = [
        "The company reported record quarterly earnings, beating analyst estimates.",
        "The firm announced massive layoffs amid declining revenues.",
        "Operating cash flow remained stable in Q3.",
    ]
    print("\n── Sample Predictions ──────────────────────────────────────")
    for result in predictor.predict(samples):
        print(f"\n  Text   : {result['text']}")
        print(f"  Label  : {result['label_name']}")
        if result["probabilities"]:
            probs = "  ".join(f"{k}: {v:.2%}" for k, v in result["probabilities"].items())
            print(f"  Probs  : {probs}")

    # Interactive loop
    print("\n── Enter your own text (Ctrl+C to exit) ────────────────────")
    while True:
        try:
            text = input("\n> ").strip()
            if not text:
                continue
            result = predictor.predict_one(text)
            print(f"  → {result['label_name']}")
            if result["probabilities"]:
                for k, v in result["probabilities"].items():
                    bar = "█" * int(v * 30)
                    print(f"     {k:<20} {v:5.1%}  {bar}")
        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    interactive_demo()
