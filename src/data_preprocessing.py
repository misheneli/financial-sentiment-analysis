
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return [self._clean(text) for text in X]
    
    @staticmethod
    def _clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+", "", text)          # убрать ссылки
        text = re.sub(r"[^a-zA-Z\s]", " ", text)    # только буквы
        text = re.sub(r"\s+", " ", text).strip()
        return text

def build_pipeline(max_features: int = 10_000, ngram_range: tuple = (1, 2)) -> Pipeline:
    return Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,   # log-scaling — важно для NLP
        )),
    ])
