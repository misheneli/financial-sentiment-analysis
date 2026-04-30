# БЫЛО (типичный вариант)
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)
print(accuracy_score(y_test, y_pred))

# СТАЛО
import logging
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    C: float = 1.0
    max_iter: int = 1000
    random_state: int = 42
    n_splits: int = 5
    model_path: str = "models/lr_model.pkl"
    metrics_path: str = "models/metrics.json"

def train(X_train, y_train, X_test, y_test, cfg: TrainConfig = TrainConfig()):
    model = LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, random_state=cfg.random_state)
    
    # Cross-val вместо одного сплита
    cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
    logger.info(f"CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "test_f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "classification_report": report,
    }
    
    logger.info(f"Test F1: {metrics['test_f1_weighted']:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    Path(cfg.model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.model_path)
    with open(cfg.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return model, metrics
