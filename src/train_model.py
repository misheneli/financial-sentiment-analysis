"""
Model training module.
Trains multiple classifiers, performs cross-validation,
selects the best model, and saves it.
"""

import logging
from pathlib import Path

import numpy as np
import joblib
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from sklearn.calibration import CalibratedClassifierCV

from data_preprocessing import (
    load_config, load_data, preprocess_dataframe,
    build_tfidf, get_train_test_split, prepare_features,
    save_vectorizer,
)
from evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curves

logger = logging.getLogger(__name__)


def build_models(config: dict) -> dict:
    """Return dict of model_name -> sklearn estimator."""
    cfg = config["model"]
    lr_cfg = cfg["logistic_regression"]
    svm_cfg = cfg["svm"]
    rf_cfg = cfg["random_forest"]

    return {
        "logistic_regression": LogisticRegression(
            C=lr_cfg["C"],
            max_iter=lr_cfg["max_iter"],
            solver=lr_cfg["solver"],
            multi_class=lr_cfg["multi_class"],
            class_weight=cfg["class_weight"],
            random_state=cfg["random_state"],
        ),
        "svm": CalibratedClassifierCV(
            LinearSVC(
                C=svm_cfg["C"],
                class_weight=cfg["class_weight"],
                random_state=cfg["random_state"],
                max_iter=2000,
            )
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            class_weight=cfg["class_weight"],
            random_state=cfg["random_state"],
            n_jobs=-1,
        ),
    }


def cross_validate_models(models: dict, X_train, y_train, config: dict) -> dict:
    """Run stratified k-fold CV for all models. Returns mean macro-F1 per model."""
    cfg = config["model"]
    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True,
                         random_state=cfg["random_state"])
    cv_results = {}

    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        cv_results[name] = scores
        logger.info(
            f"{name}: CV macro-F1 = {scores.mean():.4f} ± {scores.std():.4f}"
        )

    return cv_results


def select_best_model(models: dict, cv_results: dict) -> tuple:
    """Return (name, model) with highest mean CV macro-F1."""
    best_name = max(cv_results, key=lambda k: cv_results[k].mean())
    logger.info(f"Best model: {best_name}")
    return best_name, models[best_name]


def train_and_save(config_path: str = "configs/config.yaml"):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    config = load_config(config_path)
    paths = config["paths"]
    Path(paths["models_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["results_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    df = load_data(config["data"]["raw_path"])
    df = preprocess_dataframe(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df, config)

    vectorizer = build_tfidf(config)
    X_train_vec, X_test_vec = prepare_features(X_train, X_test, vectorizer)
    save_vectorizer(vectorizer, paths["vectorizer_file"])

    # ── Cross-validation ──────────────────────────────────────────────────────
    models = build_models(config)
    logger.info("Running cross-validation …")
    cv_results = cross_validate_models(models, X_train_vec, y_train, config)

    # ── Final training ────────────────────────────────────────────────────────
    best_name, best_model = select_best_model(models, cv_results)
    best_model.fit(X_train_vec, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    label_names = ["negative", "neutral", "positive"]
    metrics = evaluate_model(best_model, X_test_vec, y_test,
                             label_names=label_names,
                             results_dir=paths["results_dir"])

    # Confusion matrix & ROC curves
    plot_confusion_matrix(best_model, X_test_vec, y_test,
                          label_names=label_names,
                          results_dir=paths["results_dir"])
    plot_roc_curves(best_model, X_test_vec, y_test,
                    label_names=label_names,
                    results_dir=paths["results_dir"])

    # ── Save ──────────────────────────────────────────────────────────────────
    model_meta = {
        "model": best_model,
        "model_name": best_name,
        "cv_results": {k: v.tolist() for k, v in cv_results.items()},
        "test_metrics": metrics,
        "label_map": {-1: "negative", 0: "neutral", 1: "positive"},
        "config": config,
    }
    joblib.dump(model_meta, paths["model_file"])
    logger.info(f"Model saved to {paths['model_file']}")

    print("\n" + "=" * 60)
    print(f"Best model : {best_name}")
    print(f"Test macro-F1 : {metrics['macro_f1']:.4f}")
    print(f"Test accuracy : {metrics['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save()
