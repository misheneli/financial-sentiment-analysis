"""
Evaluation module.
Computes classification metrics, plots confusion matrix, ROC curves,
and SHAP feature importance for the trained model.
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test,
                   label_names: list,
                   results_dir: str = "analysis_results/") -> dict:
    """Compute and print full classification metrics."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    report = classification_report(y_test, y_pred, target_names=label_names)
    logger.info(f"\nClassification Report:\n{report}")
    print(f"\nClassification Report:\n{report}")

    # Save report to file
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{results_dir}/classification_report.txt", "w") as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report)
        f.write(f"\nAccuracy    : {acc:.4f}")
        f.write(f"\nMacro-F1    : {macro_f1:.4f}")
        f.write(f"\nWeighted-F1 : {weighted_f1:.4f}")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
    }


def plot_confusion_matrix(model, X_test, y_test,
                          label_names: list,
                          results_dir: str = "analysis_results/"):
    """Plot and save normalized confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax
    )
    ax.set_title("Normalized Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()

    out = Path(results_dir) / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {out}")


def plot_roc_curves(model, X_test, y_test,
                    label_names: list,
                    results_dir: str = "analysis_results/"):
    """Plot one-vs-rest ROC curves for each class."""
    classes = sorted(set(y_test))
    y_bin = label_binarize(y_test, classes=classes)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        logger.warning("Model has no predict_proba, skipping ROC curves.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#e74c3c", "#95a5a6", "#2ecc71"]

    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        auc = roc_auc_score(y_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label_names[i]} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()

    out = Path(results_dir) / "roc_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"ROC curves saved to {out}")


def plot_shap_importance(model, X_test, vectorizer,
                         n_top: int = 20,
                         results_dir: str = "analysis_results/"):
    """
    Plot top SHAP feature importances for the sentiment model.
    Works with LogisticRegression. Falls back to coef_ for linear models.
    """
    try:
        import shap
        feature_names = vectorizer.get_feature_names_out()

        # SHAP LinearExplainer is fastest for LR / LinearSVC
        if hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X_test,
                                             feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_test)
        else:
            # Tree / generic
            explainer = shap.Explainer(model, X_test)
            shap_values = explainer(X_test).values

        # Mean absolute SHAP value across all classes
        mean_shap = np.abs(np.array(shap_values)).mean(axis=(0, 2)) \
            if isinstance(shap_values, list) else np.abs(shap_values).mean(axis=0)

        top_idx = np.argsort(mean_shap)[-n_top:][::-1]
        top_features = [feature_names[i] for i in top_idx]
        top_vals = mean_shap[top_idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(n_top), top_vals[::-1], color="#3498db", alpha=0.85)
        ax.set_yticks(range(n_top))
        ax.set_yticklabels(top_features[::-1], fontsize=9)
        ax.set_title(f"Top {n_top} SHAP Feature Importances", fontsize=13,
                     fontweight="bold")
        ax.set_xlabel("Mean |SHAP value|")
        plt.tight_layout()

        out = Path(results_dir) / "shap_importance.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info(f"SHAP importance saved to {out}")

    except ImportError:
        logger.warning("shap not installed. Run: pip install shap")
    except Exception as e:
        logger.warning(f"SHAP failed: {e}. Skipping.")


def plot_cv_comparison(cv_results: dict, results_dir: str = "analysis_results/"):
    """Box plot comparing CV macro-F1 across all models."""
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(cv_results.keys())
    data = [cv_results[k] for k in names]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    bp = ax.boxplot(data, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels([n.replace("_", "\n") for n in names])
    ax.set_ylabel("Macro-F1")
    ax.set_title("Cross-Validation Macro-F1 by Model", fontsize=13,
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = Path(results_dir) / "cv_model_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"CV comparison saved to {out}")
