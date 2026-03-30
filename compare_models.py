"""
Supply Chain Disruption Predictor - Model Comparison
Benchmarks RandomForest against XGBoost using the same walk-forward split,
SMOTE oversampling, and optimised decision threshold.

Outputs:
  model_comparison.json  — side-by-side metric table
  best_model.pkl         — joblib-serialised best model
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pipeline_utils import load_config, setup_logging
from train_ml_model import (
    apply_smote,
    optimize_decision_threshold,
    prepare_data_for_ml,
    temporal_split,
)

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
logger = setup_logging("model_comparison")


# ===================================================================
# Single-model evaluation helper
# ===================================================================

def _evaluate_model(
    model: Any,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    cfg: dict,
) -> Dict[str, Any]:
    """Fit *model*, evaluate on the test fold, and return a metrics dict.

    Args:
        model: Unfitted sklearn-compatible estimator.
        model_name: Human-readable name for the model.
        X_train: SMOTE-resampled training features (numpy array).
        y_train: SMOTE-resampled training labels (numpy array).
        X_test: Held-out test features (DataFrame).
        y_test: Held-out test labels (Series).
        X_full: Full feature matrix for CV scoring.
        y_full: Full target for CV scoring.
        cfg: Loaded config.

    Returns:
        Metrics dictionary.
    """
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    thr_cfg = cfg.get("threshold_optimization", {})
    if thr_cfg.get("enabled", True):
        strategy = thr_cfg.get("strategy", "f1")
        opt_thr = optimize_decision_threshold(np.array(y_test), y_proba, strategy=strategy)
    else:
        strategy = "fixed_0.5"
        opt_thr = 0.5

    y_pred = (y_proba >= opt_thr).astype(int)

    cv_folds = cfg.get("model", {}).get("cv_folds", 5)
    seed = cfg.get("random_seed", 42)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring="roc_auc")

    cm = confusion_matrix(y_test, y_pred)

    result: Dict[str, Any] = {
        "model_name": model_name,
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_late": round(float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "recall_late": round(float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_late": round(float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        "threshold_used": round(opt_thr, 4),
        "threshold_strategy": strategy,
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        },
    }

    logger.info(
        "[%s] ROC-AUC=%.4f  CV=%.4f±%.4f  F1=%.4f  Recall=%.4f  Thr=%.4f",
        model_name,
        result["roc_auc"],
        result["cv_roc_auc_mean"],
        result["cv_roc_auc_std"],
        result["f1_late"],
        result["recall_late"],
        opt_thr,
    )
    return result


# ===================================================================
# Main comparison runner
# ===================================================================

def compare_models(cfg: Optional[dict] = None) -> Dict[str, Any]:
    """Train and compare all configured models; save best to disk.

    Args:
        cfg: Loaded config dictionary. Loads from file if None.

    Returns:
        Comparison report dictionary (also written to ``model_comparison.json``).
    """
    if cfg is None:
        cfg = load_config()

    outputs = cfg.get("outputs", {})
    cmp_cfg = cfg.get("model_comparison", {})
    models_to_run: List[str] = cmp_cfg.get("models", ["random_forest", "xgboost"])

    # --- Load data ---
    ml_path = os.path.join(
        _PROJECT_ROOT,
        cfg.get("data", {}).get("output_files", {}).get("ml_features", "ml_features.csv"),
    )
    logger.info("Loading ML features from %s", ml_path)
    ml_data = pd.read_csv(ml_path)

    if "po_date" in ml_data.columns:
        ml_data["po_date"] = pd.to_datetime(ml_data["po_date"])

    X_full, y_full, feature_columns = prepare_data_for_ml(ml_data, cfg)

    # --- Walk-forward split ---
    completed_df = ml_data[ml_data["delivery_status"].isin(["Delivered", "Partial"])].copy()
    if "po_date" in completed_df.columns:
        completed_df["po_date"] = pd.to_datetime(completed_df["po_date"])

    X_tr_df, X_te_df, y_tr, y_te = temporal_split(completed_df, feature_columns, cfg)

    # --- SMOTE on training fold ---
    X_tr_fit, y_tr_fit = apply_smote(X_tr_df, y_tr, cfg)

    # ----------------------------------------------------------------
    # Build candidate models
    # ----------------------------------------------------------------
    seed = cfg.get("random_seed", 42)
    model_cfg = cfg.get("model", {})
    xgb_cfg = cmp_cfg.get("xgboost_params", {})

    candidates: List[Tuple[str, Any]] = []

    if "random_forest" in models_to_run:
        rf = RandomForestClassifier(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", 10),
            min_samples_split=model_cfg.get("min_samples_split", 20),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 10),
            random_state=seed,
            class_weight=model_cfg.get("class_weight", "balanced"),
        )
        candidates.append(("RandomForest", rf))

    if "xgboost" in models_to_run:
        if not _XGB_AVAILABLE:
            logger.warning("xgboost not installed — skipping. Run: pip install xgboost")
        else:
            scale_pos = float((y_tr_fit == 0).sum()) / max(float((y_tr_fit == 1).sum()), 1)
            xgb = XGBClassifier(
                n_estimators=xgb_cfg.get("n_estimators", 200),
                max_depth=xgb_cfg.get("max_depth", 6),
                learning_rate=xgb_cfg.get("learning_rate", 0.05),
                subsample=xgb_cfg.get("subsample", 0.8),
                colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
                scale_pos_weight=scale_pos,
                random_state=seed,
                eval_metric="logloss",
                verbosity=0,
            )
            candidates.append(("XGBoost", xgb))

    if not candidates:
        logger.error("No models available to compare. Install xgboost or check config.")
        return {}

    # ----------------------------------------------------------------
    # Evaluate each candidate
    # ----------------------------------------------------------------
    results: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}

    for name, model_obj in candidates:
        logger.info("--- Evaluating %s ---", name)
        result = _evaluate_model(
            model_obj, name,
            X_tr_fit, y_tr_fit,
            X_te_df, y_te,
            X_full, y_full,
            cfg,
        )
        results.append(result)
        trained_models[name] = model_obj

    # ----------------------------------------------------------------
    # Select best model by CV ROC-AUC
    # ----------------------------------------------------------------
    best_result = max(results, key=lambda r: r["cv_roc_auc_mean"])
    best_name = best_result["model_name"]
    best_model = trained_models[best_name]

    logger.info("Best model: %s  (CV ROC-AUC=%.4f)", best_name, best_result["cv_roc_auc_mean"])

    # ----------------------------------------------------------------
    # Save best model
    # ----------------------------------------------------------------
    best_model_path = os.path.join(
        _PROJECT_ROOT, cmp_cfg.get("best_model_file", "best_model.pkl"),
    )
    joblib.dump(best_model, best_model_path)
    logger.info("Best model saved: %s", best_model_path)

    # ----------------------------------------------------------------
    # Build comparison report
    # ----------------------------------------------------------------
    report: Dict[str, Any] = {
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "_disclaimer": (
            "All metrics are evaluated on a walk-forward temporal test fold "
            "(most-recent 20%% of orders by PO date). CV scores use full dataset."
        ),
        "split_method": "walk_forward_temporal",
        "smote_applied": True,
        "best_model": best_name,
        "best_cv_roc_auc": best_result["cv_roc_auc_mean"],
        "results": results,
    }

    cmp_path = os.path.join(
        _PROJECT_ROOT, cmp_cfg.get("output_file", "model_comparison.json"),
    )
    with open(cmp_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Comparison report saved: %s", cmp_path)

    # ----------------------------------------------------------------
    # Summary table to stdout
    # ----------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("%-20s  %8s  %8s  %8s  %8s  %8s", "Model", "ROC-AUC", "CV-AUC", "Recall", "F1", "Thr")
    for r in sorted(results, key=lambda x: x["cv_roc_auc_mean"], reverse=True):
        logger.info(
            "%-20s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f",
            r["model_name"], r["roc_auc"], r["cv_roc_auc_mean"],
            r["recall_late"], r["f1_late"], r["threshold_used"],
        )
    logger.info("Winner: %s", best_name)
    logger.info("=" * 70)

    return report


# ===================================================================
# Entrypoint
# ===================================================================

if __name__ == "__main__":
    try:
        compare_models()
    except Exception as exc:
        logger.exception("Model comparison failed: %s", exc)
        raise
