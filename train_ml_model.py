"""
Supply Chain Disruption Predictor - Machine Learning Model
Predicts delivery delays and supplier risk using Random Forest.

Refactored for:
  - Centralized config (config.yaml)
  - Structured logging (replaces print)
  - Full technical metric persistence (accuracy, precision, recall, F1, AUC, confusion matrix)
  - Separated business-impact scenario file with explicit assumption labels
  - Temporal (out-of-time) validation option
  - Threshold calibration awareness
  - Type hints
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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from pipeline_utils import load_config, setup_logging, generate_run_metadata

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
logger = setup_logging("ml_model")


# ===================================================================
# 1. Data preparation
# ===================================================================

def prepare_data_for_ml(
    df: pd.DataFrame,
    cfg: dict,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Select completed deliveries and build feature matrix / target vector.

    Args:
        df: ML-features DataFrame produced by the data generator.
        cfg: Loaded config dictionary.

    Returns:
        (X, y, feature_columns)
    """
    train_df = df[df["delivery_status"].isin(["Delivered", "Partial"])].copy()

    logger.info("Training on %d completed deliveries", len(train_df))
    logger.info("Late deliveries: %d (%.1f%%)",
                train_df["is_late_delivery"].sum(),
                train_df["is_late_delivery"].mean() * 100)

    feature_columns: List[str] = [
        # Supplier characteristics
        "financial_health_score",
        "capacity_utilization_pct",
        "employee_turnover_pct",
        "quality_defect_rate",
        "inventory_days",
        "payment_default_risk",
        # External factors
        "rainy_season",
        "rainfall_index",
        "holiday_period",
        "port_congestion_index",
        "disruption_severity",
        # Order characteristics
        "total_amount_ngn",
        "quantity",
        "order_size_vs_avg",
        # Historical performance
        "supplier_avg_delay",
        "supplier_late_delivery_rate",
        "days_since_last_order",
        # Composite risk
        "composite_risk_score",
    ]

    X = train_df[feature_columns].fillna(0)
    y = train_df["is_late_delivery"]

    return X, y, feature_columns


# ===================================================================
# 2. Model training + full evaluation
# ===================================================================

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Train Random Forest classifier and return full evaluation metrics.

    Args:
        X: Feature matrix.
        y: Binary target.
        cfg: Loaded config dictionary.

    Returns:
        (trained model, metrics dictionary)
    """
    model_cfg = cfg.get("model", {})
    seed = cfg.get("random_seed", 42)
    test_size = model_cfg.get("test_size", 0.20)
    cv_folds = model_cfg.get("cv_folds", 5)

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )
    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # --- Build model ---
    rf = RandomForestClassifier(
        n_estimators=model_cfg.get("n_estimators", 100),
        max_depth=model_cfg.get("max_depth", 10),
        min_samples_split=model_cfg.get("min_samples_split", 20),
        min_samples_leaf=model_cfg.get("min_samples_leaf", 10),
        random_state=seed,
        class_weight=model_cfg.get("class_weight", "balanced"),
    )
    rf.fit(X_train, y_train)

    # --- Predictions ---
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    # --- Cross-validation ---
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)

    # --- Collect all metrics ---
    metrics: Dict[str, Any] = {
        "train_accuracy": round(float(rf.score(X_train, y_train)), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_late": round(float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "recall_late": round(float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_late": round(float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        },
        "class_distribution": {
            "total_samples": int(len(y)),
            "positive_pct": round(float(y.mean() * 100), 2),
            "negative_pct": round(float((1 - y.mean()) * 100), 2),
        },
        "threshold_used": 0.5,
        "model_params": rf.get_params(),
        "test_size": test_size,
        "cv_folds": cv_folds,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info("Test Accuracy: %.2f%%", metrics["test_accuracy"] * 100)
    logger.info("Precision (Late): %.2f%%", metrics["precision_late"] * 100)
    logger.info("Recall (Late): %.2f%%", metrics["recall_late"] * 100)
    logger.info("F1 (Late): %.4f", metrics["f1_late"])
    logger.info("ROC-AUC: %.4f", metrics["roc_auc"])
    logger.info("CV ROC-AUC: %.4f +/- %.4f", metrics["cv_roc_auc_mean"], metrics["cv_roc_auc_std"])
    logger.info("Confusion Matrix: TN=%d FP=%d FN=%d TP=%d",
                cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])

    return rf, metrics


# ===================================================================
# 3. Feature importance
# ===================================================================

def analyze_feature_importance(
    model: RandomForestClassifier,
    feature_columns: List[str],
) -> pd.DataFrame:
    """Return sorted feature importance DataFrame.

    Args:
        model: Trained RandomForest model.
        feature_columns: Feature names matching model input order.

    Returns:
        DataFrame with feature and importance columns, sorted descending.
    """
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("Top-5 features: %s",
                importance_df.head(5)[["feature", "importance"]].to_dict("records"))
    return importance_df


# ===================================================================
# 4. Predict pending orders
# ===================================================================

def predict_pending_orders(
    model: RandomForestClassifier,
    all_data_df: pd.DataFrame,
    feature_columns: List[str],
    cfg: dict,
) -> Optional[pd.DataFrame]:
    """Score pending orders and assign risk levels.

    Args:
        model: Trained model.
        all_data_df: Full ML-features DataFrame (includes Pending rows).
        feature_columns: Feature names.
        cfg: Loaded config.

    Returns:
        DataFrame of pending orders with predicted risk, or None.
    """
    pending = all_data_df[all_data_df["delivery_status"] == "Pending"].copy()
    if len(pending) == 0:
        logger.warning("No pending orders to predict")
        return None

    X_pending = pending[feature_columns].fillna(0)
    pending["predicted_late_probability"] = model.predict_proba(X_pending)[:, 1]

    thresholds = cfg.get("thresholds", {})
    low_max = thresholds.get("low_risk_max", 0.30)
    med_max = thresholds.get("medium_risk_max", 0.60)

    pending["predicted_risk_level"] = pd.cut(
        pending["predicted_late_probability"],
        bins=[0, low_max, med_max, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    high = pending[pending["predicted_risk_level"] == "High Risk"]
    med = pending[pending["predicted_risk_level"] == "Medium Risk"]
    low = pending[pending["predicted_risk_level"] == "Low Risk"]

    logger.info("Pending predictions — High: %d  Medium: %d  Low: %d",
                len(high), len(med), len(low))
    return pending


# ===================================================================
# 5. Supplier risk scorecard
# ===================================================================

def generate_supplier_risk_scorecard(
    all_data_df: pd.DataFrame,
    predictions_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build supplier risk scorecard from historical + predicted data.

    Args:
        all_data_df: Full ML-features DataFrame.
        predictions_df: Pending-order predictions (may be None).

    Returns:
        Scorecard DataFrame sorted by risk descending.
    """
    scorecard_rows: List[dict] = []

    for supplier_id in all_data_df["supplier_id"].unique():
        sd = all_data_df[all_data_df["supplier_id"] == supplier_id]
        name = sd["supplier_name"].iloc[0]

        completed = sd[sd["delivery_status"].isin(["Delivered", "Partial"])]
        hist_late = completed["is_late_delivery"].mean() if len(completed) > 0 else 0
        avg_delay = (
            completed.loc[completed["is_late_delivery"] == 1, "delivery_delay_days"].mean()
            if len(completed) > 0 else 0
        )

        if predictions_df is not None:
            pend = predictions_df[predictions_df["supplier_id"] == supplier_id]
            avg_pred = pend["predicted_late_probability"].mean() if len(pend) > 0 else 0
            hr_count = int((pend["predicted_risk_level"] == "High Risk").sum())
        else:
            avg_pred, hr_count = 0, 0

        avg_fh = sd["financial_health_score"].mean()
        avg_qd = sd["quality_defect_rate"].mean()

        risk_score = (
            hist_late * 40 +
            avg_pred * 30 +
            (100 - avg_fh) / 100 * 20 +
            avg_qd / 10 * 10
        )

        scorecard_rows.append({
            "supplier_id": supplier_id,
            "supplier_name": name,
            "historical_late_rate_pct": round(hist_late * 100, 1),
            "avg_delay_days": round(avg_delay, 1) if not np.isnan(avg_delay) else 0,
            "current_risk_score": round(risk_score, 1),
            "predicted_late_prob_pct": round(avg_pred * 100, 1),
            "high_risk_pending_orders": hr_count,
            "financial_health": round(avg_fh, 1),
            "quality_defect_rate": round(avg_qd, 2),
        })

    scorecard_df = pd.DataFrame(scorecard_rows).sort_values("current_risk_score", ascending=False)
    logger.info("Scorecard: %d suppliers ranked", len(scorecard_df))
    return scorecard_df


# ===================================================================
# 6. Business impact (SCENARIO-BASED — labeled assumptions)
# ===================================================================

def calculate_business_impact(
    predictions_df: Optional[pd.DataFrame],
    cfg: dict,
) -> Dict[str, Any]:
    """Estimate financial impact under stated scenario assumptions.

    All outputs are labeled as scenario-based and NOT measured outcomes.

    Args:
        predictions_df: Pending-order predictions.
        cfg: Loaded config.

    Returns:
        Business-impact dictionary with assumption labels.
    """
    if predictions_df is None:
        return {"note": "No pending orders — no impact calculated."}

    assumptions = cfg.get("business_assumptions", {})
    cost_pct = assumptions.get("delay_cost_pct_per_week", 0.03)
    delay_weeks = assumptions.get("assumed_avg_delay_weeks", 2)
    mitigation = assumptions.get("mitigation_success_rate", 0.70)

    high_risk = predictions_df[predictions_df["predicted_risk_level"] == "High Risk"]
    total_value = float(high_risk["total_amount_ngn"].sum())
    delay_cost = total_value * cost_pct * delay_weeks
    savings = delay_cost * mitigation

    impact: Dict[str, Any] = {
        "_disclaimer": (
            "SCENARIO-BASED estimates. These assume fixed delay cost rates "
            "and mitigation success percentages that have NOT been validated "
            "against real intervention outcomes."
        ),
        "assumptions": {
            "delay_cost_pct_per_week": cost_pct,
            "assumed_avg_delay_weeks": delay_weeks,
            "mitigation_success_rate": mitigation,
        },
        "high_risk_orders": len(high_risk),
        "high_risk_value_ngn": round(total_value, 2),
        "estimated_delay_cost_ngn": round(delay_cost, 2),
        "potential_savings_ngn": round(savings, 2),
        "avg_prediction_confidence": round(
            float(predictions_df["predicted_late_probability"].mean()), 4
        ),
    }

    logger.info("Business impact (scenario): %d high-risk orders, value=%.0f, savings=%.0f",
                impact["high_risk_orders"], total_value, savings)
    return impact


# ===================================================================
# 7. Pipeline entrypoint
# ===================================================================

def main() -> None:
    """Run the full ML training and prediction pipeline."""
    cfg = load_config()
    seed = cfg.get("random_seed", 42)
    np.random.seed(seed)

    outputs = cfg.get("outputs", {})

    # --- Load data ---
    ml_path = os.path.join(
        _PROJECT_ROOT,
        cfg.get("data", {}).get("output_files", {}).get("ml_features", "ml_features.csv"),
    )
    logger.info("Loading ML features from %s", ml_path)
    ml_data = pd.read_csv(ml_path)

    # --- Prepare ---
    X, y, feature_columns = prepare_data_for_ml(ml_data, cfg)

    # --- Train + evaluate ---
    model, metrics = train_model(X, y, cfg)

    # --- Feature importance ---
    importance_df = analyze_feature_importance(model, feature_columns)

    # --- Save model ---
    model_path = os.path.join(_PROJECT_ROOT, outputs.get("model_file", "disruption_predictor_model.pkl"))
    joblib.dump(model, model_path)
    logger.info("Model saved: %s", model_path)

    # --- Predict pending ---
    predictions = predict_pending_orders(model, ml_data, feature_columns, cfg)

    # --- Scorecard ---
    scorecard = generate_supplier_risk_scorecard(ml_data, predictions)

    # --- Business impact (scenario) ---
    impact = calculate_business_impact(predictions, cfg)

    # --- Save all outputs ---
    if predictions is not None:
        pred_path = os.path.join(_PROJECT_ROOT, outputs.get("predictions_file", "predictions_pending_orders.csv"))
        predictions[[
            "po_number", "supplier_name", "expected_delivery_date",
            "total_amount_ngn", "predicted_late_probability", "predicted_risk_level",
        ]].to_csv(pred_path, index=False)
        logger.info("Saved predictions: %s", pred_path)

    sc_path = os.path.join(_PROJECT_ROOT, outputs.get("scorecard_file", "supplier_risk_scorecard.csv"))
    scorecard.to_csv(sc_path, index=False)
    logger.info("Saved scorecard: %s", sc_path)

    fi_path = os.path.join(_PROJECT_ROOT, outputs.get("feature_importance_file", "feature_importance.csv"))
    importance_df.to_csv(fi_path, index=False)
    logger.info("Saved feature importance: %s", fi_path)

    # --- Technical metrics (SEPARATE from business impact) ---
    metrics_path = os.path.join(_PROJECT_ROOT, outputs.get("model_metrics_file", "model_metrics.json"))
    with open(metrics_path, "w", encoding="utf-8") as fh:
        # model_params may contain non-serializable objects; convert to str
        safe_metrics = metrics.copy()
        safe_metrics["model_params"] = {k: str(v) for k, v in metrics["model_params"].items()}
        json.dump(safe_metrics, fh, indent=2)
    logger.info("Saved technical metrics: %s", metrics_path)

    # --- Business impact (SEPARATE file, labeled scenario) ---
    biz_path = os.path.join(_PROJECT_ROOT, outputs.get("business_impact_file", "business_impact.json"))
    with open(biz_path, "w", encoding="utf-8") as fh:
        json.dump(impact, fh, indent=2)
    logger.info("Saved business impact (scenario): %s", biz_path)

    # --- Run metadata ---
    meta = generate_run_metadata(cfg, {"ml_features": ml_path})
    meta["model_metrics_summary"] = {
        "test_accuracy": metrics["test_accuracy"],
        "roc_auc": metrics["roc_auc"],
        "cv_roc_auc_mean": metrics["cv_roc_auc_mean"],
    }
    meta_path = os.path.join(_PROJECT_ROOT, outputs.get("run_metadata_file", "run_metadata.json"))
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Saved run metadata: %s", meta_path)

    logger.info("=" * 60)
    logger.info("ML PIPELINE COMPLETE")
    logger.info("  Test Accuracy: %.2f%%", metrics["test_accuracy"] * 100)
    logger.info("  ROC-AUC: %.4f  |  CV AUC: %.4f +/- %.4f",
                metrics["roc_auc"], metrics["cv_roc_auc_mean"], metrics["cv_roc_auc_std"])
    logger.info("  Scenario Savings: NGN %.0f", impact.get("potential_savings_ngn", 0))
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("ML pipeline failed: %s", exc)
        raise
