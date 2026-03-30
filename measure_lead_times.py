"""
Supply Chain Disruption Predictor - Lead-Time Measurement
Computes actual vs expected lead times per order and per supplier,
substantiating the project's 'early warning' claim with real measurements.

Outputs:
  lead_time_analysis.csv  — per-order lead time detail
  lead_time_summary.json  — supplier & aggregate summary statistics
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd

from pipeline_utils import load_config, setup_logging

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
logger = setup_logging("lead_times")


# ===================================================================
# 1. Load completed orders
# ===================================================================

def load_completed_orders(cfg: dict) -> pd.DataFrame:
    """Load ml_features.csv and return completed deliveries only.

    Args:
        cfg: Loaded config dictionary.

    Returns:
        DataFrame of completed orders with date columns parsed.

    Raises:
        FileNotFoundError: If ml_features.csv does not exist.
    """
    ml_path = os.path.join(
        _PROJECT_ROOT,
        cfg.get("data", {}).get("output_files", {}).get("ml_features", "ml_features.csv"),
    )
    if not os.path.exists(ml_path):
        raise FileNotFoundError(
            f"ml_features.csv not found at {ml_path}. "
            "Run generate_disruption_data.py first."
        )

    df = pd.read_csv(ml_path)
    completed = df[df["delivery_status"].isin(["Delivered", "Partial"])].copy()

    # Parse dates
    for col in ("po_date", "expected_delivery_date", "actual_delivery_date"):
        if col in completed.columns:
            completed[col] = pd.to_datetime(completed[col], errors="coerce")

    logger.info("Loaded %d completed orders for lead-time analysis", len(completed))
    return completed


# ===================================================================
# 2. Compute per-order lead times
# ===================================================================

def compute_lead_times(df: pd.DataFrame) -> pd.DataFrame:
    """Add lead-time columns to the completed-orders DataFrame.

    New columns:
      - ``actual_lead_days``     : actual_delivery - po_date
      - ``expected_lead_days``   : expected_delivery - po_date
      - ``lead_time_variance``   : actual_lead - expected_lead (+ = late)
      - ``lateness_flag``        : 1 if lead_time_variance > 0 else 0
      - ``lead_time_week``       : ISO week of po_date (for trend analysis)

    Args:
        df: Completed orders with parsed date columns.

    Returns:
        DataFrame with new lead-time columns appended.
    """
    out = df.copy()

    if "actual_delivery_date" in out.columns and "po_date" in out.columns:
        out["actual_lead_days"] = (
            out["actual_delivery_date"] - out["po_date"]
        ).dt.days
    else:
        out["actual_lead_days"] = np.nan

    if "expected_delivery_date" in out.columns and "po_date" in out.columns:
        out["expected_lead_days"] = (
            out["expected_delivery_date"] - out["po_date"]
        ).dt.days
    else:
        out["expected_lead_days"] = np.nan

    out["lead_time_variance"] = out["actual_lead_days"] - out["expected_lead_days"]
    out["lateness_flag"] = (out["lead_time_variance"] > 0).astype(int)

    if "po_date" in out.columns:
        out["lead_time_week"] = out["po_date"].dt.isocalendar().week.astype(int)
        out["lead_time_year"] = out["po_date"].dt.year

    logger.info(
        "Lead-time variance: mean=%.1f days  std=%.1f days  on-time rate=%.1f%%",
        out["lead_time_variance"].mean(),
        out["lead_time_variance"].std(),
        (1 - out["lateness_flag"].mean()) * 100,
    )
    return out


# ===================================================================
# 3. Supplier-level summary
# ===================================================================

def supplier_lead_time_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate lead-time statistics per supplier.

    Metrics per supplier:
      - order_count
      - avg_actual_lead_days
      - avg_expected_lead_days
      - avg_variance_days      (+ = typically late)
      - std_variance_days      (consistency measure)
      - on_time_rate_pct
      - worst_delay_days       (max lateness)
      - early_warning_score    (0–100: lower is better; high = chronic lateness + variance)

    Args:
        df: DataFrame output from :func:`compute_lead_times`.

    Returns:
        Supplier summary DataFrame sorted by ``early_warning_score`` descending.
    """
    rows = []
    for supplier_id, group in df.groupby("supplier_id"):
        name = group["supplier_name"].iloc[0] if "supplier_name" in group.columns else supplier_id
        n = len(group)
        avg_actual = group["actual_lead_days"].mean()
        avg_expected = group["expected_lead_days"].mean()
        avg_var = group["lead_time_variance"].mean()
        std_var = group["lead_time_variance"].std()
        on_time_rate = (1 - group["lateness_flag"].mean()) * 100
        worst_delay = group["lead_time_variance"].max()

        # Early-warning score: penalises high average lateness + high variance
        # Normalised to 0-100: 50 * (avg_var / max_possible) + 50 * (std_var / max_var)
        max_possible_var = max(df["lead_time_variance"].abs().max(), 1)
        ew_score = min(100, max(0, round(
            50 * (max(avg_var, 0) / max_possible_var) +
            50 * (std_var / max(df["lead_time_variance"].std(), 1)),
            1,
        )))

        rows.append({
            "supplier_id": supplier_id,
            "supplier_name": name,
            "order_count": n,
            "avg_actual_lead_days": round(avg_actual, 1) if not np.isnan(avg_actual) else None,
            "avg_expected_lead_days": round(avg_expected, 1) if not np.isnan(avg_expected) else None,
            "avg_variance_days": round(avg_var, 1) if not np.isnan(avg_var) else None,
            "std_variance_days": round(std_var, 1) if not np.isnan(std_var) else None,
            "on_time_rate_pct": round(on_time_rate, 1),
            "worst_delay_days": round(worst_delay, 1) if not np.isnan(worst_delay) else None,
            "early_warning_score": ew_score,
        })

    summary = pd.DataFrame(rows).sort_values("early_warning_score", ascending=False)
    logger.info("Supplier lead-time summary: %d suppliers ranked", len(summary))
    return summary


# ===================================================================
# 4. Overall aggregate summary
# ===================================================================

def overall_summary(df: pd.DataFrame, supplier_summary: pd.DataFrame) -> Dict[str, Any]:
    """Build an overall aggregate summary for reporting.

    Args:
        df: Per-order DataFrame from :func:`compute_lead_times`.
        supplier_summary: Per-supplier DataFrame from :func:`supplier_lead_time_summary`.

    Returns:
        Dictionary suitable for JSON serialisation.
    """
    total_orders = len(df)
    on_time = int((df["lateness_flag"] == 0).sum())
    late = int((df["lateness_flag"] == 1).sum())

    late_orders = df[df["lead_time_variance"] > 0]
    avg_delay_when_late = (
        round(float(late_orders["lead_time_variance"].mean()), 1)
        if len(late_orders) > 0 else 0.0
    )

    # Worst and best months for on-time delivery
    if "lead_time_year" in df.columns and "lead_time_week" in df.columns:
        df["year_week"] = df["lead_time_year"].astype(str) + "-W" + df["lead_time_week"].astype(str).str.zfill(2)
        weekly = df.groupby("year_week")["lateness_flag"].mean().reset_index()
        weekly.columns = ["year_week", "late_rate"]
        worst_week = weekly.loc[weekly["late_rate"].idxmax(), "year_week"] if len(weekly) > 0 else "N/A"
        best_week = weekly.loc[weekly["late_rate"].idxmin(), "year_week"] if len(weekly) > 0 else "N/A"
    else:
        worst_week = best_week = "N/A"

    high_risk_suppliers = (
        supplier_summary[supplier_summary["early_warning_score"] >= 50]["supplier_name"].tolist()
        if "early_warning_score" in supplier_summary.columns else []
    )

    summary: Dict[str, Any] = {
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "_note": (
            "Lead times measured from PO date to actual delivery date. "
            "Variance = actual_lead - expected_lead. Positive = late."
        ),
        "total_completed_orders": total_orders,
        "on_time_orders": on_time,
        "late_orders": late,
        "on_time_rate_pct": round(on_time / total_orders * 100, 1) if total_orders > 0 else 0.0,
        "avg_lead_time_days": round(float(df["actual_lead_days"].mean()), 1),
        "avg_expected_lead_days": round(float(df["expected_lead_days"].mean()), 1),
        "avg_variance_days": round(float(df["lead_time_variance"].mean()), 1),
        "std_variance_days": round(float(df["lead_time_variance"].std()), 1),
        "avg_delay_when_late_days": avg_delay_when_late,
        "max_delay_days": round(float(df["lead_time_variance"].max()), 1),
        "worst_week_for_lateness": worst_week,
        "best_week_for_on_time": best_week,
        "high_early_warning_suppliers": high_risk_suppliers,
        "total_suppliers_analysed": len(supplier_summary),
    }

    logger.info(
        "Overall: %d orders | on-time=%.1f%% | avg_var=%.1f days | avg_delay_when_late=%.1f days",
        total_orders,
        summary["on_time_rate_pct"],
        summary["avg_variance_days"],
        avg_delay_when_late,
    )
    return summary


# ===================================================================
# 5. Pipeline entrypoint
# ===================================================================

def main() -> None:
    """Run the full lead-time analysis pipeline."""
    cfg = load_config()
    lt_cfg = cfg.get("lead_time", {})

    # --- Load and compute ---
    completed = load_completed_orders(cfg)
    detail_df = compute_lead_times(completed)
    supplier_df = supplier_lead_time_summary(detail_df)
    summary = overall_summary(detail_df, supplier_df)

    # --- Save per-order detail ---
    detail_cols = [
        "po_number", "supplier_id", "supplier_name",
        "po_date", "expected_delivery_date", "actual_delivery_date",
        "actual_lead_days", "expected_lead_days",
        "lead_time_variance", "lateness_flag",
    ]
    existing_cols = [c for c in detail_cols if c in detail_df.columns]
    out_path = os.path.join(
        _PROJECT_ROOT, lt_cfg.get("output_file", "lead_time_analysis.csv"),
    )
    detail_df[existing_cols].to_csv(out_path, index=False)
    logger.info("Saved lead-time detail: %s  (%d rows)", out_path, len(detail_df))

    # --- Save supplier summary inside the detail CSV with a blank separator ---
    # (supplier breakdown appended as a separate section for easy reading)
    supplier_path = out_path.replace(".csv", "_by_supplier.csv")
    supplier_df.to_csv(supplier_path, index=False)
    logger.info("Saved supplier summary: %s", supplier_path)

    # --- Save JSON summary ---
    # supplier rows need to be JSON-serialisable
    summary["supplier_breakdown"] = supplier_df.to_dict(orient="records")
    json_path = os.path.join(
        _PROJECT_ROOT, lt_cfg.get("summary_file", "lead_time_summary.json"),
    )
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Saved lead-time summary: %s", json_path)

    logger.info("=" * 60)
    logger.info("LEAD-TIME ANALYSIS COMPLETE")
    logger.info("  On-time rate: %.1f%%", summary["on_time_rate_pct"])
    logger.info("  Avg variance: %.1f days", summary["avg_variance_days"])
    logger.info("  Avg delay when late: %.1f days", summary["avg_delay_when_late_days"])
    logger.info("  High early-warning suppliers: %d", len(summary["high_early_warning_suppliers"]))
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Lead-time analysis failed: %s", exc)
        raise
