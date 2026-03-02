"""
Supply Chain Disruption Predictor - Enhanced Data Generator
Generates risk factors and external variables, enriches purchase orders,
and produces ML-ready features with validated join coverage.

Refactored for:
  - Centralized config (config.yaml)
  - Structured logging
  - Data-contract validation & join diagnostics
  - Dynamic date-range alignment (fixes external-feature null bug)
  - Type hints
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline_utils import (
    load_config,
    setup_logging,
    validate_dataframe_schema,
    validate_join_coverage,
    generate_run_metadata,
)

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
logger = setup_logging("data_generator")


def _seed(cfg: dict) -> None:
    """Set global random seeds from config."""
    seed = cfg.get("random_seed", 42)
    np.random.seed(seed)
    random.seed(seed)


# ===================================================================
# 1. External Risk Factors
# ===================================================================

def generate_external_risk_factors(
    date_range: pd.DatetimeIndex,
    cfg: dict,
) -> pd.DataFrame:
    """Generate daily external risk factors covering *date_range*.

    The date range is passed explicitly so it can be aligned with PO dates
    rather than depending on ``datetime.now()``.

    Args:
        date_range: DatetimeIndex of dates to generate factors for.
        cfg: Loaded config dictionary.

    Returns:
        DataFrame with one row per date and risk-factor columns.
    """
    ext = cfg.get("external_risk", {})
    rainy_months: List[int] = ext.get("rainy_months", [4, 5, 6, 7, 8, 9, 10])
    holiday_months: List[int] = ext.get("holiday_months", [12, 1])
    rain_rainy = ext.get("rainfall_range_rainy", [0.3, 0.9])
    rain_dry = ext.get("rainfall_range_dry", [0.1, 0.4])
    base_fuel = ext.get("base_fuel_price", 200)
    fuel_inc = ext.get("fuel_price_daily_increment", 0.15)
    fuel_vol = ext.get("fuel_price_volatility", [0.9, 1.2])
    base_fx = ext.get("base_fx_rate", 800)
    fx_dep = ext.get("fx_daily_depreciation", 1.1)
    fx_vol = ext.get("fx_volatility", [0.95, 1.15])
    base_cong = ext.get("base_congestion", 30)
    rainy_cong_add = ext.get("rainy_congestion_add", 20)
    holiday_cong_add = ext.get("holiday_congestion_add", 15)
    cong_mult = ext.get("congestion_multiplier", [0.8, 1.4])
    disruption_prob = ext.get("disruption_probability", 0.05)

    start_date = date_range.min()
    records: List[dict] = []

    for date in date_range:
        month = date.month
        day_offset = (date - start_date).days

        rainy_season = 1 if month in rainy_months else 0
        rainfall_index = (
            random.uniform(*rain_rainy) if rainy_season else random.uniform(*rain_dry)
        )
        holiday_period = 1 if month in holiday_months else 0

        fuel_price = (base_fuel + day_offset * fuel_inc) * random.uniform(*fuel_vol)
        fx_rate = (base_fx + day_offset * fx_dep) * random.uniform(*fx_vol)

        congestion = base_cong
        if rainy_season:
            congestion += rainy_cong_add
        if holiday_period:
            congestion += holiday_cong_add
        port_congestion = min(100, congestion * random.uniform(*cong_mult))

        has_disruption = random.random() < disruption_prob
        disruption_type = (
            random.choice([
                "Strike", "Weather Event", "Political Unrest",
                "Port Closure", "Customs Delay",
            ])
            if has_disruption else "None"
        )
        disruption_severity = random.randint(1, 10) if has_disruption else 0

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "rainy_season": rainy_season,
            "rainfall_index": round(rainfall_index, 2),
            "holiday_period": holiday_period,
            "fuel_price_ngn": round(fuel_price, 2),
            "usd_ngn_rate": round(fx_rate, 2),
            "port_congestion_index": round(port_congestion, 0),
            "disruption_event": disruption_type,
            "disruption_severity": disruption_severity,
        })

    df = pd.DataFrame(records)
    logger.info("Generated %d daily external risk records (%s -> %s)",
                len(df), date_range.min().date(), date_range.max().date())
    return df


# ===================================================================
# 2. Supplier Health Metrics
# ===================================================================

def generate_supplier_health_metrics(
    suppliers_df: pd.DataFrame,
    cfg: dict,
    date_range: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Generate monthly supplier health metrics.

    Args:
        suppliers_df: Supplier master data.
        cfg: Loaded config dictionary.
        date_range: Optional DatetimeIndex aligned to PO dates.
                    If provided, months are derived from this range instead of
                    ``datetime.now()``, ensuring deterministic output.

    Returns:
        DataFrame with monthly health records per supplier.
    """
    if date_range is not None:
        # Derive distinct months from the PO-aligned date range
        month_labels = sorted(date_range.strftime("%Y-%m").unique())
    else:
        months = cfg.get("data", {}).get("supplier_health_months", 24)
        lookback = cfg.get("data", {}).get("lookback_days", 730)
        start_date = datetime.now() - timedelta(days=lookback)
        month_labels = [
            (start_date + timedelta(days=m * 30)).strftime("%Y-%m")
            for m in range(months)
        ]

    records: List[dict] = []

    for _, supplier in suppliers_df.iterrows():
        for month_str in month_labels:

            base_health = (
                70 if supplier["risk_level"] == "Low"
                else 50 if supplier["risk_level"] == "Medium"
                else 35
            )
            financial_health = max(0, min(100, base_health + random.randint(-15, 15)))
            capacity_util = random.uniform(60, 95)
            turnover_rate = random.uniform(2, 15)
            defect_rate = random.uniform(0.5, 8.0)
            inventory_days = random.randint(15, 90)
            default_risk = (
                random.uniform(10, 30)
                if supplier["country"] == "Nigeria"
                else random.uniform(5, 25)
            )

            records.append({
                "supplier_id": supplier["supplier_id"],
                "supplier_name": supplier["supplier_name"],
                "month": month_str,
                "financial_health_score": round(financial_health, 1),
                "capacity_utilization_pct": round(capacity_util, 1),
                "employee_turnover_pct": round(turnover_rate, 1),
                "quality_defect_rate": round(defect_rate, 2),
                "inventory_days": inventory_days,
                "payment_default_risk": round(default_risk, 1),
            })

    df = pd.DataFrame(records)
    logger.info("Generated %d supplier-health records for %d suppliers over %d months",
                len(df), suppliers_df["supplier_id"].nunique(), len(month_labels))
    return df


# ===================================================================
# 3. Enrich Purchase Orders  (FIX: date alignment + diagnostics)
# ===================================================================

def enhance_purchase_orders_with_risk(
    pos_df: pd.DataFrame,
    risk_factors_df: pd.DataFrame,
    supplier_health_df: pd.DataFrame,
    cfg: dict,
) -> Tuple[pd.DataFrame, Dict]:
    """Merge external risk and supplier health onto purchase orders.

    Returns:
        Tuple of (enriched DataFrame, join diagnostics dict).
    """
    weights = cfg.get("risk_weights", {})
    supplier_w = weights.get("supplier_weight", 0.40)
    external_w = weights.get("external_weight", 0.30)
    operational_w = weights.get("operational_weight", 0.30)

    enhanced = pos_df.copy()

    # --- External risk join (on date string) ---
    enhanced["po_date"] = pd.to_datetime(enhanced["po_date"]).dt.strftime("%Y-%m-%d")
    risk_factors_df = risk_factors_df.copy()
    risk_factors_df["date"] = pd.to_datetime(risk_factors_df["date"]).dt.strftime("%Y-%m-%d")

    enhanced = enhanced.merge(
        risk_factors_df,
        left_on="po_date",
        right_on="date",
        how="left",
    )

    # --- Supplier health join (on supplier_id + month) ---
    enhanced["po_month"] = pd.to_datetime(enhanced["po_date"]).dt.strftime("%Y-%m")
    enhanced = enhanced.merge(
        supplier_health_df,
        left_on=["supplier_id", "po_month"],
        right_on=["supplier_id", "month"],
        how="left",
        suffixes=("", "_supplier"),
    )

    # --- Join diagnostics ---
    ext_cols = ["rainy_season", "rainfall_index", "holiday_period",
                "port_congestion_index", "disruption_severity"]
    health_cols = ["financial_health_score", "capacity_utilization_pct",
                   "employee_turnover_pct", "quality_defect_rate",
                   "inventory_days", "payment_default_risk"]

    contracts = cfg.get("data_contracts", {})
    min_ext = contracts.get("min_external_coverage_pct", 80)
    min_health = contracts.get("min_health_coverage_pct", 90)

    diagnostics: Dict = {"external": {}, "health": {}}
    total = len(enhanced)
    for col in ext_cols:
        nulls = int(enhanced[col].isna().sum())
        diagnostics["external"][col] = {
            "nulls": nulls,
            "coverage_pct": round((1 - nulls / total) * 100, 2),
        }
    for col in health_cols:
        nulls = int(enhanced[col].isna().sum())
        diagnostics["health"][col] = {
            "nulls": nulls,
            "coverage_pct": round((1 - nulls / total) * 100, 2),
        }

    ext_avg_cov = np.mean([v["coverage_pct"] for v in diagnostics["external"].values()])
    health_avg_cov = np.mean([v["coverage_pct"] for v in diagnostics["health"].values()])
    diagnostics["external_avg_coverage_pct"] = round(float(ext_avg_cov), 2)
    diagnostics["health_avg_coverage_pct"] = round(float(health_avg_cov), 2)

    if ext_avg_cov < min_ext:
        logger.warning("External join coverage %.1f%% < threshold %.1f%% -- check date alignment!",
                        ext_avg_cov, min_ext)
    else:
        logger.info("External join coverage: %.1f%% (threshold: %.1f%%)", ext_avg_cov, min_ext)

    if health_avg_cov < min_health:
        logger.warning("Health join coverage %.1f%% < threshold %.1f%%", health_avg_cov, min_health)
    else:
        logger.info("Health join coverage: %.1f%% (threshold: %.1f%%)", health_avg_cov, min_health)

    # --- Delivery delay & target ---
    enhanced["delivery_delay_days"] = (
        pd.to_datetime(enhanced["actual_delivery_date"]) -
        pd.to_datetime(enhanced["expected_delivery_date"])
    ).dt.days
    enhanced.loc[enhanced["delivery_status"] == "Pending", "delivery_delay_days"] = np.nan
    enhanced["is_late_delivery"] = (enhanced["delivery_delay_days"] > 0).astype(int)

    # --- Composite risk score ---
    supplier_risk = (100 - enhanced["financial_health_score"].fillna(50)) / 100 * supplier_w
    external_risk = (
        enhanced["port_congestion_index"].fillna(30) / 100 * (external_w / 2) +
        enhanced["disruption_severity"].fillna(0) / 10 * (external_w / 2)
    )
    operational_risk = (
        enhanced["quality_defect_rate"].fillna(3) / 10 * (operational_w / 2) +
        enhanced["employee_turnover_pct"].fillna(5) / 15 * (operational_w / 2)
    )
    enhanced["composite_risk_score"] = ((supplier_risk + external_risk + operational_risk) * 100).round(1)

    thresholds = cfg.get("thresholds", {})
    low_max = thresholds.get("low_risk_max", 0.30) * 100
    med_max = thresholds.get("medium_risk_max", 0.60) * 100
    enhanced["risk_category"] = pd.cut(
        enhanced["composite_risk_score"],
        bins=[0, low_max, med_max, 100],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    enhanced.drop(columns=["date", "month", "po_month"], errors="ignore", inplace=True)

    logger.info("Enhanced %d purchase orders with risk factors", len(enhanced))
    return enhanced, diagnostics


# ===================================================================
# 4. Feature Engineering
# ===================================================================

def create_prediction_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create ML-ready features with rolling supplier metrics.

    Args:
        df: Enhanced purchase orders.
        cfg: Loaded config dictionary.

    Returns:
        DataFrame with engineered features ready for modeling.
    """
    feat_cfg = cfg.get("features", {})
    delay_win = feat_cfg.get("delay_rolling_window", 5)
    late_win = feat_cfg.get("late_rate_rolling_window", 10)
    default_delay = feat_cfg.get("default_avg_delay", 0)
    default_late = feat_cfg.get("default_late_rate", 0.5)
    default_days = feat_cfg.get("default_days_since_order", 30)

    features_df = df.copy()
    features_df["po_date"] = pd.to_datetime(features_df["po_date"])
    features_df.sort_values(["supplier_id", "po_date"], inplace=True)

    features_df["supplier_avg_delay"] = features_df.groupby("supplier_id")[
        "delivery_delay_days"
    ].transform(lambda x: x.shift(1).rolling(window=delay_win, min_periods=1).mean())

    features_df["supplier_late_delivery_rate"] = features_df.groupby("supplier_id")[
        "is_late_delivery"
    ].transform(lambda x: x.shift(1).rolling(window=late_win, min_periods=1).mean())

    features_df["order_size_vs_avg"] = features_df.groupby("supplier_id")[
        "total_amount_ngn"
    ].transform(lambda x: x / x.mean())

    features_df["days_since_last_order"] = features_df.groupby("supplier_id")[
        "po_date"
    ].diff().dt.days

    features_df["supplier_avg_delay"] = features_df["supplier_avg_delay"].fillna(default_delay)
    features_df["supplier_late_delivery_rate"] = features_df["supplier_late_delivery_rate"].fillna(default_late)
    features_df["days_since_last_order"] = features_df["days_since_last_order"].fillna(default_days)

    logger.info("Created ML feature set: %d rows x %d columns", *features_df.shape)
    return features_df


# ===================================================================
# 5. Pipeline entrypoint
# ===================================================================

def main() -> None:
    """Run the full data-generation pipeline."""
    cfg = load_config()
    _seed(cfg)

    data_cfg = cfg.get("data", {})
    src = data_cfg.get("source_files", {})
    out = data_cfg.get("output_files", {})

    # --- Load source data ---
    suppliers_path = os.path.join(_PROJECT_ROOT, src.get("suppliers", "suppliers.csv"))
    pos_path = os.path.join(_PROJECT_ROOT, src.get("purchase_orders", "purchase_orders.csv"))

    # Fallback: try parent workspace for source files (common in multi-project workspaces)
    if not os.path.exists(suppliers_path):
        alt = os.path.join(_PROJECT_ROOT, "..", "files", "suppliers.csv")
        if os.path.exists(alt):
            suppliers_path = alt
            logger.info("Using fallback suppliers path: %s", alt)
        else:
            raise FileNotFoundError(
                f"suppliers.csv not found at {suppliers_path} or {alt}. "
                "Place source data in project root or ../files/."
            )
    if not os.path.exists(pos_path):
        alt = os.path.join(_PROJECT_ROOT, "..", "files", "purchase_orders.csv")
        if os.path.exists(alt):
            pos_path = alt
            logger.info("Using fallback purchase_orders path: %s", alt)
        else:
            raise FileNotFoundError(
                f"purchase_orders.csv not found at {pos_path} or {alt}. "
                "Place source data in project root or ../files/."
            )

    logger.info("Loading source data...")
    suppliers_df = pd.read_csv(suppliers_path)
    pos_df = pd.read_csv(pos_path)

    contracts = cfg.get("data_contracts", {})
    validate_dataframe_schema(
        suppliers_df,
        contracts.get("required_supplier_columns", []),
        label="suppliers.csv",
    )
    validate_dataframe_schema(
        pos_df,
        contracts.get("required_po_columns", []),
        label="purchase_orders.csv",
    )
    logger.info("Loaded %d suppliers, %d purchase orders", len(suppliers_df), len(pos_df))

    # --- Generate risk factors aligned to PO date range ---
    po_dates = pd.to_datetime(pos_df["po_date"])
    min_date = po_dates.min() - timedelta(days=7)   # small buffer
    max_date = po_dates.max() + timedelta(days=7)
    full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")
    logger.info("Risk-factor date range: %s -> %s (%d days)",
                min_date.date(), max_date.date(), len(full_date_range))

    risk_factors_df = generate_external_risk_factors(full_date_range, cfg)

    # --- Supplier health (aligned to PO date range) ---
    supplier_health_df = generate_supplier_health_metrics(
        suppliers_df, cfg, date_range=full_date_range,
    )

    # --- Enrich POs ---
    enhanced_df, join_diag = enhance_purchase_orders_with_risk(
        pos_df, risk_factors_df, supplier_health_df, cfg
    )

    # --- Feature engineering ---
    ml_ready_df = create_prediction_features(enhanced_df, cfg)

    # --- Save outputs ---
    def _save(df: pd.DataFrame, key: str) -> str:
        path = os.path.join(_PROJECT_ROOT, out.get(key, f"{key}.csv"))
        df.to_csv(path, index=False)
        logger.info("Saved %s (%d rows)", path, len(df))
        return path

    _save(risk_factors_df, "risk_factors")
    _save(supplier_health_df, "supplier_health")
    _save(enhanced_df, "enhanced_orders")
    _save(ml_ready_df, "ml_features")

    # Save join diagnostics
    outputs_cfg = cfg.get("outputs", {})
    diag_path = os.path.join(_PROJECT_ROOT, outputs_cfg.get("join_diagnostics_file", "join_diagnostics.json"))
    with open(diag_path, "w", encoding="utf-8") as fh:
        json.dump(join_diag, fh, indent=2)
    logger.info("Saved join diagnostics: %s", diag_path)

    # Save run metadata
    meta = generate_run_metadata(cfg, {
        "suppliers": suppliers_path,
        "purchase_orders": pos_path,
    })
    meta["join_diagnostics_summary"] = {
        "external_coverage_pct": join_diag.get("external_avg_coverage_pct"),
        "health_coverage_pct": join_diag.get("health_avg_coverage_pct"),
    }
    meta_path = os.path.join(_PROJECT_ROOT, outputs_cfg.get("run_metadata_file", "run_metadata.json"))
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Saved run metadata: %s", meta_path)

    # --- Summary ---
    late = enhanced_df[enhanced_df["is_late_delivery"] == 1]
    logger.info("=" * 60)
    logger.info("DATA GENERATION COMPLETE")
    logger.info("  Completed deliveries: %d", enhanced_df["is_late_delivery"].notna().sum())
    logger.info("  Late deliveries: %d", len(late))
    logger.info("  On-time rate: %.1f%%", (1 - enhanced_df["is_late_delivery"].mean()) * 100)
    logger.info("  External coverage: %.1f%%", join_diag.get("external_avg_coverage_pct", 0))
    logger.info("  Health coverage: %.1f%%", join_diag.get("health_avg_coverage_pct", 0))
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        raise
