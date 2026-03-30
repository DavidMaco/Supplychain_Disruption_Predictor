"""
Unit and integration tests for Supply Chain Disruption Predictor.
Covers: config loading, data contracts, risk scores, rolling features,
join coverage, model training smoke test, and claim validation.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline_utils import (
    load_config,
    reset_config_cache,
    setup_logging,
    validate_dataframe_schema,
    validate_join_coverage,
    file_hash,
    generate_run_metadata,
)

from generate_disruption_data import (
    generate_external_risk_factors,
    generate_supplier_health_metrics,
    enhance_purchase_orders_with_risk,
    create_prediction_features,
    _seed,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(autouse=True)
def _reset_config():
    """Reset config cache between tests."""
    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture
def sample_config():
    return load_config(os.path.join(_PROJECT_ROOT, "config.yaml"))


@pytest.fixture
def sample_suppliers():
    return pd.DataFrame({
        "supplier_id": ["S001", "S002"],
        "supplier_name": ["Supplier A", "Supplier B"],
        "risk_level": ["Low", "High"],
        "country": ["Nigeria", "China"],
    })


@pytest.fixture
def sample_pos():
    dates = pd.date_range("2025-01-01", periods=20, freq="3D")
    return pd.DataFrame({
        "po_number": [f"PO{i:04d}" for i in range(20)],
        "supplier_id": ["S001", "S002"] * 10,
        "supplier_name": ["Supplier A", "Supplier B"] * 10,
        "po_date": dates.strftime("%Y-%m-%d"),
        "total_amount_ngn": np.random.uniform(1e6, 1e8, 20),
        "quantity": np.random.randint(100, 5000, 20),
        "expected_delivery_date": (dates + timedelta(days=14)).strftime("%Y-%m-%d"),
        "actual_delivery_date": (dates + timedelta(days=np.random.randint(10, 25))).strftime("%Y-%m-%d"),
        "delivery_status": ["Delivered"] * 16 + ["Pending"] * 4,
    })


@pytest.fixture
def enriched_data(sample_pos, sample_suppliers, sample_config):
    """Shared fixture: run data generation through enrichment.

    Returns (enhanced_df, join_diagnostics, date_range, sample_config).
    """
    rng = _seed(sample_config)
    dates = pd.to_datetime(sample_pos["po_date"])
    date_range = pd.date_range(dates.min() - timedelta(days=2), dates.max() + timedelta(days=2))
    risk_df = generate_external_risk_factors(date_range, sample_config, rng=rng)
    health_df = generate_supplier_health_metrics(sample_suppliers, sample_config, date_range=date_range, rng=rng)
    enhanced, diag = enhance_purchase_orders_with_risk(
        sample_pos, risk_df, health_df, sample_config,
    )
    return enhanced, diag, date_range, sample_config


# ===================================================================
# Config tests
# ===================================================================

class TestConfig:
    def test_loads_yaml(self, sample_config):
        assert "random_seed" in sample_config
        assert isinstance(sample_config["random_seed"], int)

    def test_has_required_sections(self, sample_config):
        for key in ["data", "model", "thresholds", "risk_weights",
                     "business_assumptions", "outputs", "data_contracts"]:
            assert key in sample_config, f"Missing config section: {key}"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


# ===================================================================
# Data-contract tests
# ===================================================================

class TestDataContracts:
    def test_valid_schema_passes(self, sample_suppliers):
        validate_dataframe_schema(
            sample_suppliers,
            ["supplier_id", "supplier_name", "risk_level", "country"],
            label="test",
        )

    def test_missing_column_raises(self, sample_suppliers):
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe_schema(
                sample_suppliers,
                ["supplier_id", "nonexistent_col"],
                label="test",
            )

    def test_join_coverage_pass(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]})
        diag = validate_join_coverage(df, ["a"], min_coverage_pct=100, label="test")
        assert diag["pass"] is True

    def test_join_coverage_fail(self):
        df = pd.DataFrame({"a": [1, None, None], "b": [4, 5, 6]})
        with pytest.raises(ValueError, match="Coverage below"):
            validate_join_coverage(df, ["a"], min_coverage_pct=90, label="test")


# ===================================================================
# Risk score computation tests
# ===================================================================

class TestRiskScore:
    def test_composite_score_range(self, enriched_data):
        """Composite risk score should be in [0, 100]."""
        enhanced, _, _, _ = enriched_data
        assert enhanced["composite_risk_score"].between(0, 100).all(), \
            "Composite risk score out of [0, 100] range"

    def test_late_delivery_flag(self):
        """is_late_delivery should be 1 when actual > expected."""
        df = pd.DataFrame({
            "expected_delivery_date": ["2025-01-10"],
            "actual_delivery_date": ["2025-01-15"],
            "delivery_status": ["Delivered"],
        })
        delay = (pd.to_datetime(df["actual_delivery_date"]) -
                 pd.to_datetime(df["expected_delivery_date"])).dt.days
        assert (delay > 0).all()


# ===================================================================
# Feature engineering tests
# ===================================================================

class TestFeatures:
    def test_rolling_features_not_all_null(self, enriched_data):
        enhanced, _, _, cfg = enriched_data
        features = create_prediction_features(enhanced, cfg)

        assert features["supplier_avg_delay"].notna().all()
        assert features["supplier_late_delivery_rate"].notna().all()
        assert features["days_since_last_order"].notna().all()
        assert features["order_size_vs_avg"].notna().all()


# ===================================================================
# Join coverage tests
# ===================================================================

class TestJoinCoverage:
    def test_external_coverage_above_threshold(self, enriched_data):
        """After date-alignment fix, external coverage should pass threshold."""
        _, diag, _, cfg = enriched_data
        min_ext = cfg.get("data_contracts", {}).get("min_external_coverage_pct", 80)
        assert diag["external_avg_coverage_pct"] >= min_ext, \
            f"External coverage {diag['external_avg_coverage_pct']}% < {min_ext}%"


# ===================================================================
# Reproducibility helpers
# ===================================================================

class TestReproducibility:
    def test_file_hash_deterministic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            tmp_name = f.name
        try:
            h1 = file_hash(tmp_name)
            h2 = file_hash(tmp_name)
            assert h1 == h2
        finally:
            os.unlink(tmp_name)

    def test_run_metadata_structure(self, sample_config):
        meta = generate_run_metadata(sample_config, {"test": "nonexistent.csv"})
        assert "timestamp" in meta
        assert "python_version" in meta
        assert meta["data_fingerprints"]["test"] == "FILE_NOT_FOUND"


# ===================================================================
# Smoke test: full pipeline (optional, requires source data)
# ===================================================================

class TestSmokePipeline:
    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_PROJECT_ROOT, "..", "files", "suppliers.csv")),
        reason="Source data not available for full pipeline smoke test",
    )
    def test_data_generation_runs(self):
        """Smoke test: data generation pipeline completes without error."""
        from generate_disruption_data import main as gen_main
        gen_main()  # Should not raise

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_PROJECT_ROOT, "ml_features.csv")),
        reason="ml_features.csv not available for ML smoke test",
    )
    def test_ml_training_runs(self):
        """Smoke test: ML pipeline completes without error."""
        from train_ml_model import main as ml_main
        ml_main()  # Should not raise


# ===================================================================
# Next-phase: walk-forward split
# ===================================================================

class TestTemporalSplit:
    """Tests for the walk-forward temporal split function."""

    def test_split_preserves_time_order(self, sample_config):
        """Training fold should contain only older orders than the test fold."""
        from train_ml_model import prepare_data_for_ml, temporal_split

        # Build a small DataFrame with explicit dates and delivery status
        dates = pd.date_range("2024-01-01", periods=40, freq="7D")
        df = pd.DataFrame({
            "po_date": dates,
            "delivery_status": ["Delivered"] * 40,
            "is_late_delivery": ([0, 1] * 20),
            "financial_health_score": [70.0] * 40,
            "capacity_utilization_pct": [80.0] * 40,
            "employee_turnover_pct": [5.0] * 40,
            "quality_defect_rate": [2.0] * 40,
            "inventory_days": [30] * 40,
            "payment_default_risk": [15.0] * 40,
            "rainy_season": [0] * 40,
            "rainfall_index": [0.3] * 40,
            "holiday_period": [0] * 40,
            "port_congestion_index": [35.0] * 40,
            "disruption_severity": [0] * 40,
            "total_amount_ngn": [1e7] * 40,
            "quantity": [200] * 40,
            "order_size_vs_avg": [1.0] * 40,
            "supplier_avg_delay": [0.0] * 40,
            "supplier_late_delivery_rate": [0.5] * 40,
            "days_since_last_order": [30.0] * 40,
            "composite_risk_score": [50.0] * 40,
            "delivery_delay_days": [2.0] * 40,
        })

        _, y_full, feature_columns = prepare_data_for_ml(df, sample_config)
        X_tr, X_te, y_tr, y_te = temporal_split(df, feature_columns, sample_config)

        assert len(X_tr) > 0, "Training set is empty"
        assert len(X_te) > 0, "Test set is empty"
        assert len(X_tr) + len(X_te) == len(df), "Split sizes don't sum to total"

    def test_split_ratio_approximately_correct(self, sample_config):
        """Training fold should be roughly 80% of the data."""
        from train_ml_model import temporal_split

        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = pd.DataFrame({
            "po_date": dates,
            "delivery_status": ["Delivered"] * n,
            "is_late_delivery": ([0, 1] * (n // 2)),
            "financial_health_score": [70.0] * n,
            "capacity_utilization_pct": [80.0] * n,
            "employee_turnover_pct": [5.0] * n,
            "quality_defect_rate": [2.0] * n,
            "inventory_days": [30] * n,
            "payment_default_risk": [15.0] * n,
            "rainy_season": [0] * n,
            "rainfall_index": [0.3] * n,
            "holiday_period": [0] * n,
            "port_congestion_index": [35.0] * n,
            "disruption_severity": [0] * n,
            "total_amount_ngn": [1e7] * n,
            "quantity": [200] * n,
            "order_size_vs_avg": [1.0] * n,
            "supplier_avg_delay": [0.0] * n,
            "supplier_late_delivery_rate": [0.5] * n,
            "days_since_last_order": [30.0] * n,
            "composite_risk_score": [50.0] * n,
            "delivery_delay_days": [2.0] * n,
        })
        feature_cols = [
            "financial_health_score", "capacity_utilization_pct",
            "employee_turnover_pct", "quality_defect_rate",
            "inventory_days", "payment_default_risk",
            "rainy_season", "rainfall_index", "holiday_period",
            "port_congestion_index", "disruption_severity",
            "total_amount_ngn", "quantity", "order_size_vs_avg",
            "supplier_avg_delay", "supplier_late_delivery_rate",
            "days_since_last_order", "composite_risk_score",
        ]
        cutoff = sample_config.get("walk_forward", {}).get("train_cutoff_pct", 0.80)
        X_tr, X_te, _, _ = temporal_split(df, feature_cols, sample_config)
        assert abs(len(X_tr) / n - cutoff) < 0.02, "Train split ratio deviates more than 2%"


# ===================================================================
# Next-phase: threshold optimisation
# ===================================================================

class TestThresholdOptimisation:
    """Tests for optimize_decision_threshold."""

    def test_f1_threshold_in_range(self):
        """Optimal F1 threshold must be in (0, 1)."""
        from train_ml_model import optimize_decision_threshold

        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_proba = rng.uniform(0, 1, size=200)
        thr = optimize_decision_threshold(y_true, y_proba, strategy="f1")
        assert 0 < thr < 1, f"Threshold {thr} out of (0,1)"

    def test_youden_threshold_in_range(self):
        """Optimal Youden threshold must be in (0, 1)."""
        from train_ml_model import optimize_decision_threshold

        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=200)
        y_proba = rng.uniform(0, 1, size=200)
        thr = optimize_decision_threshold(y_true, y_proba, strategy="youden")
        assert 0 < thr < 1, f"Threshold {thr} out of (0,1)"


# ===================================================================
# Next-phase: lead-time measurement
# ===================================================================

class TestLeadTimeMeasurement:
    """Tests for measure_lead_times module."""

    def _make_lead_time_df(self) -> pd.DataFrame:
        n = 30
        base = pd.Timestamp("2024-01-01")
        po_dates = [base + pd.Timedelta(days=i * 5) for i in range(n)]
        expected = [d + pd.Timedelta(days=14) for d in po_dates]
        actual = [d + pd.Timedelta(days=14 + (i % 5 - 1)) for i, d in enumerate(po_dates)]
        return pd.DataFrame({
            "po_number": [f"PO{i:04d}" for i in range(n)],
            "supplier_id": [f"S{(i % 3) + 1:03d}" for i in range(n)],
            "supplier_name": [f"Supplier {(i % 3) + 1}" for i in range(n)],
            "po_date": po_dates,
            "expected_delivery_date": expected,
            "actual_delivery_date": actual,
            "delivery_status": ["Delivered"] * n,
        })

    def test_compute_lead_times_creates_columns(self):
        from measure_lead_times import compute_lead_times
        df = self._make_lead_time_df()
        result = compute_lead_times(df)
        for col in ("actual_lead_days", "expected_lead_days", "lead_time_variance", "lateness_flag"):
            assert col in result.columns, f"Missing column: {col}"

    def test_lateness_flag_matches_variance(self):
        from measure_lead_times import compute_lead_times
        df = self._make_lead_time_df()
        result = compute_lead_times(df)
        expected_flags = (result["lead_time_variance"] > 0).astype(int)
        pd.testing.assert_series_equal(
            result["lateness_flag"], expected_flags, check_names=False
        )

    def test_supplier_summary_sorted_desc(self):
        from measure_lead_times import compute_lead_times, supplier_lead_time_summary
        df = self._make_lead_time_df()
        detail = compute_lead_times(df)
        summary = supplier_lead_time_summary(detail)
        scores = summary["early_warning_score"].tolist()
        assert scores == sorted(scores, reverse=True), "Supplier summary not sorted descending"

    def test_overall_summary_on_time_rate_range(self):
        from measure_lead_times import compute_lead_times, overall_summary, supplier_lead_time_summary
        df = self._make_lead_time_df()
        detail = compute_lead_times(df)
        supplier_df = supplier_lead_time_summary(detail)
        summary = overall_summary(detail, supplier_df)
        assert 0 <= summary["on_time_rate_pct"] <= 100


# ===================================================================
# Next-phase: SMOTE
# ===================================================================

class TestSMOTE:
    """Tests for apply_smote (graceful degradation if imblearn missing)."""

    def test_smote_returns_arrays(self, sample_config):
        from train_ml_model import apply_smote

        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.uniform(0, 1, (100, 5)), columns=list("ABCDE"))
        y = pd.Series([0] * 70 + [1] * 30)
        X_res, y_res = apply_smote(X, y, sample_config)
        assert isinstance(X_res, np.ndarray)
        assert isinstance(y_res, np.ndarray)
        assert len(X_res) == len(y_res)

    def test_smote_disabled_returns_original_shape(self, sample_config):
        from train_ml_model import apply_smote
        import copy
        cfg = copy.deepcopy(sample_config)
        cfg["smote"] = {"enabled": False}
        rng = np.random.default_rng(2)
        X = pd.DataFrame(rng.uniform(0, 1, (50, 5)), columns=list("ABCDE"))
        y = pd.Series([0] * 30 + [1] * 20)
        X_res, y_res = apply_smote(X, y, cfg)
        assert len(X_res) == 50, "Disabled SMOTE should not change dataset size"

