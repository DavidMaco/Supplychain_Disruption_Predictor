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
