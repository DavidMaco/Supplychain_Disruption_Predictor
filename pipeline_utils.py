"""
Supply Chain Disruption Predictor - Pipeline Utilities
Centralized config loading, logging setup, and shared helpers.
"""

import yaml
import logging
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load and cache the YAML configuration file.

    Args:
        path: Absolute or relative path to config.yaml.
              Defaults to config.yaml in the project root.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    if path is None:
        path = os.path.join(_PROJECT_ROOT, "config.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        _CONFIG_CACHE = yaml.safe_load(fh)

    return _CONFIG_CACHE


def reset_config_cache() -> None:
    """Clear cached config (useful for testing)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(name: str = "disruption_predictor", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with console and file handlers.

    Args:
        name: Logger name.
        level: Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_dir = os.path.join(_PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(log_dir, f"{name}.log"), encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Data-contract validation
# ---------------------------------------------------------------------------

def validate_dataframe_schema(df, required_columns: list, label: str = "DataFrame") -> None:
    """Raise ValueError if required columns are missing from a DataFrame.

    Args:
        df: pandas DataFrame to validate.
        required_columns: List of column names that must be present.
        label: Human-readable label for error messages.

    Raises:
        ValueError: With details about which columns are missing.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"[{label}] Missing required columns: {sorted(missing)}")


def validate_join_coverage(
    df,
    columns: list,
    min_coverage_pct: float,
    label: str = "join",
) -> Dict[str, Any]:
    """Check that enriched columns meet minimum non-null coverage.

    Args:
        df: DataFrame after a join operation.
        columns: Columns to check for nulls.
        min_coverage_pct: Minimum acceptable non-null percentage (0-100).
        label: Human-readable label for diagnostics.

    Returns:
        Dictionary with per-column coverage stats and pass/fail flag.

    Raises:
        ValueError: If any column falls below the threshold.
    """
    total = len(df)
    diagnostics: Dict[str, Any] = {"label": label, "total_rows": total, "columns": {}}
    failures = []

    for col in columns:
        if col not in df.columns:
            diagnostics["columns"][col] = {"coverage_pct": 0.0, "nulls": total, "status": "MISSING"}
            failures.append(col)
            continue
        nulls = int(df[col].isna().sum())
        coverage = round((1 - nulls / total) * 100, 2) if total > 0 else 0.0
        status = "PASS" if coverage >= min_coverage_pct else "FAIL"
        diagnostics["columns"][col] = {"coverage_pct": coverage, "nulls": nulls, "status": status}
        if status == "FAIL":
            failures.append(col)

    diagnostics["pass"] = len(failures) == 0
    diagnostics["failed_columns"] = failures

    if failures:
        raise ValueError(
            f"[{label}] Coverage below {min_coverage_pct}% for: {failures}. "
            f"Full diagnostics: {json.dumps(diagnostics, indent=2)}"
        )

    return diagnostics


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

def file_hash(path: str) -> str:
    """Return SHA-256 hex digest of a file for reproducibility tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_run_metadata(config: Dict[str, Any], data_files: Dict[str, str]) -> Dict[str, Any]:
    """Build a run-metadata record for reproducibility.

    Args:
        config: The loaded config dictionary.
        data_files: Mapping of label -> filepath for input data files.

    Returns:
        Metadata dictionary ready to be serialized.
    """
    meta: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "config_seed": config.get("random_seed"),
        "data_fingerprints": {},
    }
    for label, path in data_files.items():
        if os.path.exists(path):
            meta["data_fingerprints"][label] = file_hash(path)
        else:
            meta["data_fingerprints"][label] = "FILE_NOT_FOUND"
    return meta
