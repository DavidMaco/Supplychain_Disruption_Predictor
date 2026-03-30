"""
Supply Chain Disruption Predictor - Next-Phase Pipeline Orchestrator
Runs all next-phase components in sequence:
  1. Data generation       (generate_disruption_data.py)
  2. ML training           (train_ml_model.py)          — walk-forward + SMOTE + threshold
  3. Model comparison      (compare_models.py)           — RF vs XGBoost
  4. Lead-time analysis    (measure_lead_times.py)       — validates early-warning claim

Usage:
  python run_next_phase.py             # full pipeline
  python run_next_phase.py --skip-gen  # skip data generation (reuse existing CSVs)
  python run_next_phase.py --compare-only  # run comparison only
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

from pipeline_utils import load_config, setup_logging

logger = setup_logging("next_phase")


def _run_step(name: str, fn, *args, **kwargs) -> bool:
    """Run a pipeline step with timing and error handling.

    Returns True on success, False on failure.
    """
    logger.info("=" * 60)
    logger.info("STEP: %s", name)
    logger.info("=" * 60)
    t0 = time.perf_counter()
    try:
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("✓ %s completed in %.1fs", name, elapsed)
        return True
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("✗ %s FAILED after %.1fs: %s", name, elapsed, exc)
        return False


def main(skip_gen: bool = False, compare_only: bool = False) -> None:
    """Orchestrate the full next-phase pipeline.

    Args:
        skip_gen: If True, skip data generation (reuse existing CSVs).
        compare_only: If True, run only the model comparison step.
    """
    cfg = load_config()
    started_at = datetime.now(timezone.utc).isoformat()
    logger.info("Next-phase pipeline started at %s", started_at)

    steps_run = 0
    steps_failed = 0

    if compare_only:
        # ── Model comparison only ──────────────────────────────────
        from compare_models import compare_models
        ok = _run_step("Model comparison (RF vs XGBoost)", compare_models, cfg)
        steps_run += 1
        if not ok:
            steps_failed += 1
    else:
        # ── Step 1: Data generation ────────────────────────────────
        if not skip_gen:
            from generate_disruption_data import main as gen_main
            ok = _run_step("Data generation", gen_main)
            steps_run += 1
            if not ok:
                steps_failed += 1
                logger.warning("Data generation failed — downstream steps may fail.")
        else:
            logger.info("Skipping data generation (--skip-gen flag set)")

        # ── Step 2: ML training (walk-forward + SMOTE + threshold) ─
        from train_ml_model import main as ml_main
        ok = _run_step("ML training (walk-forward + SMOTE + threshold)", ml_main)
        steps_run += 1
        if not ok:
            steps_failed += 1

        # ── Step 3: Model comparison ───────────────────────────────
        from compare_models import compare_models
        ok = _run_step("Model comparison (RF vs XGBoost)", compare_models, cfg)
        steps_run += 1
        if not ok:
            steps_failed += 1

        # ── Step 4: Lead-time analysis ─────────────────────────────
        from measure_lead_times import main as lt_main
        ok = _run_step("Lead-time analysis", lt_main)
        steps_run += 1
        if not ok:
            steps_failed += 1

    # ── Final summary ──────────────────────────────────────────────
    logger.info("=" * 60)
    if steps_failed == 0:
        logger.info("ALL %d STEPS COMPLETED SUCCESSFULLY", steps_run)
    else:
        logger.warning("%d/%d steps completed  |  %d FAILED", steps_run - steps_failed, steps_run, steps_failed)
    logger.info(
        "Dashboard: run  -->  streamlit run dashboard.py"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the next-phase Supply Chain Disruption Predictor pipeline"
    )
    parser.add_argument(
        "--skip-gen",
        action="store_true",
        help="Skip data generation and reuse existing CSVs",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Run model comparison only (requires existing ml_features.csv)",
    )
    args = parser.parse_args()

    try:
        main(skip_gen=args.skip_gen, compare_only=args.compare_only)
    except Exception as exc:
        logger.exception("Next-phase pipeline failed: %s", exc)
        sys.exit(1)
