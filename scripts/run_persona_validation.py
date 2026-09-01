#!/usr/bin/env python3
"""
Run Phase 5 persona representation validation.

Performs:
    1. Layer-sweep evaluation across train/validation/test splits.
    2. Validation-based selection of layer L*.
    3. Direction stability analysis via bootstrap resampling.
    4. Random-direction empirical null distribution.
    5. Train-fitted PCA visualization.
    6. Generates tables, publication figures, and validation report.

Usage:
    python scripts/run_persona_validation.py \\
        --activations-dir results/raw/ACT-001/activations \\
        --config configs/persona_validation.yaml \\
        --experiment-id PERSONA-VALIDATION-001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 5 persona representation validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--activations-dir",
        type=str,
        required=True,
        help="Path to directory containing activations.npz and manifest.json.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/persona_validation.yaml",
        help="Path to persona validation YAML config (default: configs/persona_validation.yaml).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Override experiment ID.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.activations.storage import load_activations
    from src.activations.validation import run_persona_validation
    from src.utils.logging import get_logger

    # Load validation config
    config_path = Path(args.config)
    cfg: dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    criterion = cfg.get("layer_selection_criterion", "val_roc_auc")
    n_random = cfg.get("n_random_directions", 100)
    n_stability = cfg.get("n_stability_resamples", 100)
    bootstrap_samples = cfg.get("bootstrap_samples", 1000)
    confidence_level = cfg.get("confidence_level", 0.95)
    seed = cfg.get("seed", 42)

    # Load activations
    act_dir = Path(args.activations_dir)
    activations_by_layer, manifest = load_activations(act_dir)

    experiment_id = args.experiment_id or f"VAL-{manifest.get('experiment_id', 'EXP')}"
    logger = get_logger(experiment_id=experiment_id)

    logger.info("=" * 60)
    logger.info("PHASE 5: PERSONA REPRESENTATION VALIDATION")
    logger.info(f"Experiment ID:        {experiment_id}")
    logger.info(f"Selection criterion:  {criterion} (validation set only)")
    logger.info(f"Stability resamples:  {n_stability}")
    logger.info(f"Random null count:    {n_random}")
    logger.info("=" * 60)

    # Run validation pipeline
    df, report, artifacts = run_persona_validation(
        activations_by_layer=activations_by_layer,
        manifest=manifest,
        experiment_id=experiment_id,
        layer_selection_criterion=criterion,
        n_random_directions=n_random,
        n_stability_resamples=n_stability,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )

    logger.info("=" * 60)
    logger.info("VALIDATION COMPLETE")
    logger.info(f"Selected Layer:         Layer {report['selected_layer']}")
    logger.info(f"Validation ROC-AUC:     {report['validation_metrics']['roc_auc']:.4f}")
    logger.info(f"Held-out Test ROC-AUC:  {report['test_metrics']['roc_auc']:.4f}")
    logger.info(f"Held-out Cohen's d:     {report['test_metrics']['cohens_d']:.4f}")
    logger.info(f"Direction Stability:    {report['direction_stability']['mean_cosine_similarity']:.4f}")
    logger.info(f"Empirical Null p-value: {report['random_direction_control']['empirical_p_value']:.4f}")
    logger.info(f"Validation Table:       {artifacts['table']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
