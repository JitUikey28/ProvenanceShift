#!/usr/bin/env python3
"""
Run persona representation analysis across model layers.

Reads extracted activations and evaluates candidate persona directions,
effect sizes (Cohen's d with bootstrap CIs), classification baselines,
and baseline controls.

Usage:
    python scripts/run_persona_analysis.py \\
        --activations-dir results/raw/ACT-001/activations \\
        --config configs/persona.yaml
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
        description="Run persona representation layer sweep and statistical analysis.",
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
        default="configs/persona.yaml",
        help="Path to persona analysis YAML config (default: configs/persona.yaml).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Override experiment ID (defaults to ID recorded in activation manifest).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Override number of bootstrap iterations.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.activations.analysis import run_layer_sweep, save_analysis_results
    from src.activations.storage import load_activations
    from src.utils.logging import get_logger

    # Load persona analysis configuration
    config_path = Path(args.config)
    cfg: dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    bootstrap_samples = (
        args.bootstrap_samples
        if args.bootstrap_samples is not None
        else cfg.get("bootstrap_samples", 1000)
    )
    confidence_level = cfg.get("confidence_level", 0.95)
    seed = cfg.get("seed", 42)
    run_controls = cfg.get("run_random_direction_control", True)

    # Load activations & manifest
    act_dir = Path(args.activations_dir)
    activations_by_layer, manifest = load_activations(act_dir)

    experiment_id = args.experiment_id or manifest.get("experiment_id", "PERSONA-EXP")
    logger = get_logger(experiment_id=experiment_id)

    logger.info("=" * 60)
    logger.info("PERSONA REPRESENTATION ANALYSIS")
    logger.info(f"Experiment ID:      {experiment_id}")
    logger.info(f"Activations Dir:    {act_dir}")
    logger.info(f"Layers loaded:      {len(activations_by_layer)}")
    logger.info(f"Bootstrap samples:  {bootstrap_samples}")
    logger.info("=" * 60)

    # Run layer sweep
    df, layer_details = run_layer_sweep(
        activations_by_layer=activations_by_layer,
        manifest=manifest,
        experiment_id=experiment_id,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
        run_controls=run_controls,
    )

    # Save tables, figures, and metadata
    saved_artifacts = save_analysis_results(
        df=df,
        layer_details=layer_details,
        manifest=manifest,
        experiment_id=experiment_id,
    )

    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Table saved:     {saved_artifacts['table']}")
    logger.info(f"Accuracy plot:   {saved_artifacts['fig_accuracy']}")
    logger.info(f"ROC-AUC plot:    {saved_artifacts['fig_roc_auc']}")
    logger.info(f"Cohen's d plot:  {saved_artifacts['fig_cohens_d']}")
    logger.info(f"Distributions:   {saved_artifacts['fig_projection_distribution']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
