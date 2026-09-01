#!/usr/bin/env python3
"""Run Phase 5.5: Persona Representation Confound Audit.

Executes dataset property auditing, length/format/lexical/neutral control evaluations,
expanded training stability scaling (B=500), effect-size comparisons, and report generation.

Usage:
    python scripts/run_persona_confound_audit.py \\
        --config configs/persona_confound_audit.yaml \\
        --pilot-activations results/raw/ACT-001/activations \\
        --control-activations results/raw/ACT-CONTROLS-001/activations \\
        --expanded-activations results/raw/ACT-EXPANDED-001/activations \\
        --experiment-id PERSONA-CONFOUND-AUDIT-001
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.activations.confound_audit import (
    audit_dataset_properties,
    compute_expanded_training_direction,
    cross_direction_comparison,
    evaluate_control_shifts,
    generate_confound_audit_figures,
    run_expanded_bootstrap_stability,
)
from src.activations.direction import compute_mean_difference_direction
from src.activations.storage import load_activations
from src.utils.logging import get_logger
from src.utils.reproducibility import capture_environment, set_seed

logger = get_logger("run_confound_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 5.5 Confound Audit.")
    parser.add_argument("--config", type=str, default="configs/persona_confound_audit.yaml", help="Path to config YAML.")
    parser.add_argument("--pilot-activations", type=str, default="results/raw/ACT-001/activations", help="Path to ACT-001 activations directory.")
    parser.add_argument("--control-activations", type=str, required=True, help="Path to ACT-CONTROLS-001 activations directory.")
    parser.add_argument("--expanded-activations", type=str, required=True, help="Path to ACT-EXPANDED-001 activations directory.")
    parser.add_argument("--experiment-id", type=str, default="PERSONA-CONFOUND-AUDIT-001", help="Experiment ID.")
    parser.add_argument("--output-dir", type=str, default="results", help="Base output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_id = args.experiment_id

    # Output paths
    base_out = Path(args.output_dir)
    raw_dir = base_out / "raw" / exp_id
    tables_dir = base_out / "tables" / exp_id
    figures_dir = base_out / "figures" / exp_id
    reports_dir = base_out / "reports" / exp_id

    for d in [raw_dir, tables_dir, figures_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load YAML config
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    target_layer = cfg.get("target_layer", 2)
    n_stability_resamples = cfg.get("n_stability_resamples", 500)
    n_delta_bootstrap = cfg.get("n_delta_bootstrap", 1000)
    n_random_directions = cfg.get("n_random_directions", 100)
    seed = cfg.get("seed", 42)

    set_seed(seed)

    logger.info("=" * 60)
    logger.info("PHASE 5.5: PERSONA REPRESENTATION CONFOUND AUDIT")
    logger.info(f"Experiment ID:        {exp_id}")
    logger.info(f"Target Layer:         Layer {target_layer}")
    logger.info(f"Stability Resamples:  {n_stability_resamples}")
    logger.info(f"Delta Bootstrap:      {n_delta_bootstrap}")
    logger.info("=" * 60)

    # 1. Part 1: Dataset property audit
    logger.info("Part 1: Auditing dataset properties...")
    pilot_manifest_path = Path(args.pilot_activations) / "manifest.json"
    with open(pilot_manifest_path, "r", encoding="utf-8") as f:
        pilot_manifest = json.load(f)
    pilot_prompts = pilot_manifest["prompts"]

    csv_stat_path = tables_dir / "persona_condition_statistics.csv"
    df_dataset_stats, dataset_audit_raw = audit_dataset_properties(pilot_prompts, output_csv_path=csv_stat_path)

    # Load original training activations to reconstruct D_original
    act_pilot_dict, _ = load_activations(Path(args.pilot_activations))
    splits = np.array([p["split"] for p in pilot_prompts])
    labels = np.array([1 if p["persona_label"] == "assistant" else 0 for p in pilot_prompts])
    train_mask = (splits == "train")

    X_train_orig = act_pilot_dict[target_layer][train_mask]
    y_train_orig = labels[train_mask]
    D_original = compute_mean_difference_direction(X_train_orig, y_train_orig, assistant_label=1)

    # 2. Parts 2-5, 8-10: Evaluate controls
    logger.info("Parts 2-5, 8-10: Evaluating controlled manipulations...")
    ctrl_manifest_path = Path(args.control_activations) / "manifest.json"
    with open(ctrl_manifest_path, "r", encoding="utf-8") as f:
        ctrl_manifest = json.load(f)
    control_prompts = ctrl_manifest["prompts"]

    act_ctrl_dict, _ = load_activations(Path(args.control_activations))
    X_ctrl = act_ctrl_dict[target_layer]

    df_controls, controls_raw = evaluate_control_shifts(
        control_prompts=control_prompts,
        control_activations=X_ctrl,
        direction=D_original,
        seed=seed,
        n_bootstrap=n_delta_bootstrap,
    )

    csv_ctrl_path = tables_dir / "confound_control_deltas.csv"
    df_controls.to_csv(csv_ctrl_path, index=False)
    logger.info(f"Saved control deltas table to {csv_ctrl_path}")

    # 3. Parts 6-7: Expanded training direction & stability scaling (B=500)
    logger.info("Parts 6-7: Fitting expanded training direction and running B=500 bootstrap stability...")
    exp_manifest_path = Path(args.expanded_activations) / "manifest.json"
    with open(exp_manifest_path, "r", encoding="utf-8") as f:
        exp_manifest = json.load(f)
    exp_prompts = exp_manifest["prompts"]

    act_exp_dict, _ = load_activations(Path(args.expanded_activations))
    X_exp = act_exp_dict[target_layer]
    y_exp = np.array([1 if p["persona_label"] == "assistant" else 0 for p in exp_prompts])

    D_expanded, cos_orig_exp = compute_expanded_training_direction(X_exp, y_exp, D_original)
    logger.info(f"Cosine similarity between D_original (N=24) and D_expanded (N={len(y_exp)}): {cos_orig_exp:.4f}")

    exp_stability = run_expanded_bootstrap_stability(
        X_exp_train=X_exp,
        y_exp_train=y_exp,
        expanded_direction=D_expanded,
        n_resamples=n_stability_resamples,
        seed=seed,
    )
    logger.info(
        f"Expanded Stability across B={n_stability_resamples}: mean cos_sim = {exp_stability['mean_cosine_similarity']:.4f} "
        f"± {exp_stability['std_cosine_similarity']:.4f} (95% CI: [{exp_stability['ci_95_lower']:.4f}, {exp_stability['ci_95_upper']:.4f}])"
    )

    cross_dir_stats = cross_direction_comparison(
        D_original=D_original,
        D_expanded=D_expanded,
        n_random=n_random_directions,
        seed=seed,
    )

    # 4. Part 10: Comparative effect size table
    effect_size_rows = []
    for _, row in df_controls.iterrows():
        effect_size_rows.append({
            "manipulation_type": row["control_type"],
            "n_pairs": int(row["n_pairs"]),
            "mean_abs_delta": float(row["mean_abs_delta"]),
            "signed_mean_delta": float(row["mean_delta"]),
            "cohens_dz": float(row["cohens_dz"]),
            "ci_95": f"[{row['ci_95_lower']:.4f}, {row['ci_95_upper']:.4f}]",
            "p_value_wilcoxon": float(row["p_val_wilcoxon"]),
        })
    df_effect_sizes = pd.DataFrame(effect_size_rows)
    csv_eff_path = tables_dir / "effect_size_comparison.csv"
    df_effect_sizes.to_csv(csv_eff_path, index=False)

    # 5. Part 13: Generate 9 publication-grade figures
    logger.info("Part 13: Generating 9 publication-grade figures...")
    audit_results_dict = {
        "dataset_audit": dataset_audit_raw,
        "controls_summary": df_controls,
        "controls_raw": controls_raw,
        "cross_direction": cross_dir_stats,
        "expanded_stability": exp_stability,
        "n_expanded_prompts": len(y_exp),
    }
    fig_paths = generate_confound_audit_figures(audit_results_dict, figures_dir)
    logger.info(f"Generated {len(fig_paths)} figures in {figures_dir}")

    # 6. Part 14: Save comprehensive metadata
    env = capture_environment(seed=seed)
    metadata = {
        "experiment_id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_layer": target_layer,
        "n_original_prompts": len(pilot_prompts),
        "n_expanded_prompts": len(y_exp),
        "n_control_prompts": len(control_prompts),
        "cos_orig_expanded": cos_orig_exp,
        "expanded_stability": {
            "mean": exp_stability["mean_cosine_similarity"],
            "median": exp_stability["median_cosine_similarity"],
            "std": exp_stability["std_cosine_similarity"],
            "min": exp_stability["min_cosine_similarity"],
            "max": exp_stability["max_cosine_similarity"],
            "ci_95": [exp_stability["ci_95_lower"], exp_stability["ci_95_upper"]],
        },
        "cross_direction": cross_dir_stats,
        "controls_summary": df_controls.to_dict(orient="records"),
        "environment": env.to_dict(),
    }

    meta_path = raw_dir / "confound_audit_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved audit metadata to {meta_path}")

    logger.info("=" * 60)
    logger.info("PHASE 5.5 AUDIT COMPLETE")
    logger.info(f"Original vs Expanded Cosine Similarity: {cos_orig_exp:.4f}")
    logger.info(f"Expanded Stability (B={n_stability_resamples}): {exp_stability['mean_cosine_similarity']:.4f}")
    logger.info(f"Report Directory: {reports_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
