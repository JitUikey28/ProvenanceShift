"""
Persona representation analysis — layer sweep, evaluation, statistical testing, and visualization.

Orchestrates:
    - Layer-by-layer evaluation on train / validation / test splits
    - Candidate direction projection and Cohen's d effect size
    - Linear classifier comparison
    - Random direction and label shuffling controls
    - Results table generation (CSV)
    - Publication-ready visualizations (Matplotlib)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.activations.direction import (
    bootstrap_confidence_interval,
    compute_cohens_d,
    compute_mean_difference_direction,
    compute_projection_metrics,
    evaluate_classifier,
    generate_random_direction,
    project_representations,
    shuffle_labels,
    train_linear_classifier,
)
from src.utils.logging import get_logger

logger = get_logger()


def run_layer_sweep(
    activations_by_layer: Dict[int, np.ndarray],
    manifest: Dict[str, Any],
    experiment_id: str,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    run_controls: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run full layer sweep evaluation on extracted activations.

    Parameters
    ----------
    activations_by_layer:
        Dictionary mapping layer index to activation array ``(N, hidden_dim)``.
    manifest:
        Extraction manifest containing prompt metadata (splits, labels).
    experiment_id:
        Experiment identifier.
    bootstrap_samples:
        Number of bootstrap iterations for CI calculation.
    confidence_level:
        Confidence level for CI (e.g. 0.95).
    seed:
        Random seed for reproducibility.
    run_controls:
        Whether to compute random direction and label shuffling controls.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - DataFrame containing metrics across all layers.
        - Detailed results dictionary including per-layer projections and metadata.
    """
    prompts = manifest["prompts"]

    # Extract splits and binary labels (1 = assistant, 0 = alternative)
    splits = np.array([p.get("split", "train") for p in prompts])
    labels = np.array([1 if p.get("persona_label") == "assistant" else 0 for p in prompts])

    train_mask = (splits == "train")
    val_mask = (splits == "validation")
    test_mask = (splits == "test")

    n_train = int(np.sum(train_mask))
    n_val = int(np.sum(val_mask))
    n_test = int(np.sum(test_mask))

    logger.info(
        f"Running layer sweep: {len(activations_by_layer)} layers "
        f"(train={n_train}, val={n_val}, test={n_test})..."
    )

    y_train = labels[train_mask]
    y_val = labels[val_mask]
    y_test = labels[test_mask]

    records: List[Dict[str, Any]] = []
    layer_details: Dict[int, Any] = {}

    for l_idx in sorted(activations_by_layer.keys()):
        X_all = activations_by_layer[l_idx]
        X_train = X_all[train_mask]
        X_val = X_all[val_mask]
        X_test = X_all[test_mask]

        hidden_dim = X_train.shape[1]

        # 1. Compute candidate direction strictly on TRAINING data
        try:
            direction = compute_mean_difference_direction(X_train, y_train, assistant_label=1)
        except ValueError:
            direction = np.zeros(hidden_dim, dtype=np.float64)

        # 2. Project all splits
        s_train = project_representations(X_train, direction)
        s_val = project_representations(X_val, direction)
        s_test = project_representations(X_test, direction)

        # 3. Compute decision threshold from training data
        train_metrics = compute_projection_metrics(y_train, s_train)
        threshold = train_metrics["threshold"]

        # 4. Evaluate on validation split
        val_metrics = compute_projection_metrics(y_val, s_val, threshold=threshold)

        # 5. Evaluate on held-out test split
        test_metrics = compute_projection_metrics(y_test, s_test, threshold=threshold)

        # 6. Bootstrap CI for test Cohen's d
        def cohen_boot_fn(yb: np.ndarray, sb: np.ndarray) -> float:
            return compute_cohens_d(sb[yb == 1], sb[yb == 0])

        ci_lower, ci_upper = bootstrap_confidence_interval(
            cohen_boot_fn,
            y_test,
            s_test,
            n_bootstrap=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        )

        # 7. Baseline Classifier (Logistic Regression)
        clf = train_linear_classifier(X_train, y_train, seed=seed)
        clf_val = evaluate_classifier(clf, X_val, y_val)
        clf_test = evaluate_classifier(clf, X_test, y_test)

        # 8. Controls (if requested)
        rand_auc = 0.5
        shuff_auc = 0.5
        if run_controls:
            # Control A: Random unit direction
            rand_dir = generate_random_direction(hidden_dim, seed=seed + l_idx)
            s_test_rand = project_representations(X_test, rand_dir)
            rand_metrics = compute_projection_metrics(y_test, s_test_rand)
            rand_auc = rand_metrics["roc_auc"]

            # Control B: Shuffled training labels
            y_train_shuff = shuffle_labels(y_train, seed=seed + l_idx)
            try:
                shuff_dir = compute_mean_difference_direction(X_train, y_train_shuff)
                s_test_shuff = project_representations(X_test, shuff_dir)
                shuff_metrics = compute_projection_metrics(y_test, s_test_shuff)
                shuff_auc = shuff_metrics["roc_auc"]
            except Exception:
                shuff_auc = 0.5

        rec = {
            "experiment_id": experiment_id,
            "layer": l_idx,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_f1": val_metrics["f1"],
            "val_cohens_d": val_metrics["cohens_d"],
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_f1": test_metrics["f1"],
            "test_cohens_d": test_metrics["cohens_d"],
            "test_mean_asst": test_metrics["mean_assistant_projection"],
            "test_mean_alt": test_metrics["mean_alternative_projection"],
            "test_cohens_d_ci_lower": ci_lower,
            "test_cohens_d_ci_upper": ci_upper,
            "clf_val_accuracy": clf_val["clf_accuracy"],
            "clf_val_roc_auc": clf_val["clf_roc_auc"],
            "clf_test_accuracy": clf_test["clf_accuracy"],
            "clf_test_roc_auc": clf_test["clf_roc_auc"],
            "control_random_test_auc": rand_auc,
            "control_shuffled_test_auc": shuff_auc,
        }
        records.append(rec)

        layer_details[l_idx] = {
            "direction": direction,
            "test_scores": s_test,
            "test_labels": y_test,
            "threshold": threshold,
        }

    df = pd.DataFrame(records)
    return df, layer_details


def save_analysis_results(
    df: pd.DataFrame,
    layer_details: Dict[int, Any],
    manifest: Dict[str, Any],
    experiment_id: str,
    output_tables_dir: Union[str, Path] = "results/tables",
    output_figures_dir: Union[str, Path] = "results/figures",
    output_raw_dir: Union[str, Path] = "results/raw",
) -> Dict[str, Path]:
    """Save analysis tables, figures, and metadata.

    Parameters
    ----------
    df:
        DataFrame from ``run_layer_sweep``.
    layer_details:
        Dictionary of layer-specific projection scores and directions.
    manifest:
        Original extraction manifest.
    experiment_id:
        Experiment identifier.
    output_tables_dir:
        Base path for tables.
    output_figures_dir:
        Base path for figures.
    output_raw_dir:
        Base path for raw metadata.

    Returns
    -------
    Dict[str, Path]
        Paths to created artifacts.
    """
    tbl_dir = Path(output_tables_dir) / experiment_id
    fig_dir = Path(output_figures_dir) / experiment_id
    raw_dir = Path(output_raw_dir) / experiment_id

    tbl_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save CSV table
    csv_path = tbl_dir / "layer_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved layer sweep table: {csv_path}")

    # 2. Select best layer by VALIDATION ROC-AUC (never test!)
    best_val_idx = int(df["val_roc_auc"].idxmax())
    best_layer = int(df.loc[best_val_idx, "layer"])
    logger.info(f"Selected best layer based on validation ROC-AUC: Layer {best_layer}")

    # 3. Generate figures
    # Style setup
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    layers = df["layer"].values

    # Figure 1: Layer vs Accuracy & ROC-AUC
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, df["val_accuracy"], marker="o", label="Validation Accuracy", color="#1f77b4", linewidth=2)
    ax.plot(layers, df["test_accuracy"], marker="s", label="Test Accuracy", color="#ff7f0e", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance Level (0.5)")
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)
    ax.set_title("Persona Classification Accuracy Across Model Layers", fontsize=14, pad=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig1_path = fig_dir / "layer_accuracy.png"
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)

    # Figure 2: Layer vs ROC-AUC with Controls
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, df["test_roc_auc"], marker="o", label="Test ROC-AUC", color="#2ca02c", linewidth=2)
    ax.plot(layers, df["control_random_test_auc"], marker="x", linestyle=":", label="Random Direction Control", color="#7f7f7f")
    ax.plot(layers, df["control_shuffled_test_auc"], marker="^", linestyle=":", label="Label Shuffled Control", color="#d62728")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("Test ROC-AUC and Baseline Controls Across Layers", fontsize=14, pad=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig2_path = fig_dir / "layer_roc_auc.png"
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)

    # Figure 3: Layer vs Cohen's d with Bootstrap CI
    fig, ax = plt.subplots(figsize=(8, 5))
    y_err_lower = np.maximum(0, df["test_cohens_d"].values - df["test_cohens_d_ci_lower"].values)
    y_err_upper = np.maximum(0, df["test_cohens_d_ci_upper"].values - df["test_cohens_d"].values)
    ax.errorbar(
        layers,
        df["test_cohens_d"],
        yerr=[y_err_lower, y_err_upper],
        fmt="o-",
        capsize=4,
        capthick=1.5,
        color="#9467bd",
        linewidth=2,
        label="Cohen's d (Test Set ± 95% CI)",
    )
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Standardized Effect Size (Cohen's d)", fontsize=12)
    ax.set_title("Separation Effect Size (Cohen's d) Across Layers", fontsize=14, pad=10)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig3_path = fig_dir / "layer_cohens_d.png"
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)

    # Figure 4: Projection Distributions on Held-Out Test Set (at Best Validation Layer)
    best_details = layer_details[best_layer]
    test_scores = best_details["test_scores"]
    test_labels = best_details["test_labels"]
    threshold = best_details["threshold"]

    asst_scores = test_scores[test_labels == 1]
    alt_scores = test_scores[test_labels == 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(np.min(test_scores) - 0.5, np.max(test_scores) + 0.5, 15)
    ax.hist(asst_scores, bins=bins, alpha=0.65, label=f"Assistant (n={len(asst_scores)})", color="#1f77b4", edgecolor="black")
    ax.hist(alt_scores, bins=bins, alpha=0.65, label=f"Alternative Persona (n={len(alt_scores)})", color="#ff7f0e", edgecolor="black")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Train Threshold ({threshold:.2f})")
    ax.set_xlabel(f"Projection onto Candidate Direction (Layer {best_layer})", fontsize=12)
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.set_title(f"Test Projection Distributions at Best Validation Layer (Layer {best_layer})", fontsize=14, pad=10)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig4_path = fig_dir / "projection_distribution.png"
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)

    # 4. Save analysis metadata
    meta_path = raw_dir / "persona_analysis_metadata.json"
    analysis_meta = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_validation_layer": best_layer,
        "best_validation_roc_auc": float(df.loc[best_val_idx, "val_roc_auc"]),
        "best_layer_test_roc_auc": float(df.loc[best_val_idx, "test_roc_auc"]),
        "best_layer_test_accuracy": float(df.loc[best_val_idx, "test_accuracy"]),
        "best_layer_test_cohens_d": float(df.loc[best_val_idx, "test_cohens_d"]),
        "model": manifest.get("model"),
        "extraction_config": manifest.get("extraction_config"),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(analysis_meta, fh, indent=2, default=str)
    logger.info(f"Saved analysis metadata: {meta_path}")

    return {
        "table": csv_path,
        "fig_accuracy": fig1_path,
        "fig_roc_auc": fig2_path,
        "fig_cohens_d": fig3_path,
        "fig_projection_distribution": fig4_path,
        "metadata": meta_path,
    }
