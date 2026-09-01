"""
Rigorous validation of candidate persona representations (Phase 5).

Methodological principles:
    - Layer and method selection are performed strictly on VALIDATION data.
    - Test data is evaluated only once on the chosen configuration.
    - Stability under bootstrap resampling is quantified via cosine similarity.
    - Random-direction performance is evaluated as an empirical null distribution.
    - PCA is fitted strictly on training data for descriptive visualization.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.activations.analysis import run_layer_sweep
from src.activations.direction import (
    bootstrap_confidence_interval,
    compute_cohens_d,
    compute_mean_difference_direction,
    compute_projection_metrics,
    generate_random_direction,
    project_representations,
    shuffle_labels,
    train_linear_classifier,
    evaluate_classifier,
)
from src.utils.logging import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Validation-Based Selection
# ---------------------------------------------------------------------------

def select_best_layer(
    layer_df: pd.DataFrame,
    criterion: str = "val_roc_auc",
) -> int:
    """Select the best layer based exclusively on validation set metrics.

    Parameters
    ----------
    layer_df:
        DataFrame containing layer-by-layer sweep metrics.
    criterion:
        Column name to maximize on validation data (e.g. ``"val_roc_auc"``, ``"val_cohens_d"``).

    Returns
    -------
    int
        Selected layer index.
    """
    if criterion not in layer_df.columns:
        raise ValueError(
            f"Criterion '{criterion}' not found in layer sweep table. "
            f"Available columns: {list(layer_df.columns)}"
        )
    best_idx = layer_df[criterion].idxmax()
    best_layer = int(layer_df.loc[best_idx, "layer"])
    logger.info(
        f"Validation layer selection: Layer {best_layer} selected based on "
        f"maximum {criterion} ({layer_df.loc[best_idx, criterion]:.4f})."
    )
    return best_layer


# ---------------------------------------------------------------------------
# Stability Analysis
# ---------------------------------------------------------------------------

def compute_direction_stability(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_resamples: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Assess representation stability by bootstrap resampling the training set.

    Parameters
    ----------
    X_train:
        Training activations of shape ``(n_train, hidden_dim)``.
    y_train:
        Training labels of shape ``(n_train,)``.
    n_resamples:
        Number of bootstrap resampling iterations.
    seed:
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Dictionary with mean cosine similarity to the full-sample direction,
        standard deviation, 95% confidence interval, and individual similarities.
    """
    rng = np.random.default_rng(seed)
    n_train = len(y_train)

    # Base direction computed on full training set
    base_dir = compute_mean_difference_direction(X_train, y_train, assistant_label=1)

    cosine_sims: List[float] = []

    for _ in range(n_resamples):
        boot_idx = rng.choice(n_train, size=n_train, replace=True)
        X_b = X_train[boot_idx]
        y_b = y_train[boot_idx]

        # Check that both classes exist in resample
        if len(np.unique(y_b)) < 2:
            continue

        try:
            boot_dir = compute_mean_difference_direction(X_b, y_b, assistant_label=1)
            cos_sim = float(np.dot(base_dir, boot_dir))
            cosine_sims.append(cos_sim)
        except Exception:
            continue

    if not cosine_sims:
        return {
            "mean_cosine_similarity": 0.0,
            "std_cosine_similarity": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "resamples_collected": 0,
        }

    sims_arr = np.array(cosine_sims)
    mean_sim = float(np.mean(sims_arr))
    std_sim = float(np.std(sims_arr))
    ci_lower = float(np.percentile(sims_arr, 2.5))
    ci_upper = float(np.percentile(sims_arr, 97.5))

    logger.info(
        f"Direction stability across {len(sims_arr)} resamples: "
        f"mean cos_sim = {mean_sim:.4f} ± {std_sim:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
    )

    return {
        "mean_cosine_similarity": mean_sim,
        "std_cosine_similarity": std_sim,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "resamples_collected": len(sims_arr),
        "cosine_similarities": cosine_sims,
    }


# ---------------------------------------------------------------------------
# Random Direction Null Distribution
# ---------------------------------------------------------------------------

def run_random_direction_distribution(
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_directions: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Construct the empirical null distribution from random unit directions.

    Parameters
    ----------
    X_test:
        Test set activations of shape ``(n_test, hidden_dim)``.
    y_test:
        Test set labels of shape ``(n_test,)``.
    n_directions:
        Number of random directions to evaluate.
    seed:
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Empirical null distribution metrics (mean, std, 95th percentile, raw scores).
    """
    hidden_dim = X_test.shape[1]
    aucs: List[float] = []
    accuracies: List[float] = []
    cohens_ds: List[float] = []

    for i in range(n_directions):
        r_dir = generate_random_direction(hidden_dim, seed=seed + i * 13)
        scores = project_representations(X_test, r_dir)
        m = compute_projection_metrics(y_test, scores)
        aucs.append(m["roc_auc"])
        accuracies.append(m["accuracy"])
        cohens_ds.append(m["cohens_d"])

    return {
        "n_directions": n_directions,
        "mean_random_roc_auc": float(np.mean(aucs)),
        "std_random_roc_auc": float(np.std(aucs)),
        "p95_random_roc_auc": float(np.percentile(aucs, 95.0)),
        "mean_random_accuracy": float(np.mean(accuracies)),
        "std_random_accuracy": float(np.std(accuracies)),
        "mean_random_cohens_d": float(np.mean(cohens_ds)),
        "raw_aucs": aucs,
    }


# ---------------------------------------------------------------------------
# Train-Isolated PCA Projection
# ---------------------------------------------------------------------------

def compute_train_fitted_pca(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 2,
) -> Tuple[PCA, np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA strictly on training activations and transform all splits.

    Parameters
    ----------
    X_train:
        Training activations.
    X_val:
        Validation activations.
    X_test:
        Held-out test activations.
    n_components:
        Number of principal components.

    Returns
    -------
    Tuple[PCA, np.ndarray, np.ndarray, np.ndarray]
        (fitted_pca, X_train_pca, X_val_pca, X_test_pca)
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    return pca, X_train_pca, X_val_pca, X_test_pca


# ---------------------------------------------------------------------------
# Full Validation Pipeline
# ---------------------------------------------------------------------------

def run_persona_validation(
    activations_by_layer: Dict[int, np.ndarray],
    manifest: Dict[str, Any],
    experiment_id: str,
    layer_selection_criterion: str = "val_roc_auc",
    n_random_directions: int = 100,
    n_stability_resamples: int = 100,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    output_tables_dir: Union[str, Path] = "results/tables",
    output_figures_dir: Union[str, Path] = "results/figures",
    output_raw_dir: Union[str, Path] = "results/raw",
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Path]]:
    """Execute full Phase 5 persona representation validation.

    Parameters
    ----------
    activations_by_layer:
        Dictionary mapping layer index to activation array.
    manifest:
        Extraction manifest with prompts and metadata.
    experiment_id:
        Unique experiment identifier.
    layer_selection_criterion:
        Metric on validation split used to select layer $L^*$.
    n_random_directions:
        Number of random directions for empirical null distribution.
    n_stability_resamples:
        Number of resamples for direction stability.
    bootstrap_samples:
        Bootstrap iterations for test CI.
    confidence_level:
        Confidence level for CI (0.95).
    seed:
        Random seed.
    output_tables_dir:
        Directory for CSV tables.
    output_figures_dir:
        Directory for figures.
    output_raw_dir:
        Directory for metadata.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Path]]
        - Layer sweep DataFrame.
        - Detailed validation report dictionary.
        - Dictionary of generated artifact paths.
    """
    tbl_dir = Path(output_tables_dir) / experiment_id
    fig_dir = Path(output_figures_dir) / experiment_id
    raw_dir = Path(output_raw_dir) / experiment_id

    tbl_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    prompts = manifest["prompts"]
    splits = np.array([p.get("split", "train") for p in prompts])
    labels = np.array([1 if p.get("persona_label") == "assistant" else 0 for p in prompts])

    train_mask = (splits == "train")
    val_mask = (splits == "validation")
    test_mask = (splits == "test")

    # 1. Run full layer sweep
    layer_df, layer_details = run_layer_sweep(
        activations_by_layer=activations_by_layer,
        manifest=manifest,
        experiment_id=experiment_id,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
        run_controls=True,
    )

    # 2. Select layer L* based strictly on VALIDATION split
    selected_layer = select_best_layer(layer_df, criterion=layer_selection_criterion)

    # 3. Deep-dive validation on selected layer L*
    X_all = activations_by_layer[selected_layer]
    X_train = X_all[train_mask]
    y_train = labels[train_mask]
    X_val = X_all[val_mask]
    y_val = labels[val_mask]
    X_test = X_all[test_mask]
    y_test = labels[test_mask]

    # Compute candidate direction
    direction = compute_mean_difference_direction(X_train, y_train, assistant_label=1)
    s_test = project_representations(X_test, direction)

    train_m = compute_projection_metrics(y_train, project_representations(X_train, direction))
    val_m = compute_projection_metrics(y_val, project_representations(X_val, direction), threshold=train_m["threshold"])
    test_m = compute_projection_metrics(y_test, s_test, threshold=train_m["threshold"])

    # 4. Stability analysis at selected layer
    stability_res = compute_direction_stability(
        X_train=X_train,
        y_train=y_train,
        n_resamples=n_stability_resamples,
        seed=seed,
    )

    # 5. Empirical random direction null distribution at selected layer
    rand_dist_res = run_random_direction_distribution(
        X_test=X_test,
        y_test=y_test,
        n_directions=n_random_directions,
        seed=seed,
    )

    # Compute empirical p-value for test ROC-AUC against random distribution
    empirical_p = float(np.mean(np.array(rand_dist_res["raw_aucs"]) >= test_m["roc_auc"]))

    # 6. Train-fitted PCA
    pca, X_train_pca, X_val_pca, X_test_pca = compute_train_fitted_pca(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        n_components=2,
    )

    # 7. Save CSV Table
    csv_path = tbl_dir / "persona_validation.csv"
    layer_df.to_csv(csv_path, index=False)

    # 8. Generate Publication Figures
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure A: Stability Cosine Similarity Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(stability_res.get("cosine_similarities", [1.0]), bins=15, color="#17becf", edgecolor="black", alpha=0.7)
    ax.axvline(stability_res["mean_cosine_similarity"], color="red", linestyle="--", label=f"Mean Cosine Sim ({stability_res['mean_cosine_similarity']:.3f})")
    ax.set_xlabel("Cosine Similarity to Full-Sample Direction", fontsize=11)
    ax.set_ylabel("Resample Count", fontsize=11)
    ax.set_title(f"Direction Stability Across {n_stability_resamples} Bootstrap Resamples (Layer {selected_layer})", fontsize=12)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig_stab_path = fig_dir / "direction_stability.png"
    fig.savefig(fig_stab_path, dpi=300)
    plt.close(fig)

    # Figure B: Random Direction Null Distribution vs Learned Direction
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(rand_dist_res["raw_aucs"], bins=15, color="#7f7f7f", edgecolor="black", alpha=0.6, label="Random Directions Null")
    ax.axvline(test_m["roc_auc"], color="#2ca02c", linewidth=2.5, linestyle="-", label=f"Learned Direction (AUC={test_m['roc_auc']:.2f})")
    ax.axvline(0.5, color="black", linestyle=":", alpha=0.7, label="Chance Level (0.50)")
    ax.set_xlabel("Test ROC-AUC", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Learned Direction vs Random Direction Null (Layer {selected_layer}, p={empirical_p:.3f})", fontsize=12)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig_null_path = fig_dir / "random_direction_null.png"
    fig.savefig(fig_null_path, dpi=300)
    plt.close(fig)

    # Figure C: Train-Fitted PCA Visualization (Descriptive Only)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], marker="o", color="#1f77b4", label="Train: Assistant", alpha=0.7)
    ax.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], marker="o", color="#ff7f0e", label="Train: Alternative", alpha=0.7)
    ax.scatter(X_test_pca[y_test == 1, 0], X_test_pca[y_test == 1, 1], marker="^", s=80, color="#1f77b4", edgecolor="black", label="Test: Assistant (Held-out)")
    ax.scatter(X_test_pca[y_test == 0, 0], X_test_pca[y_test == 0, 1], marker="s", s=80, color="#ff7f0e", edgecolor="black", label="Test: Alternative (Held-out)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    ax.set_title(f"Train-Fitted PCA (Layer {selected_layer}) — Descriptive Projection", fontsize=12)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    fig_pca_path = fig_dir / "train_fitted_pca.png"
    fig.savefig(fig_pca_path, dpi=300)
    plt.close(fig)

    # 9. Save Validation Metadata JSON
    meta_path = raw_dir / "persona_validation_metadata.json"
    validation_report = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "selected_layer": selected_layer,
        "selection_criterion": layer_selection_criterion,
        "validation_metrics": val_m,
        "test_metrics": test_m,
        "direction_stability": {
            "mean_cosine_similarity": stability_res["mean_cosine_similarity"],
            "std_cosine_similarity": stability_res["std_cosine_similarity"],
            "ci_95": [stability_res["ci_lower"], stability_res["ci_upper"]],
        },
        "random_direction_control": {
            "n_directions": n_random_directions,
            "mean_random_roc_auc": rand_dist_res["mean_random_roc_auc"],
            "empirical_p_value": empirical_p,
        },
        "model": manifest.get("model"),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(validation_report, fh, indent=2, default=str)

    artifacts = {
        "table": csv_path,
        "fig_stability": fig_stab_path,
        "fig_null": fig_null_path,
        "fig_pca": fig_pca_path,
        "metadata": meta_path,
    }

    return layer_df, validation_report, artifacts
