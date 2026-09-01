"""
Controlled provenance experiments — matched comparisons, paired statistics, and behavioral association (Phase 6).

Scientific logic:
    - Tests H1 (perceived provenance shifts persona representation) vs H0 (no systematic shift beyond confounds).
    - Compares 4 matched conditions: baseline, provenance_manipulation, surface_control, neutral_control.
    - Evaluates paired differences (Delta = Condition - Baseline) and Primary Paired Net Shift (Delta_net = Delta_provenance - Delta_surface).
    - Applies paired task-unit bootstrap resampling.
    - Analyzes association between representation shift and behavioral output shift.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.behavioral import BehavioralEvaluator, BehavioralMetrics
from src.utils.logging import get_logger

logger = get_logger("provenance_experiment")


# ---------------------------------------------------------------------------
# Matched Data Structuring & Delta Computation
# ---------------------------------------------------------------------------

def group_matched_tasks(
    prompt_items: Sequence[Dict[str, Any]],
    projection_scores: Sequence[float],
    behavioral_metrics: Optional[Sequence[BehavioralMetrics]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Group items by task_id and condition, pairing projections and behaviors.

    Parameters
    ----------
    prompt_items:
        List of prompt item dictionaries containing ``task_id`` and ``condition``.
    projection_scores:
        List of scalar persona projection scores.
    behavioral_metrics:
        Optional list of ``BehavioralMetrics`` for generated responses.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, Any]]]
        Nested dictionary: ``task_id -> condition -> {score, prompt_item, behavior}``.
    """
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for i, item in enumerate(prompt_items):
        t_id = item.get("task_id", f"task_{i}")
        cond = item.get("condition", "baseline")
        score = float(projection_scores[i])
        bm = behavioral_metrics[i] if behavioral_metrics and i < len(behavioral_metrics) else None

        if t_id not in grouped:
            grouped[t_id] = {}

        grouped[t_id][cond] = {
            "score": score,
            "prompt_item": item,
            "behavior": bm,
        }

    return grouped


def compute_paired_deltas(
    grouped_tasks: Dict[str, Dict[str, Dict[str, Any]]],
    baseline_condition: str = "baseline",
    target_conditions: Sequence[str] = ("provenance_manipulation", "surface_control", "neutral_control"),
) -> pd.DataFrame:
    """Compute paired shifts relative to baseline for representation and behavior.
    Also computes the primary contrast: delta_net = delta_provenance - delta_surface.

    Parameters
    ----------
    grouped_tasks:
        Grouped task mapping from ``group_matched_tasks``.
    baseline_condition:
        Condition to use as reference (default ``"baseline"``).
    target_conditions:
        Conditions to compare against baseline.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per valid matched task group.
    """
    records: List[Dict[str, Any]] = []

    for t_id, cond_map in grouped_tasks.items():
        if baseline_condition not in cond_map:
            continue

        base_entry = cond_map[baseline_condition]
        base_score = base_entry["score"]
        base_bm = base_entry["behavior"]
        base_item = base_entry["prompt_item"]
        domain = base_item.get("metadata", {}).get("topic", base_item.get("domain", "general"))

        rec: Dict[str, Any] = {
            "task_id": t_id,
            "domain": domain,
            "baseline_score": base_score,
        }
        if base_bm:
            rec["baseline_formality"] = base_bm.formality_score
            rec["baseline_fp_rate"] = base_bm.first_person_rate
            rec["baseline_word_count"] = base_bm.word_count

        for cond in target_conditions:
            if cond in cond_map:
                target_entry = cond_map[cond]
                target_score = target_entry["score"]
                delta_score = target_score - base_score
                rec[f"{cond}_score"] = target_score
                rec[f"delta_score_{cond}"] = delta_score

                target_bm = target_entry["behavior"]
                if target_bm and base_bm:
                    rec[f"delta_formality_{cond}"] = target_bm.formality_score - base_bm.formality_score
                    rec[f"delta_fp_rate_{cond}"] = target_bm.first_person_rate - base_bm.first_person_rate
                    rec[f"delta_word_count_{cond}"] = target_bm.word_count - base_bm.word_count

        # Compute PRIMARY paired net shift: delta_net = delta_provenance - delta_surface
        if "delta_score_provenance_manipulation" in rec and "delta_score_surface_control" in rec:
            rec["delta_net_score"] = rec["delta_score_provenance_manipulation"] - rec["delta_score_surface_control"]

        if "delta_formality_provenance_manipulation" in rec and "delta_formality_surface_control" in rec:
            rec["delta_net_formality"] = rec["delta_formality_provenance_manipulation"] - rec["delta_formality_surface_control"]

        if "delta_fp_rate_provenance_manipulation" in rec and "delta_fp_rate_surface_control" in rec:
            rec["delta_net_fp_rate"] = rec["delta_fp_rate_provenance_manipulation"] - rec["delta_fp_rate_surface_control"]

        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Paired Statistical Tests & Effect Sizes
# ---------------------------------------------------------------------------

def compute_paired_statistics(
    deltas: np.ndarray,
    test_name: str = "provenance_vs_baseline",
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute comprehensive paired statistics, effect sizes, and bootstrap CIs on differences.

    Parameters
    ----------
    deltas:
        1D array of paired differences.
    test_name:
        Descriptive label for this comparison.
    confidence_level:
        Confidence level for CI (e.g. 0.95).
    n_bootstrap:
        Number of paired bootstrap iterations.
    seed:
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Dictionary with mean delta, median, std, paired Cohen's d_z, p-values, and bootstrap CI.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    n = len(deltas)
    if n < 3:
        return {"n_pairs": n, "mean_delta": float(np.mean(deltas)) if n > 0 else 0.0}

    mean_delta = float(np.mean(deltas))
    median_delta = float(np.median(deltas))
    std_delta = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
    mean_abs_delta = float(np.mean(np.abs(deltas)))
    median_abs_delta = float(np.median(np.abs(deltas)))

    pct_positive = float(np.mean(deltas > 0) * 100.0)
    pct_negative = float(np.mean(deltas < 0) * 100.0)

    # Paired Cohen's d_z = mean(delta) / std(delta)
    cohens_dz = float(mean_delta / std_delta) if std_delta > 1e-12 else 0.0

    # Normality test
    try:
        shapiro_stat, shapiro_p = stats.shapiro(deltas)
    except Exception:
        shapiro_stat, shapiro_p = 1.0, 1.0

    # Parametric paired t-test (vs 0)
    t_stat, t_pvalue = stats.ttest_1samp(deltas, popmean=0.0)

    # Non-parametric Wilcoxon signed-rank test (vs 0)
    if not np.all(deltas == deltas[0]):
        try:
            w_stat, w_pvalue = stats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
        except Exception:
            w_stat, w_pvalue = 0.0, 1.0
    else:
        w_stat, w_pvalue = 0.0, 1.0

    # Paired bootstrap CI on mean delta
    rng = np.random.default_rng(seed)
    boot_means = []
    alpha = (1.0 - confidence_level) / 2.0
    for _ in range(n_bootstrap):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means.append(np.mean(sample))

    ci_lower = float(np.percentile(boot_means, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))

    return {
        "test_name": test_name,
        "n_pairs": n,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "std_delta": std_delta,
        "mean_abs_delta": mean_abs_delta,
        "median_abs_delta": median_abs_delta,
        "pct_positive": pct_positive,
        "pct_negative": pct_negative,
        "cohens_dz": cohens_dz,
        "shapiro_normality_p": float(shapiro_p),
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pvalue),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pvalue),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# ---------------------------------------------------------------------------
# Multiple Comparisons Correction
# ---------------------------------------------------------------------------

def apply_multiple_comparisons_correction(
    p_values: Sequence[float],
    method: str = "fdr_bh",
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply family-wise or false discovery rate corrections to p-values."""
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = len(p_arr)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)

    if method == "bonferroni":
        adj_p = np.clip(p_arr * m, 0.0, 1.0)
    elif method == "holm":
        order = np.argsort(p_arr)
        adj_p = np.zeros(m)
        cum_max = 0.0
        for i, idx in enumerate(order):
            step_adj = p_arr[idx] * (m - i)
            cum_max = max(cum_max, step_adj)
            adj_p[idx] = min(cum_max, 1.0)
    elif method == "fdr_bh":
        order = np.argsort(p_arr)
        ranks = np.empty(m, int)
        ranks[order] = np.arange(1, m + 1)
        adj_p = np.clip(p_arr * m / ranks, 0.0, 1.0)
        rev_order = order[::-1]
        for i in range(1, m):
            curr_idx = rev_order[i]
            prev_idx = rev_order[i - 1]
            if adj_p[curr_idx] > adj_p[prev_idx]:
                adj_p[curr_idx] = adj_p[prev_idx]
    else:
        raise ValueError(f"Unknown correction method '{method}'.")

    reject = adj_p < 0.05
    return adj_p, reject


# ---------------------------------------------------------------------------
# Representation vs Behavior Association
# ---------------------------------------------------------------------------

def compute_representation_behavior_association(
    delta_representations: np.ndarray,
    delta_behaviors: np.ndarray,
) -> Dict[str, Any]:
    """Compute correlation metrics between persona representation shift and behavioral shift."""
    dr = np.asarray(delta_representations, dtype=np.float64)
    db = np.asarray(delta_behaviors, dtype=np.float64)

    valid_mask = ~np.isnan(dr) & ~np.isnan(db)
    dr = dr[valid_mask]
    db = db[valid_mask]

    if len(dr) < 3 or np.all(dr == dr[0]) or np.all(db == db[0]):
        return {
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
        }

    r, p_r = stats.pearsonr(dr, db)
    rho, p_rho = stats.spearmanr(dr, db)

    return {
        "pearson_r": float(r),
        "pearson_p": float(p_r),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
    }


# ---------------------------------------------------------------------------
# Full Provenance Experiment Analysis & Visualizations
# ---------------------------------------------------------------------------

def run_provenance_analysis(
    delta_df: pd.DataFrame,
    experiment_id: str,
    output_tables_dir: Union[str, Path] = "results/tables",
    output_figures_dir: Union[str, Path] = "results/figures",
    output_raw_dir: Union[str, Path] = "results/raw",
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Path]]:
    """Run paired statistical analysis across provenance, controls, and primary net shift."""
    tbl_dir = Path(output_tables_dir) / experiment_id
    fig_dir = Path(output_figures_dir) / experiment_id
    raw_dir = Path(output_raw_dir) / experiment_id

    tbl_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    comparisons = [
        ("provenance_manipulation_vs_baseline", "delta_score_provenance_manipulation", "Raw Provenance Shift (Prov - Base)"),
        ("surface_control_vs_baseline", "delta_score_surface_control", "Surface Control Shift (Surf - Base)"),
        ("neutral_control_vs_baseline", "delta_score_neutral_control", "Neutral Control Shift (Neut - Base)"),
    ]
    if "delta_net_score" in delta_df.columns:
        comparisons.append(("net_provenance_vs_surface", "delta_net_score", "PRIMARY Net Provenance Shift (Delta_prov - Delta_surf)"))

    stat_records: List[Dict[str, Any]] = []

    for comp_key, col_name, label in comparisons:
        if col_name in delta_df.columns:
            deltas = delta_df[col_name].dropna().values
            res = compute_paired_statistics(
                deltas=deltas,
                test_name=comp_key,
                seed=seed,
            )
            res["label"] = label
            stat_records.append(res)

    summary_df = pd.DataFrame(stat_records)

    # Multiple comparison correction on Wilcoxon p-values
    if len(summary_df) > 0 and "wilcoxon_pvalue" in summary_df.columns:
        p_vals = summary_df["wilcoxon_pvalue"].values
        adj_p, reject = apply_multiple_comparisons_correction(p_vals, method="fdr_bh")
        summary_df["wilcoxon_pvalue_fdr_adj"] = adj_p
        summary_df["reject_null_fdr"] = reject

    # Save summary table
    csv_path = tbl_dir / "provenance_pilot.csv"
    summary_df.to_csv(csv_path, index=False)

    # Save paired deltas table
    delta_csv_path = tbl_dir / "provenance_paired_deltas.csv"
    delta_df.to_csv(delta_csv_path, index=False)

    # Behavioral Associations
    assoc_records = {}
    if "delta_score_provenance_manipulation" in delta_df.columns:
        for b_metric in ["formality", "fp_rate", "word_count", "ttr"]:
            b_col = f"delta_{b_metric}_provenance_manipulation"
            if b_col in delta_df.columns:
                assoc_records[f"raw_prov_vs_{b_metric}"] = compute_representation_behavior_association(
                    delta_df["delta_score_provenance_manipulation"].values,
                    delta_df[b_col].values,
                )

    if "delta_net_score" in delta_df.columns:
        for b_metric in ["formality", "fp_rate"]:
            b_col = f"delta_net_{b_metric}"
            if b_col in delta_df.columns:
                assoc_records[f"net_prov_vs_net_{b_metric}"] = compute_representation_behavior_association(
                    delta_df["delta_net_score"].values,
                    delta_df[b_col].values,
                )

    # Generate Figures
    fig_paths = {}

    # Figure 1: Distribution of Paired Deltas Across Conditions (including Net Shift)
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_data = []
    labels_list = []
    for comp_key, col_name, label in comparisons:
        if col_name in delta_df.columns:
            plot_data.append(delta_df[col_name].dropna().values)
            labels_list.append(comp_key.replace("_", " ").title())

    if plot_data:
        try:
            ax.boxplot(plot_data, tick_labels=labels_list, patch_artist=True)
        except TypeError:
            ax.boxplot(plot_data, labels=labels_list, patch_artist=True)
        ax.axhline(0.0, linestyle="--", linewidth=1.5)
        ax.set_ylabel("Representation Shift (Delta Score)")
        ax.set_title("Distribution of Representation Shifts (32 Matched Tasks)")
    fig.tight_layout()
    fig_deltas_path = fig_dir / "1_provenance_delta_distribution.png"
    fig.savefig(fig_deltas_path, dpi=300)
    plt.close(fig)
    fig_paths["fig_deltas"] = fig_deltas_path

    # Figure 2: Task-level Trajectories (Baseline -> Surface vs Baseline -> Provenance)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    sub_conditions = [
        ("provenance_manipulation", "Provenance Manipulation"),
        ("surface_control", "Surface Control"),
        ("neutral_control", "Neutral Control"),
    ]
    for ax, (c_key, c_title) in zip(axes, sub_conditions):
        s_col = f"{c_key}_score"
        if s_col in delta_df.columns and "baseline_score" in delta_df.columns:
            for _, row in delta_df.iterrows():
                ax.plot([0, 1], [row["baseline_score"], row[s_col]], marker="o", alpha=0.5)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Baseline", c_title])
            ax.set_title(c_title)
            if ax is axes[0]:
                ax.set_ylabel("Layer-2 Projection Score")
    fig.tight_layout()
    fig_traj_path = fig_dir / "2_task_level_trajectories.png"
    fig.savefig(fig_traj_path, dpi=300)
    plt.close(fig)
    fig_paths["fig_trajectories"] = fig_traj_path

    # Figure 3: Mean Effect Sizes with 95% Bootstrap CIs
    fig, ax = plt.subplots(figsize=(8, 4.5))
    comp_labels = summary_df["test_name"].tolist()
    means = summary_df["mean_delta"].tolist()
    errs = [
        [m - l for m, l in zip(means, summary_df["ci_lower"])],
        [u - m for m, u in zip(means, summary_df["ci_upper"])],
    ]
    ax.bar(range(len(comp_labels)), means, yerr=errs, capsize=5, edgecolor="black", alpha=0.7)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(comp_labels)))
    ax.set_xticklabels([c.replace("_", "\n").title() for c in comp_labels])
    ax.set_ylabel("Mean Delta (95% Bootstrap CI)")
    ax.set_title("Mean Shift vs Control Subtraction Across Matched Tasks")
    fig.tight_layout()
    fig_means_path = fig_dir / "3_condition_effects_ci95.png"
    fig.savefig(fig_means_path, dpi=300)
    plt.close(fig)
    fig_paths["fig_means"] = fig_means_path

    # Figure 4: Representation Shift vs Behavioral Shift
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if "delta_score_provenance_manipulation" in delta_df.columns and "delta_formality_provenance_manipulation" in delta_df.columns:
        x_pts = delta_df["delta_score_provenance_manipulation"].values
        y_pts = delta_df["delta_formality_provenance_manipulation"].values
        ax.scatter(x_pts, y_pts, alpha=0.8, edgecolors="black")
        r_info = assoc_records.get("raw_prov_vs_formality", {})
        ax.set_xlabel("Persona Representation Shift (Delta Score)")
        ax.set_ylabel("Behavioral Formality Shift (Delta Formality)")
        ax.set_title(
            f"Representation Shift vs Behavioral Formality (r={r_info.get('pearson_r', 0.0):.2f}, p={r_info.get('pearson_p', 1.0):.3f})"
        )
    fig.tight_layout()
    fig_assoc_path = fig_dir / "4_representation_vs_behavior.png"
    fig.savefig(fig_assoc_path, dpi=300)
    plt.close(fig)
    fig_paths["fig_association"] = fig_assoc_path

    # Save Experiment Metadata
    meta_path = raw_dir / "provenance_experiment_metadata.json"
    provenance_meta = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_matched_tasks": len(delta_df),
        "comparisons": summary_df.to_dict(orient="records"),
        "behavioral_associations": assoc_records,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(provenance_meta, fh, indent=2, default=str)

    artifacts = {
        "summary_table": csv_path,
        "paired_deltas_table": delta_csv_path,
        "fig_deltas": fig_deltas_path,
        "fig_association": fig_assoc_path,
        "fig_trajectories": fig_traj_path,
        "fig_means": fig_means_path,
        "figures": fig_paths,
        "metadata": meta_path,
    }

    return summary_df, provenance_meta, artifacts
