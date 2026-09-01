# =============================================================================
# Confound Audit Engine — ProvenanceShift Phase 5.5
# =============================================================================
"""Module for auditing potential prompt construction confounds (length, format,
lexical style, neutral context) and assessing representation stability scaling.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.activations.direction import (
    compute_cohens_d,
    compute_mean_difference_direction,
    project_representations,
)
from src.utils.logging import get_logger

logger = get_logger("confound_audit")


# ---------------------------------------------------------------------------
# Part 1: Dataset Property Audit
# ---------------------------------------------------------------------------

def audit_dataset_properties(
    prompts: List[Dict[str, Any]],
    output_csv_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Extract and compare lexical and structural properties across persona conditions.

    Parameters
    ----------
    prompts:
        List of prompt dictionaries from the pilot dataset.
    output_csv_path:
        Optional path to save the resulting summary CSV.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Summary statistics table and detailed raw item records.
    """
    records = []
    for p in prompts:
        label = p.get("persona_label", "unknown")
        messages = p.get("messages", [])
        
        system_text = ""
        user_text = ""
        for m in messages:
            if m.get("role") == "system":
                system_text += m.get("content", "") + " "
            elif m.get("role") == "user":
                user_text += m.get("content", "") + " "

        full_text = (system_text + " " + user_text).strip()
        system_text = system_text.strip()
        user_text = user_text.strip()

        words = full_text.split()
        sys_words = system_text.split()
        user_words = user_text.split()

        char_count = len(full_text)
        word_count = len(words)
        token_count = p.get("token_metadata", {}).get("input_token_count", word_count)
        
        n_messages = len(messages)
        n_system = sum(1 for m in messages if m.get("role") == "system")
        n_user = sum(1 for m in messages if m.get("role") == "user")
        n_assistant = sum(1 for m in messages if m.get("role") == "assistant")

        # Formatting markers
        n_colons = full_text.count(":")
        n_hyphens = full_text.count("-")
        n_newlines = full_text.count("\n")
        n_quotes = full_text.count('"') + full_text.count("'")
        total_formatting_markers = n_colons + n_hyphens + n_newlines + n_quotes

        # Lexical diversity (type-token ratio)
        ttr = len(set(w.lower() for w in words)) / max(1, len(words))

        records.append({
            "prompt_id": p.get("prompt_id"),
            "split": p.get("split"),
            "persona_label": label,
            "char_count": char_count,
            "word_count": word_count,
            "token_count": token_count,
            "system_char_count": len(system_text),
            "system_word_count": len(sys_words),
            "user_char_count": len(user_text),
            "user_word_count": len(user_words),
            "n_messages": n_messages,
            "n_system_messages": n_system,
            "n_user_messages": n_user,
            "n_assistant_messages": n_assistant,
            "formatting_markers_count": total_formatting_markers,
            "lexical_type_token_ratio": ttr,
        })

    df_raw = pd.DataFrame(records)

    numeric_cols = [
        "char_count",
        "word_count",
        "token_count",
        "system_char_count",
        "system_word_count",
        "user_char_count",
        "user_word_count",
        "n_messages",
        "n_system_messages",
        "n_user_messages",
        "n_assistant_messages",
        "formatting_markers_count",
        "lexical_type_token_ratio",
    ]

    stat_rows = []
    asst_mask = (df_raw["persona_label"] == "assistant")
    alt_mask = (df_raw["persona_label"] == "alternative")

    for col in numeric_cols:
        v_asst = df_raw.loc[asst_mask, col].to_numpy(dtype=float)
        v_alt = df_raw.loc[alt_mask, col].to_numpy(dtype=float)

        d = compute_cohens_d(v_asst, v_alt)

        stat_rows.append({
            "property": col,
            "assistant_mean": float(np.mean(v_asst)),
            "assistant_median": float(np.median(v_asst)),
            "assistant_std": float(np.std(v_asst, ddof=1)) if len(v_asst) > 1 else 0.0,
            "assistant_min": float(np.min(v_asst)),
            "assistant_max": float(np.max(v_asst)),
            "alternative_mean": float(np.mean(v_alt)),
            "alternative_median": float(np.median(v_alt)),
            "alternative_std": float(np.std(v_alt, ddof=1)) if len(v_alt) > 1 else 0.0,
            "alternative_min": float(np.min(v_alt)),
            "alternative_max": float(np.max(v_alt)),
            "standardized_mean_diff_d": float(d),
        })

    df_summary = pd.DataFrame(stat_rows)

    if output_csv_path is not None:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(output_csv_path, index=False)
        logger.info(f"Saved dataset property audit to {output_csv_path}")

    return df_summary, {"raw_records": records, "df_raw": df_raw}


# ---------------------------------------------------------------------------
# Parts 2–5 & 8–10: Control Evaluation & Effect-Size Comparisons
# ---------------------------------------------------------------------------

def evaluate_control_shifts(
    control_prompts: List[Dict[str, Any]],
    control_activations: np.ndarray,
    direction: np.ndarray,
    seed: int = 42,
    n_bootstrap: int = 1000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute paired projection shifts and effect sizes across control manipulations.

    Parameters
    ----------
    control_prompts:
        List of control prompt metadata dicts.
    control_activations:
        Activations matrix of shape ``(n_prompts, hidden_dim)`` for Layer 2.
    direction:
        Candidate persona direction vector (Layer 2) from Phase 5.
    seed:
        Random seed for bootstrap CIs.
    n_bootstrap:
        Number of bootstrap resamples.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Summary statistics dataframe and per-task paired records.
    """
    scores = project_representations(control_activations, direction)

    # Attach projection scores to prompt dicts
    for i, p in enumerate(control_prompts):
        p["projection_score"] = float(scores[i])

    # Group by control_type and task_id
    grouped_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for p in control_prompts:
        ctype = p.get("control_type", "unspecified")
        grouped_by_type.setdefault(ctype, []).append(p)

    control_summaries: List[Dict[str, Any]] = []
    paired_records: List[Dict[str, Any]] = []

    rng = np.random.default_rng(seed)

    for ctype, items in grouped_by_type.items():
        # Group items by task_id
        tasks: Dict[str, List[Dict[str, Any]]] = {}
        for item in items:
            tid = item.get("metadata", {}).get("task_id", "default")
            tasks.setdefault(tid, []).append(item)

        deltas: List[float] = []
        abs_deltas: List[float] = []

        for tid, task_items in tasks.items():
            if ctype == "length_control":
                short_item = next((x for x in task_items if x.get("condition_name") == "length_short"), None)
                med_item = next((x for x in task_items if x.get("condition_name") == "length_medium"), None)
                long_item = next((x for x in task_items if x.get("condition_name") == "length_long"), None)
                
                if short_item and med_item:
                    d_sm = med_item["projection_score"] - short_item["projection_score"]
                    deltas.append(d_sm)
                    abs_deltas.append(abs(d_sm))
                    paired_records.append({"control_type": ctype, "task_id": tid, "comparison": "medium_vs_short", "delta": d_sm})
                if med_item and long_item:
                    d_ml = long_item["projection_score"] - med_item["projection_score"]
                    deltas.append(d_ml)
                    abs_deltas.append(abs(d_ml))
                    paired_records.append({"control_type": ctype, "task_id": tid, "comparison": "long_vs_medium", "delta": d_ml})

            elif ctype == "positive_persona_control":
                asst_item = next((x for x in task_items if x.get("persona_label") == "assistant"), None)
                alt_item = next((x for x in task_items if x.get("persona_label") == "alternative"), None)
                if asst_item and alt_item:
                    # Persona manipulation: Assistant - Alternative (positive shift convention)
                    d_pos = asst_item["projection_score"] - alt_item["projection_score"]
                    deltas.append(d_pos)
                    abs_deltas.append(abs(d_pos))
                    paired_records.append({"control_type": ctype, "task_id": tid, "comparison": "assistant_vs_alternative", "delta": d_pos})

            else:
                # Baseline vs Modified
                base_item = next((x for x in task_items if "baseline" in x.get("condition_name", "") or "standard" in x.get("prompt_id", "")), None)
                if not base_item and task_items:
                    base_item = task_items[0]

                for mod_item in task_items:
                    if mod_item is not base_item:
                        d_val = mod_item["projection_score"] - base_item["projection_score"]
                        deltas.append(d_val)
                        abs_deltas.append(abs(d_val))
                        paired_records.append({"control_type": ctype, "task_id": tid, "comparison": f"{mod_item.get('condition_name')}_vs_base", "delta": d_val})

        if not deltas:
            continue

        d_arr = np.array(deltas, dtype=float)
        abs_arr = np.array(abs_deltas, dtype=float)
        n = len(d_arr)

        mean_d = float(np.mean(d_arr))
        median_d = float(np.median(d_arr))
        std_d = float(np.std(d_arr, ddof=1)) if n > 1 else 0.0
        mean_abs_d = float(np.mean(abs_arr))

        # Effect size dz = mean_delta / std_delta
        dz = float(mean_d / std_d) if std_d > 1e-12 else 0.0

        # Paired statistics
        if n >= 2 and not np.all(d_arr == d_arr[0]):
            try:
                t_stat, p_ttest = stats.ttest_1samp(d_arr, 0.0)
            except Exception:
                t_stat, p_ttest = 0.0, 1.0
            try:
                w_stat, p_wilcoxon = stats.wilcoxon(d_arr)
            except Exception:
                w_stat, p_wilcoxon = 0.0, 1.0
        else:
            p_ttest = 1.0
            p_wilcoxon = 1.0

        # Bootstrap 95% CI on mean signed delta
        boot_means = []
        for _ in range(n_bootstrap):
            b_idx = rng.choice(n, size=n, replace=True)
            boot_means.append(np.mean(d_arr[b_idx]))
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        control_summaries.append({
            "control_type": ctype,
            "n_pairs": n,
            "mean_delta": mean_d,
            "median_delta": median_d,
            "std_delta": std_d,
            "mean_abs_delta": mean_abs_d,
            "cohens_dz": dz,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "p_val_wilcoxon": float(p_wilcoxon),
            "p_val_ttest": float(p_ttest),
        })

    df_control_summary = pd.DataFrame(control_summaries)
    return df_control_summary, {"paired_records": paired_records, "prompts": control_prompts}


# ---------------------------------------------------------------------------
# Part 6 & 7: Expanded Training Direction & Stability Scaling
# ---------------------------------------------------------------------------

def compute_expanded_training_direction(
    X_exp_train: np.ndarray,
    y_exp_train: np.ndarray,
    original_direction: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Fit candidate direction on expanded training set and compute cosine similarity.

    Parameters
    ----------
    X_exp_train:
        Expanded training activations of shape ``(n_exp, hidden_dim)``.
    y_exp_train:
        Binary labels (1 = assistant, 0 = alternative).
    original_direction:
        Original 24-sample direction vector.

    Returns
    -------
    Tuple[np.ndarray, float]
        Expanded direction vector and cosine similarity to original direction.
    """
    exp_dir = compute_mean_difference_direction(X_exp_train, y_exp_train, assistant_label=1)
    cos_sim = float(np.dot(original_direction, exp_dir))
    return exp_dir, cos_sim


def run_expanded_bootstrap_stability(
    X_exp_train: np.ndarray,
    y_exp_train: np.ndarray,
    expanded_direction: np.ndarray,
    n_resamples: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Assess stability of the expanded training direction across B bootstrap resamples.

    Parameters
    ----------
    X_exp_train:
        Expanded training activations.
    y_exp_train:
        Binary labels.
    expanded_direction:
        Base direction fitted on full expanded training set.
    n_resamples:
        Number of bootstrap iterations (e.g. 500).
    seed:
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Stability distribution statistics and array of similarities.
    """
    rng = np.random.default_rng(seed)
    n = len(y_exp_train)
    similarities: List[float] = []

    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_exp_train[idx]
        if len(np.unique(y_b)) < 2:
            continue
        X_b = X_exp_train[idx]
        try:
            d_boot = compute_mean_difference_direction(X_b, y_b, assistant_label=1)
            sim = float(np.dot(expanded_direction, d_boot))
            similarities.append(sim)
        except Exception:
            continue

    sim_arr = np.array(similarities)
    return {
        "n_resamples_collected": len(similarities),
        "mean_cosine_similarity": float(np.mean(sim_arr)),
        "median_cosine_similarity": float(np.median(sim_arr)),
        "std_cosine_similarity": float(np.std(sim_arr, ddof=1)),
        "min_cosine_similarity": float(np.min(sim_arr)),
        "max_cosine_similarity": float(np.max(sim_arr)),
        "ci_95_lower": float(np.percentile(sim_arr, 2.5)),
        "ci_95_upper": float(np.percentile(sim_arr, 97.5)),
        "similarities": similarities,
    }


def cross_direction_comparison(
    D_original: np.ndarray,
    D_expanded: np.ndarray,
    n_random: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare original, expanded, and random control directions.

    Parameters
    ----------
    D_original:
        Original Phase 4/5 direction.
    D_expanded:
        Expanded training direction.
    n_random:
        Number of random directions to sample.
    seed:
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Cosine similarities between D_orig, D_exp, and D_random.
    """
    dim = len(D_original)
    cos_orig_exp = float(np.dot(D_original, D_expanded))

    rng = np.random.default_rng(seed)
    cos_orig_rand: List[float] = []
    cos_exp_rand: List[float] = []

    for _ in range(n_random):
        z = rng.normal(size=dim)
        v_rand = z / np.linalg.norm(z)
        cos_orig_rand.append(float(np.dot(D_original, v_rand)))
        cos_exp_rand.append(float(np.dot(D_expanded, v_rand)))

    return {
        "cos_orig_expanded": cos_orig_exp,
        "mean_cos_orig_rand": float(np.mean(cos_orig_rand)),
        "std_cos_orig_rand": float(np.std(cos_orig_rand, ddof=1)),
        "max_cos_orig_rand": float(np.max(cos_orig_rand)),
        "mean_cos_exp_rand": float(np.mean(cos_exp_rand)),
        "std_cos_exp_rand": float(np.std(cos_exp_rand, ddof=1)),
        "max_cos_exp_rand": float(np.max(cos_exp_rand)),
    }


# ---------------------------------------------------------------------------
# Part 13: Visualizations
# ---------------------------------------------------------------------------

def generate_confound_audit_figures(
    audit_results: Dict[str, Any],
    output_dir: Path,
) -> List[Path]:
    """Generate the 9 publication-grade diagnostic figures required for Phase 5.5.

    Parameters
    ----------
    audit_results:
        Dictionary containing all audit data (dataset stats, controls, stability).
    output_dir:
        Directory where figures will be saved.

    Returns
    -------
    List[Path]
        List of generated figure file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_paths: List[Path] = []

    # 1. Prompt-length distributions by condition (char count)
    df_raw = audit_results["dataset_audit"]["df_raw"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    asst_chars = df_raw.loc[df_raw["persona_label"] == "assistant", "char_count"]
    alt_chars = df_raw.loc[df_raw["persona_label"] == "alternative", "char_count"]
    ax.hist(asst_chars, bins=8, alpha=0.6, label="Assistant Condition", edgecolor="black")
    ax.hist(alt_chars, bins=8, alpha=0.6, label="Alternative Condition", edgecolor="black")
    ax.set_xlabel("Character Count")
    ax.set_ylabel("Number of Prompts")
    ax.set_title("Prompt Character Count Distribution by Condition")
    ax.legend(frameon=True)
    p1 = output_dir / "1_prompt_length_distributions.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    fig_paths.append(p1)

    # 2. Token-count distributions
    fig, ax = plt.subplots(figsize=(7, 4.5))
    asst_tokens = df_raw.loc[df_raw["persona_label"] == "assistant", "token_count"]
    alt_tokens = df_raw.loc[df_raw["persona_label"] == "alternative", "token_count"]
    ax.hist(asst_tokens, bins=8, alpha=0.6, label="Assistant Condition", edgecolor="black")
    ax.hist(alt_tokens, bins=8, alpha=0.6, label="Alternative Condition", edgecolor="black")
    ax.set_xlabel("Input Token Count")
    ax.set_ylabel("Number of Prompts")
    ax.set_title("Input Token Count Distribution by Condition")
    ax.legend(frameon=True)
    p2 = output_dir / "2_token_count_distributions.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    fig_paths.append(p2)

    # 3. Formatting/control projection shifts
    df_controls = audit_results["controls_summary"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ctypes = df_controls["control_type"].tolist()
    mean_shifts = df_controls["mean_delta"].tolist()
    errs = [
        [m - l for m, l in zip(mean_shifts, df_controls["ci_95_lower"])],
        [u - m for m, u in zip(mean_shifts, df_controls["ci_95_upper"])],
    ]
    ax.bar(range(len(ctypes)), mean_shifts, yerr=errs, capsize=5, edgecolor="black", alpha=0.7)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(ctypes)))
    ax.set_xticklabels([c.replace("_", "\n") for c in ctypes], fontsize=9)
    ax.set_ylabel("Mean Signed Delta (Score)")
    ax.set_title("Mean Projection Shift Across Controlled Manipulations (95% CI)")
    p3 = output_dir / "3_control_projection_shifts.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    fig_paths.append(p3)

    # 4. Persona vs surface-control Delta distributions (boxplots)
    paired_records = audit_results["controls_raw"]["paired_records"]
    df_paired = pd.DataFrame(paired_records)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_groups = []
    group_labels = []
    for ctype in df_controls["control_type"].unique():
        sub = df_paired[df_paired["control_type"] == ctype]["delta"].tolist()
        if sub:
            plot_groups.append(sub)
            group_labels.append(ctype.replace("_", "\n"))
    if plot_groups:
        try:
            ax.boxplot(plot_groups, tick_labels=group_labels, patch_artist=True)
        except TypeError:
            ax.boxplot(plot_groups, labels=group_labels, patch_artist=True)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_ylabel("Paired Delta (Shift)")
        ax.set_title("Distribution of Projection Shifts Across Controls")
    p4 = output_dir / "4_persona_vs_surface_deltas.png"
    fig.tight_layout()
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    fig_paths.append(p4)

    # 5. Direction cosine similarity: original vs expanded
    cross_sim = audit_results["cross_direction"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ["D_orig vs D_exp", "D_orig vs D_rand", "D_exp vs D_rand"]
    vals = [
        cross_sim["cos_orig_expanded"],
        cross_sim["mean_cos_orig_rand"],
        cross_sim["mean_cos_exp_rand"],
    ]
    ax.bar(bars, vals, edgecolor="black", alpha=0.7)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Direction Vector Similarity Comparison")
    ax.set_ylim(-0.2, 1.1)
    p5 = output_dir / "5_direction_cosine_similarity.png"
    fig.tight_layout()
    fig.savefig(p5, dpi=300)
    plt.close(fig)
    fig_paths.append(p5)

    # 6. Bootstrap stability distribution (expanded training set B=500)
    exp_boot = audit_results["expanded_stability"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(exp_boot["similarities"], bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(exp_boot["mean_cosine_similarity"], linestyle="-", linewidth=2, label=f"Mean = {exp_boot['mean_cosine_similarity']:.3f}")
    ax.axvline(exp_boot["ci_95_lower"], linestyle="--", linewidth=1.5, label=f"95% CI [{exp_boot['ci_95_lower']:.3f}, {exp_boot['ci_95_upper']:.3f}]")
    ax.axvline(exp_boot["ci_95_upper"], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Cosine Similarity to Full-Sample Direction")
    ax.set_ylabel("Bootstrap Frequency (B=500)")
    ax.set_title(f"Direction Stability Distribution on Expanded Training Set (N={audit_results['n_expanded_prompts']})")
    ax.legend(frameon=True)
    p6 = output_dir / "6_bootstrap_stability_distribution.png"
    fig.tight_layout()
    fig.savefig(p6, dpi=300)
    plt.close(fig)
    fig_paths.append(p6)

    # 7. Comparative effect-size figure (Mean |Delta|)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    abs_vals = df_controls["mean_abs_delta"].tolist()
    ax.bar(range(len(ctypes)), abs_vals, edgecolor="black", alpha=0.7)
    ax.set_xticks(range(len(ctypes)))
    ax.set_xticklabels([c.replace("_", "\n") for c in ctypes], fontsize=9)
    ax.set_ylabel("Mean |Delta Projection|")
    ax.set_title("Comparative Magnitude of Projection Movement Across Conditions")
    p7 = output_dir / "7_comparative_effect_sizes.png"
    fig.tight_layout()
    fig.savefig(p7, dpi=300)
    plt.close(fig)
    fig_paths.append(p7)

    # 8. Projection shift vs prompt length (character count)
    control_prompts = audit_results["controls_raw"]["prompts"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lens = [len(" ".join(m.get("content", "") for m in p.get("messages", []))) for p in control_prompts]
    proj_scores = [p["projection_score"] for p in control_prompts]
    ax.scatter(lens, proj_scores, alpha=0.7, edgecolors="black")
    ax.set_xlabel("Total Prompt Character Length")
    ax.set_ylabel("Layer-2 Projection Score")
    ax.set_title("Projection Score vs Total Prompt Character Length")
    p8 = output_dir / "8_projection_vs_char_length.png"
    fig.tight_layout()
    fig.savefig(p8, dpi=300)
    plt.close(fig)
    fig_paths.append(p8)

    # 9. Projection shift vs token count
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tokens = [len(" ".join(m.get("content", "") for m in p.get("messages", [])).split()) for p in control_prompts]
    ax.scatter(tokens, proj_scores, alpha=0.7, edgecolors="black")
    ax.set_xlabel("Estimated Prompt Word/Token Count")
    ax.set_ylabel("Layer-2 Projection Score")
    ax.set_title("Projection Score vs Prompt Word Count")
    p9 = output_dir / "9_projection_vs_word_count.png"
    fig.tight_layout()
    fig.savefig(p9, dpi=300)
    plt.close(fig)
    fig_paths.append(p9)

    return fig_paths
