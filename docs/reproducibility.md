# Reproducibility

This document describes how ProvenanceShift experiments, representations, and provenance investigations are made reproducible.

## Core Principle

**Every experiment must be reproducible from its configuration and code repository state alone.**

Given an experiment/model/persona/provenance configuration file and a specific git commit, any researcher should be able to re-run the pipeline and obtain identical representations, paired differences, and statistical distributions.

## What is Recorded

### 1. Generation Runs (Phase 2)
Stored in `results/raw/<experiment_id>/metadata.json` and `outputs.jsonl`:
- **Identity:** `experiment_id`, `timestamp`
- **Model:** `model_name`, `revision`, `dtype`, `device`, `quantized`, `load_time_seconds`
- **Generation Parameters:** `seed`, `temperature`, `top_p`, `top_k`, `do_sample`, `max_new_tokens`, `repetition_penalty`
- **Input/Output Metrics:** `prompt`, `input_tokens`, `text`, `output_tokens`, `generation_seconds`, `tokens_per_second`
- **Environment Snapshot:** Python, PyTorch, Transformers, NumPy versions, CUDA availability, GPU name, VRAM, git commit hash, git dirty status.

### 2. Activation Extractions (Phase 3)
Stored in `results/raw/<experiment_id>/activations/manifest.json`:
- `experiment_id`, `timestamp`, `model` metadata
- `extraction_config`: `layers`, `token_position` (pooling strategy), `batch_size`, `storage_dtype`
- `n_samples`, `hidden_dimension`, `layers` list
- Full list of prompt items with dataset `split`, `persona_label`, and input token counts.

### 3. Representation Validation (Phase 5)
Stored in `results/tables/<experiment_id>/persona_validation.csv` and `results/raw/<experiment_id>/persona_validation_metadata.json`:
- Validation-selected layer $L^*$ index and criteria
- Direction stability cosine similarity mean, standard deviation, and 95% CI
- Empirical random-direction null distribution metrics ($K=100$) and empirical $p$-value
- Held-out test performance at $L^*$ (accuracy, ROC-AUC, Cohen's $d$, 95% bootstrap CI).

### 4. Controlled Provenance Experiments (Phase 6)
Stored in `results/tables/<experiment_id>/provenance_pilot.csv` and `results/raw/<experiment_id>/provenance_experiment_metadata.json`:
- Paired differences: $\Delta_{\text{persona}}$, $\Delta_{\text{surface}}$, $\Delta_{\text{neutral}}$
- Paired Wilcoxon signed-rank and paired Student's t-test statistics and $p$-values
- False Discovery Rate (Benjamini-Hochberg) adjusted $p$-values
- Paired unit bootstrap confidence intervals
- Representation vs. behavioral shift correlation coefficients (Pearson $r$, Spearman $\rho$).

## Data Leakage Safeguards

1. Candidate direction vectors $\hat{v} = \frac{\mu_{\text{asst}} - \mu_{\text{alt}}}{\|\mu_{\text{asst}} - \mu_{\text{alt}}\|_2}$ are calculated **exclusively on the training split**.
2. Decision thresholds are calibrated on the training split.
3. Layer selection is performed strictly on the **validation split**.
4. The test split is evaluated only once on the chosen layer/model.
5. All test tasks evaluate semantic topics absent from training data.
6. PCA transformations are fitted exclusively on training activations.
