# Architecture

## Overview

ProvenanceShift is organised as a modular research framework in standard scientific Python (`src/`). There are no heavy web or database frameworks — state is managed via dataclasses, YAML configs, NumPy archives, and JSON manifests.

## Module Dependency Graph

```
configs/
  ├── model.yaml              ─→  src/models/loader.py (ModelConfig)
  ├── persona.yaml            ─→  scripts/run_persona_analysis.py
  ├── persona_validation.yaml ─→  scripts/run_persona_validation.py
  └── provenance.yaml         ─→  scripts/run_provenance_experiment.py

src/
  ├── utils/
  │   ├── logging.py          (standalone structured logging)
  │   ├── reproducibility.py  (seeds, git tracking, environment snapshots)
  │   └── system.py           (hardware & environment diagnosis)
  │
  ├── models/
  │   ├── loader.py           (model & tokenizer loading, device/dtype resolution)
  │   ├── generation.py       (deterministic generation, chat formatting, token metrics)
  │   └── results.py          (overwrite-protected result writer, JSONL output)
  │
  ├── activations/
  │   ├── schemas.py          (ActivationMetadata dataclass)
  │   ├── extractor.py        (hidden state extraction, token pooling, inference_mode)
  │   ├── storage.py          (compressed .npz storage & manifest.json)
  │   ├── direction.py        (mean-difference direction, projection, Cohen's d, bootstrap CI)
  │   ├── analysis.py         (layer sweep orchestration, classifier baseline, table/figure generation)
  │   └── validation.py       (Phase 5 validation, layer selection, stability, random null, PCA)
  │
  ├── evaluation/
  │   ├── schemas.py          (EvaluationResult schema)
  │   └── behavioral.py       (Phase 6 behavioral & stylistic evaluation metrics)
  │
  ├── experiments/
  │   ├── schemas.py          (ExperimentConfig schema)
  │   └── provenance.py       (Phase 6 matched deltas, paired Wilcoxon/t-tests, bootstrap, multiple comparisons)
  │
  └── prompting/
      └── schemas.py          (PromptItem, PromptSet dataclasses)
```

## Result Storage Conventions

```
results/
  raw/<experiment_id>/
    ├── metadata.json                     # Experiment configuration and runtime environment
    ├── outputs.jsonl                     # Generated text and per-prompt metrics
    ├── activations/
    │   ├── activations.npz               # Compressed numerical hidden states per layer
    │   └── manifest.json                 # Activation metadata and dataset item records
    ├── persona_validation_metadata.json   # Phase 5 validation report and stability findings
    └── provenance_experiment_metadata.json# Phase 6 matched delta statistics and association results

  tables/<experiment_id>/
    ├── layer_results.csv                 # Layer-by-layer sweep metrics
    ├── persona_validation.csv            # Phase 5 validation table
    ├── provenance_pilot.csv              # Phase 6 paired hypothesis tests summary table
    └── provenance_paired_deltas.csv      # Phase 6 individual paired deltas per task

  figures/<experiment_id>/
    ├── layer_accuracy.png
    ├── layer_roc_auc.png
    ├── layer_cohens_d.png
    ├── projection_distribution.png
    ├── direction_stability.png           # Phase 5 bootstrap cosine similarity distribution
    ├── random_direction_null.png         # Phase 5 learned vs random null distribution
    ├── train_fitted_pca.png              # Phase 5 train-fitted 2D projection
    ├── provenance_delta_distribution.png # Phase 6 paired delta boxplots vs null
    └── representation_vs_behavior.png    # Phase 6 representation shift vs behavioral shift scatter
```
