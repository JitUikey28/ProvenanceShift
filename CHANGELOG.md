# Changelog

All notable changes to ProvenanceShift will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.4.0] — 2026-08-31

### Added (Phase 5 & Phase 6)
- Persona representation validation module (`src/activations/validation.py`) with validation-based layer selection, direction stability resampling, random-direction empirical null distribution, and train-fitted PCA.
- Controlled matched-pair provenance experiment module (`src/experiments/provenance.py`) supporting 4 matched conditions (`baseline`, `provenance_manipulation`, `surface_control`, `neutral_control`), paired deltas, paired Wilcoxon and t-tests, paired task-unit bootstrap, multiple comparisons correction (FDR / Bonferroni), and representation-to-behavior correlation.
- Modular behavioral evaluation engine (`src/evaluation/behavioral.py`) measuring stylistic, lexical, and persona adherence markers.
- 4-condition matched prompt dataset (`data/prompts/provenance_pilot.json`) spanning 32 matched tasks across diverse benign scientific and humanities topics.
- Paraphrase test dataset (`data/prompts/persona_paraphrase.json`).
- CLI scripts: `scripts/run_persona_validation.py` and `scripts/run_provenance_experiment.py`.
- Configurations: `configs/persona_validation.yaml` and `configs/provenance.yaml`.
- Offline unit tests for persona validation, behavioral evaluation, and provenance experiments.

## [0.3.0] — 2026-08-31

### Added (Phase 3 & Phase 4)
- Hidden-state activation extraction engine (`src/activations/extractor.py`) and compressed `.npz` storage (`src/activations/storage.py`).
- Contrastive mean-difference direction construction, Cohen's d, bootstrap CIs, and logistic regression baseline (`src/activations/direction.py`).
- Layer sweep orchestration and plotting pipeline (`src/activations/analysis.py`).
- Controlled benign persona pilot prompt dataset (`data/prompts/persona_pilot.json`).
- CLI scripts: `scripts/extract_activations.py` and `scripts/run_persona_analysis.py`.

## [0.2.0] — 2026-08-31

### Added (Phase 2)
- Causal language model generation pipeline (`src/models/generation.py`).
- Structured result storage with overwrite protection (`src/models/results.py`).
- CLI generation script (`scripts/run_generation.py`).

## [0.1.0] — 2026-08-31

### Added (Phase 1)
- Initial repository structure, configuration system, data schemas, logging, reproducibility utilities, and test harness.
