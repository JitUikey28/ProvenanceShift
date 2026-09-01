# Personas

This directory will store persona definition files if/when explicit persona
specifications are needed for experiments.

## Intended use

Persona definitions describe the behavioral profile that the model is expected
(or hypothesised) to adopt under various experimental conditions.

These are **descriptive research constructs**, not prescriptions.  We do not
assume that a specific persona is "correct" — the research question is whether
and how persona-related representations change under provenance manipulation.

## Format

Persona definitions should be YAML or JSON files with at minimum:
- `persona_id`: unique identifier
- `label`: human-readable label
- `description`: free-text description of the expected behavioral profile
