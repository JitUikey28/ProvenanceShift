"""
Experiment metadata schema.

Defines the configuration structure for a single experiment run.
Every completed experiment produces an ``ExperimentConfig`` record
that, together with the environment snapshot, contains everything
needed to reproduce the run.

This schema is intentionally extensible via the ``extra`` field.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentConfig:
    """Full metadata for one experiment run.

    Required fields will raise on ``validate()`` if missing.

    Attributes
    ----------
    experiment_id:
        Unique identifier (e.g. ``"exp_001_baseline"``).  **Required.**
    experiment_name:
        Human-readable name.  **Required.**
    description:
        Free-text description of what this experiment tests.
    model:
        Model identifier (Hugging Face name).
    model_revision:
        Model revision / commit hash.
    seed:
        Random seed for reproducibility.
    temperature:
        Sampling temperature.
    top_p:
        Nucleus sampling threshold.
    max_new_tokens:
        Maximum number of tokens to generate.
    condition:
        Experimental condition label.
    prompt_set:
        Identifier or path to the prompt set used.
    timestamp:
        ISO-8601 UTC timestamp of the run.
    git_commit:
        Repository commit hash at run time.
    output_dir:
        Where raw results are written.
    notes:
        Free-text notes.
    extra:
        Arbitrary additional metadata for future extensions.
    """

    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    description: str = ""
    model: Optional[str] = None
    model_revision: Optional[str] = None
    seed: int = 42
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 256
    condition: Optional[str] = None
    prompt_set: Optional[str] = None
    timestamp: str = ""
    git_commit: Optional[str] = None
    output_dir: str = "results/raw"
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Check that all required fields are present.

        Raises
        ------
        ValueError
            If any required field is missing or empty.
        """
        errors: List[str] = []
        if not self.experiment_id:
            errors.append("experiment_id is required.")
        if not self.experiment_name:
            errors.append("experiment_name is required.")
        if self.seed < 0:
            errors.append("seed must be non-negative.")
        if not (0.0 <= self.temperature):
            errors.append("temperature must be non-negative.")
        if not (0.0 < self.top_p <= 1.0):
            errors.append("top_p must be in (0, 1].")
        if self.max_new_tokens < 1:
            errors.append("max_new_tokens must be >= 1.")

        if errors:
            raise ValueError(
                "Experiment configuration validation failed:\n  - "
                + "\n  - ".join(errors)
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def stamp(self) -> "ExperimentConfig":
        """Set the timestamp to now (UTC) if not already set."""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: Path) -> None:
        """Write this config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)

    def save_json(self, path: Path) -> None:
        """Write this config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load an experiment config from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Experiment config not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Construct from a plain dictionary (ignoring unknown keys)."""
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
