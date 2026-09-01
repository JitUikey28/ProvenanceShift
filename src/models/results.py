"""
Result storage — structured writing of generation outputs and metadata.

Design principles:
    - Never silently overwrite previous experiment results.
    - Each experiment gets its own directory under ``results/raw/<id>/``.
    - Metadata is written as ``metadata.json``; outputs as ``outputs.jsonl``.
    - Experiment IDs are either user-supplied or auto-generated from timestamps.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Experiment ID generation
# ---------------------------------------------------------------------------

def generate_experiment_id(prefix: str = "GEN") -> str:
    """Generate a timestamp-based experiment ID.

    Format: ``{prefix}-YYYYMMDD-HHMMSS``

    Parameters
    ----------
    prefix:
        Short prefix for the ID (default ``"GEN"``).

    Returns
    -------
    str
        e.g. ``"GEN-20260831-111520"``
    """
    now = datetime.now(timezone.utc)
    return f"{prefix}-{now.strftime('%Y%m%d-%H%M%S')}"


def validate_experiment_id(experiment_id: str) -> None:
    """Validate that an experiment ID is non-empty and filesystem-safe.

    Parameters
    ----------
    experiment_id:
        The ID to validate.

    Raises
    ------
    ValueError
        If the ID is empty or contains unsafe characters.
    """
    if not experiment_id or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty string.")

    unsafe_chars = set('<>:"/\\|?*')
    found = unsafe_chars.intersection(experiment_id)
    if found:
        raise ValueError(
            f"experiment_id contains unsafe characters: {found}. "
            f"Use only alphanumeric characters, hyphens, and underscores."
        )


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

def build_result_record(
    experiment_id: str,
    timestamp: str,
    model_metadata: Dict[str, Any],
    generation_result: Any,
) -> Dict[str, Any]:
    """Build a structured result record for storage.

    Parameters
    ----------
    experiment_id:
        The experiment identifier.
    timestamp:
        ISO-8601 UTC timestamp.
    model_metadata:
        Model metadata dict from ``ModelBundle.metadata``.
    generation_result:
        A ``GenerationResult`` instance (or its ``.to_dict()`` output).

    Returns
    -------
    Dict[str, Any]
        Structured result matching the project schema.
    """
    # Accept either a GenerationResult or a dict
    if hasattr(generation_result, "to_dict"):
        gen_dict = generation_result.to_dict()
    else:
        gen_dict = generation_result

    return {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "model": {
            "name": model_metadata.get("name"),
            "revision": model_metadata.get("revision"),
            "dtype": model_metadata.get("dtype"),
            "device": model_metadata.get("device"),
            "quantized": model_metadata.get("quantized", False),
        },
        "generation": {
            "seed": gen_dict.get("seed"),
            "temperature": gen_dict.get("temperature"),
            "top_p": gen_dict.get("top_p"),
            "top_k": gen_dict.get("top_k"),
            "do_sample": gen_dict.get("do_sample"),
            "max_new_tokens": gen_dict.get("max_new_tokens"),
            "repetition_penalty": gen_dict.get("repetition_penalty"),
        },
        "input": {
            "prompt": gen_dict.get("prompt"),
            "input_tokens": gen_dict.get("input_tokens"),
        },
        "output": {
            "text": gen_dict.get("text"),
            "output_tokens": gen_dict.get("output_tokens"),
        },
        "timing": {
            "generation_seconds": gen_dict.get("generation_seconds"),
            "tokens_per_second": gen_dict.get("tokens_per_second"),
        },
    }


# ---------------------------------------------------------------------------
# Result writer
# ---------------------------------------------------------------------------

class ResultWriter:
    """Write experiment results to disk with overwrite protection.

    Creates a directory ``{base_dir}/{experiment_id}/`` and writes:
    - ``metadata.json``: experiment-level metadata
    - ``outputs.jsonl``: one JSON object per generation (appended)

    Parameters
    ----------
    base_dir:
        Root directory for results (default ``results/raw``).
    experiment_id:
        Unique experiment identifier.
    overwrite:
        If ``True``, allow writing to an existing experiment directory.
        Default ``False`` — raises ``FileExistsError`` if the directory
        already contains results.
    """

    def __init__(
        self,
        base_dir: str = "results/raw",
        experiment_id: str = "",
        overwrite: bool = False,
    ) -> None:
        validate_experiment_id(experiment_id)

        self.experiment_id = experiment_id
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_id
        self.overwrite = overwrite

        # Check for existing results
        metadata_path = self.experiment_dir / "metadata.json"
        outputs_path = self.experiment_dir / "outputs.jsonl"

        if not overwrite and (metadata_path.exists() or outputs_path.exists()):
            raise FileExistsError(
                f"Experiment '{experiment_id}' already has results in "
                f"'{self.experiment_dir}'. Use --overwrite to replace them. "
                f"CAUTION: This will destroy previous research data."
            )

        # Create the directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Result directory: {self.experiment_dir}")

    def write_metadata(self, metadata: Dict[str, Any]) -> Path:
        """Write experiment metadata to ``metadata.json``.

        Parameters
        ----------
        metadata:
            Dictionary of experiment metadata.

        Returns
        -------
        Path
            Path to the written file.
        """
        path = self.experiment_dir / "metadata.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        logger.info(f"Metadata written: {path}")
        return path

    def append_output(self, record: Dict[str, Any]) -> Path:
        """Append a single generation result to ``outputs.jsonl``.

        Parameters
        ----------
        record:
            A structured result record (from ``build_result_record``).

        Returns
        -------
        Path
            Path to the outputs file.
        """
        path = self.experiment_dir / "outputs.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return path

    def write_outputs(self, records: List[Dict[str, Any]]) -> Path:
        """Write multiple generation results to ``outputs.jsonl``.

        Parameters
        ----------
        records:
            List of structured result records.

        Returns
        -------
        Path
            Path to the outputs file.
        """
        path = self.experiment_dir / "outputs.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, default=str) + "\n")
        logger.info(f"Wrote {len(records)} outputs to: {path}")
        return path
