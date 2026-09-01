"""
Activation storage — lightweight and reproducible persistence for extracted representations.

Separates:
    - Numerical representations (stored as compressed NumPy .npz archive)
    - Metadata & provenance manifest (stored as human-readable JSON)

Layout:
    results/raw/<experiment_id>/activations/
        ├── activations.npz   # Compressed arrays: layer_0, layer_1, ..., prompt_ids
        └── manifest.json     # Metadata for model, config, dataset splits, and items
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logging import get_logger

logger = get_logger()


def save_activations(
    output_dir: Union[str, Path],
    experiment_id: str,
    activations_by_layer: Dict[int, np.ndarray],
    prompt_items: List[Dict[str, Any]],
    extraction_config: Dict[str, Any],
    model_metadata: Dict[str, Any],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save extracted activations and metadata manifest to disk.

    Parameters
    ----------
    output_dir:
        Directory under which to save activations (e.g. ``results/raw/<exp_id>/activations``).
    experiment_id:
        Unique experiment run identifier.
    activations_by_layer:
        Dictionary mapping layer index (int) to 2D NumPy array ``(N, hidden_dim)``.
    prompt_items:
        List of prompt dictionaries (including prompt_id, condition, persona_label, split, etc.).
    extraction_config:
        Dictionary capturing extraction settings (layers, pooling, dtype, batch_size).
    model_metadata:
        Dictionary describing the model identity, revision, dtype, device.
    extra_metadata:
        Optional additional metadata (e.g. git commit, timestamp).

    Returns
    -------
    Path
        Path to the saved activations directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    prompt_ids = [item.get("prompt_id", f"p_{i}") for i, item in enumerate(prompt_items)]
    n_samples = len(prompt_ids)

    # Validate shapes
    npz_dict: Dict[str, Any] = {"prompt_ids": np.array(prompt_ids, dtype=object)}
    hidden_dim = 0
    dtype_str = ""

    for l_idx, arr in activations_by_layer.items():
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"Layer {l_idx} activation array length ({arr.shape[0]}) "
                f"does not match prompt items count ({n_samples})."
            )
        npz_dict[f"layer_{l_idx}"] = arr
        hidden_dim = arr.shape[1]
        dtype_str = str(arr.dtype)

    # Save compressed numerical arrays
    npz_file = out_path / "activations.npz"
    np.savez_compressed(npz_file, **npz_dict)
    logger.info(f"Saved {len(activations_by_layer)} layers to {npz_file}")

    # Build manifest
    manifest: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_metadata,
        "extraction_config": extraction_config,
        "n_samples": n_samples,
        "hidden_dimension": hidden_dim,
        "dtype": dtype_str,
        "layers": sorted(list(activations_by_layer.keys())),
        "prompts": prompt_items,
    }
    if extra_metadata:
        manifest["extra"] = extra_metadata

    manifest_file = out_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    logger.info(f"Saved manifest to {manifest_file}")

    return out_path


def load_activations(
    directory: Union[str, Path],
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """Load activations and manifest from a saved activations directory.

    Parameters
    ----------
    directory:
        Path to the directory containing ``activations.npz`` and ``manifest.json``.

    Returns
    -------
    Tuple[Dict[int, np.ndarray], Dict[str, Any]]
        - Dictionary mapping ``layer_index -> np.ndarray``.
        - Manifest dictionary with all metadata.

    Raises
    ------
    FileNotFoundError
        If ``activations.npz`` or ``manifest.json`` is missing.
    """
    dir_path = Path(directory)
    npz_file = dir_path / "activations.npz"
    manifest_file = dir_path / "manifest.json"

    if not npz_file.exists():
        raise FileNotFoundError(f"Missing activation array file: {npz_file}")
    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    activations_by_layer: Dict[int, np.ndarray] = {}
    with np.load(npz_file, allow_pickle=True) as data:
        for key in data.files:
            if key.startswith("layer_"):
                l_idx = int(key.replace("layer_", ""))
                activations_by_layer[l_idx] = data[key]

    return activations_by_layer, manifest
