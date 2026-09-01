"""
Activation extraction — extract hidden-state representations from causal LMs.

Design principles:
    - Layer-flexible: supports all layers, selected layers, or spaced sweeps.
    - Token-position aware: last-token extraction (excluding padding) and
      attention-mask-weighted mean pooling.
    - Memory-efficient: extracts in small batches, detaches immediately,
      moves to CPU, and stores as lightweight NumPy arrays.
    - Native HF interface: uses ``output_hidden_states=True``.
    - Model-independent: uses tokenizer's chat template when available.

Indexing convention:
    Hugging Face models return ``hidden_states`` of length ``num_layers + 1``:
      - Layer 0: Embedding layer output.
      - Layer 1..L: Output of transformer block 1 through L.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.models.generation import format_prompt
from src.models.loader import ModelBundle
from src.utils.logging import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Extraction Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExtractionConfig:
    """Configuration for hidden-state extraction."""

    layers: Union[str, Sequence[int]] = "all"
    token_position: str = "last_token"  # "last_token" | "mean_pool"
    batch_size: int = 4
    storage_dtype: str = "float32"  # "float32" | "float16"

    def __post_init__(self) -> None:
        valid_positions = {"last_token", "mean_pool"}
        if self.token_position not in valid_positions:
            raise ValueError(
                f"Unsupported token_position '{self.token_position}'. "
                f"Must be one of {valid_positions}"
            )
        valid_dtypes = {"float32", "float16"}
        if self.storage_dtype not in valid_dtypes:
            raise ValueError(
                f"Unsupported storage_dtype '{self.storage_dtype}'. "
                f"Must be one of {valid_dtypes}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


def resolve_layer_indices(
    layer_spec: Union[str, Sequence[int]],
    total_layers: int,
) -> List[int]:
    """Resolve layer specification into concrete layer indices.

    Parameters
    ----------
    layer_spec:
        "all", "first_middle_last", "spaced", or an explicit list of ints.
    total_layers:
        Total number of layers in hidden_states tuple (L + 1, including embeddings).

    Returns
    -------
    List[int]
        Sorted list of unique layer indices.
    """
    max_idx = total_layers - 1  # e.g., for 32 transformer blocks, total_layers = 33 (0..32)

    if layer_spec == "all":
        return list(range(total_layers))
    elif layer_spec == "first_middle_last":
        return sorted(list({0, total_layers // 2, max_idx}))
    elif layer_spec == "spaced":
        step = max(1, total_layers // 8)
        indices = list(range(0, total_layers, step))
        if max_idx not in indices:
            indices.append(max_idx)
        return sorted(list(set(indices)))
    elif isinstance(layer_spec, (list, tuple)):
        resolved = []
        for idx in layer_spec:
            if not isinstance(idx, int):
                raise ValueError(f"Layer index must be an integer, got {idx} ({type(idx)})")
            if idx < 0 or idx > max_idx:
                raise ValueError(
                    f"Layer index {idx} out of range for model with {total_layers} "
                    f"hidden states (valid range: 0..{max_idx})."
                )
            resolved.append(idx)
        return sorted(list(set(resolved)))
    else:
        raise ValueError(
            f"Unsupported layer_spec: {layer_spec}. "
            f"Use 'all', 'first_middle_last', 'spaced', or a list of integers."
        )


# ---------------------------------------------------------------------------
# Token Pooling
# ---------------------------------------------------------------------------

def pool_hidden_states(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    method: str = "last_token",
) -> torch.Tensor:
    """Pool a sequence of hidden states into a single vector per example.

    Parameters
    ----------
    hidden_state:
        Tensor of shape ``(batch_size, seq_len, hidden_dim)``.
    attention_mask:
        Tensor of shape ``(batch_size, seq_len)`` where 1 is token, 0 is padding.
    method:
        "last_token" or "mean_pool".

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(batch_size, hidden_dim)``.
    """
    batch_size, seq_len, hidden_dim = hidden_state.shape

    if method == "last_token":
        # Find index of last non-pad token for each sequence in batch
        # attention_mask sum gives non-pad length, so last index is sum - 1
        seq_lengths = attention_mask.sum(dim=-1).long()  # (batch_size,)
        last_indices = (seq_lengths - 1).clamp(min=0)  # (batch_size,)

        # Gather the vectors at last_indices
        batch_idx = torch.arange(batch_size, device=hidden_state.device)
        pooled = hidden_state[batch_idx, last_indices, :]  # (batch_size, hidden_dim)
        return pooled

    elif method == "mean_pool":
        # Expand mask for broadcasting: (batch_size, seq_len, 1)
        mask_expanded = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1)  # (batch_size, hidden_dim)
        sum_mask = attention_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(hidden_state.dtype)
        pooled = sum_embeddings / sum_mask  # (batch_size, hidden_dim)
        return pooled

    else:
        raise ValueError(f"Unknown pooling method '{method}'.")


# ---------------------------------------------------------------------------
# Extraction Engine
# ---------------------------------------------------------------------------

def extract_hidden_states(
    bundle: ModelBundle,
    prompts: Sequence[Union[str, List[Dict[str, str]]]],
    config: ExtractionConfig,
) -> Tuple[Dict[int, np.ndarray], List[Dict[str, Any]]]:
    """Extract hidden states from the model for a list of prompts.

    Parameters
    ----------
    bundle:
        A ``ModelBundle`` containing model and tokenizer.
    prompts:
        List of prompt strings or chat message lists.
    config:
        Extraction configuration (layers, pooling, batch size, dtype).

    Returns
    -------
    Tuple[Dict[int, np.ndarray], List[Dict[str, Any]]]
        - Dictionary mapping ``layer_index -> np.ndarray`` of shape ``(N, hidden_dim)``.
        - List of per-example token metadata dicts (length, token count, etc.).
    """
    model = bundle.model
    tokenizer = bundle.tokenizer

    # Determine device from model parameters
    device = next(model.parameters()).device

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # For extraction, left or right padding is fine because we use attention_mask explicitly,
    # but right padding is standard with causal attention when attention_mask is passed.
    tokenizer.padding_side = "right"

    # Pre-format all prompts using tokenizer chat template where applicable
    formatted_prompts = [format_prompt(tokenizer, p) for p in prompts]
    n_prompts = len(formatted_prompts)

    logger.info(
        f"Extracting hidden states for {n_prompts} prompts "
        f"(batch_size={config.batch_size}, pooling={config.token_position})..."
    )

    np_dtype = np.float32 if config.storage_dtype == "float32" else np.float16

    # Accumulators
    layer_tensors: Dict[int, List[np.ndarray]] = {}
    sample_metadata: List[Dict[str, Any]] = []
    layer_indices: Optional[List[int]] = None

    start_time = time.perf_counter()

    for start_idx in range(0, n_prompts, config.batch_size):
        end_idx = min(start_idx + config.batch_size, n_prompts)
        batch_prompts = formatted_prompts[start_idx:end_idx]

        # Tokenize batch
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        batch_lengths = attention_mask.sum(dim=-1).cpu().tolist()

        # Run model forward pass under inference mode with hidden states enabled
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states  # Tuple of (L + 1) tensors
        total_layers = len(hidden_states)

        # Resolve layer indices on first batch
        if layer_indices is None:
            layer_indices = resolve_layer_indices(config.layers, total_layers)
            for l_idx in layer_indices:
                layer_tensors[l_idx] = []
            logger.info(
                f"Model has {total_layers} hidden state levels (layer 0=embedding, "
                f"1..{total_layers-1}=transformer blocks). "
                f"Extracting {len(layer_indices)} layers: {layer_indices}"
            )

        # Pool and extract requested layers
        for l_idx in layer_indices:
            hs_layer = hidden_states[l_idx]  # (B, S, H)
            pooled = pool_hidden_states(
                hs_layer,
                attention_mask,
                method=config.token_position,
            )
            # Move immediately to CPU NumPy array
            arr = pooled.detach().to(torch.float32).cpu().numpy().astype(np_dtype)
            layer_tensors[l_idx].append(arr)

        # Record metadata for this batch
        for i, length in enumerate(batch_lengths):
            sample_metadata.append({
                "prompt_index": start_idx + i,
                "input_token_count": int(length),
            })

    # Concatenate batches per layer
    result_activations: Dict[int, np.ndarray] = {}
    for l_idx, arr_list in layer_tensors.items():
        result_activations[l_idx] = np.concatenate(arr_list, axis=0)

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Extraction complete in {elapsed:.2f}s across {len(result_activations)} layers. "
        f"Representation shape: {next(iter(result_activations.values())).shape}"
    )

    return result_activations, sample_metadata
