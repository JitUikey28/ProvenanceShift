"""
Offline unit tests for activation extraction and storage (Phase 3).

Tests verify configuration, layer indexing, token pooling with attention masks,
storage round-trips, and mock extraction without GPU or model downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.activations.extractor import (
    ExtractionConfig,
    extract_hidden_states,
    pool_hidden_states,
    resolve_layer_indices,
)
from src.activations.storage import load_activations, save_activations
from src.models.loader import ModelBundle


# =====================================================================
# ExtractionConfig
# =====================================================================

class TestExtractionConfig:

    def test_default_config(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.layers == "all"
        assert cfg.token_position == "last_token"
        assert cfg.batch_size == 4
        assert cfg.storage_dtype == "float32"

    def test_invalid_token_position(self) -> None:
        with pytest.raises(ValueError, match="Unsupported token_position"):
            ExtractionConfig(token_position="invalid_pos")

    def test_invalid_storage_dtype(self) -> None:
        with pytest.raises(ValueError, match="Unsupported storage_dtype"):
            ExtractionConfig(storage_dtype="float64")

    def test_invalid_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            ExtractionConfig(batch_size=0)


# =====================================================================
# Layer Resolution
# =====================================================================

class TestResolveLayerIndices:

    def test_all_layers(self) -> None:
        indices = resolve_layer_indices("all", total_layers=5)
        assert indices == [0, 1, 2, 3, 4]

    def test_first_middle_last(self) -> None:
        indices = resolve_layer_indices("first_middle_last", total_layers=9)
        assert indices == [0, 4, 8]

    def test_spaced_layers(self) -> None:
        indices = resolve_layer_indices("spaced", total_layers=17)
        assert indices[0] == 0
        assert indices[-1] == 16
        assert len(indices) > 2

    def test_explicit_list(self) -> None:
        indices = resolve_layer_indices([0, 2, 4], total_layers=5)
        assert indices == [0, 2, 4]

    def test_out_of_bounds_layer_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            resolve_layer_indices([0, 10], total_layers=5)

    def test_negative_layer_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            resolve_layer_indices([-1, 2], total_layers=5)


# =====================================================================
# Token Pooling
# =====================================================================

class TestTokenPooling:

    def test_last_token_pooling_with_padding(self) -> None:
        """Verify that last_token ignores right-padding and picks the true last token."""
        # Batch of 2 sequences: seq 0 has 3 tokens, seq 1 has 2 tokens (padded to 4)
        # Shape: (2, 4, 2)
        hidden = torch.tensor([
            [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1], [0.0, 0.0]],  # seq 0: valid tokens at 0, 1, 2
            [[4.0, 4.1], [5.0, 5.1], [0.0, 0.0], [0.0, 0.0]],  # seq 1: valid tokens at 0, 1
        ])
        mask = torch.tensor([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ])

        pooled = pool_hidden_states(hidden, mask, method="last_token")
        assert pooled.shape == (2, 2)
        # Seq 0 last token should be at index 2: [3.0, 3.1]
        torch.testing.assert_close(pooled[0], torch.tensor([3.0, 3.1]))
        # Seq 1 last token should be at index 1: [5.0, 5.1]
        torch.testing.assert_close(pooled[1], torch.tensor([5.0, 5.1]))

    def test_mean_pooling_with_padding(self) -> None:
        """Verify that mean_pool averages non-pad tokens only."""
        hidden = torch.tensor([
            [[2.0, 4.0], [4.0, 6.0], [0.0, 0.0]],  # seq 0: 2 tokens, sum=[6, 10], mean=[3, 5]
            [[10.0, 20.0], [0.0, 0.0], [0.0, 0.0]], # seq 1: 1 token, mean=[10, 20]
        ])
        mask = torch.tensor([
            [1, 1, 0],
            [1, 0, 0],
        ])

        pooled = pool_hidden_states(hidden, mask, method="mean_pool")
        assert pooled.shape == (2, 2)
        torch.testing.assert_close(pooled[0], torch.tensor([3.0, 5.0]))
        torch.testing.assert_close(pooled[1], torch.tensor([10.0, 20.0]))


# =====================================================================
# Activation Storage
# =====================================================================

class TestActivationStorage:

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        exp_id = "test-act-001"
        save_dir = tmp_path / exp_id

        # Synthetic activations: 2 layers, 3 samples, dim=4
        activations = {
            0: np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], dtype=np.float32),
            1: np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], dtype=np.float32),
        }
        prompt_items = [
            {"prompt_id": "p0", "persona_label": "assistant", "split": "train"},
            {"prompt_id": "p1", "persona_label": "alternative", "split": "train"},
            {"prompt_id": "p2", "persona_label": "assistant", "split": "test"},
        ]
        extract_cfg = {"layers": [0, 1], "token_position": "last_token"}
        model_meta = {"name": "mock-model", "dtype": "float32", "device": "cpu"}

        saved_path = save_activations(
            output_dir=save_dir,
            experiment_id=exp_id,
            activations_by_layer=activations,
            prompt_items=prompt_items,
            extraction_config=extract_cfg,
            model_metadata=model_meta,
        )

        assert (saved_path / "activations.npz").exists()
        assert (saved_path / "manifest.json").exists()

        # Load back
        loaded_acts, manifest = load_activations(saved_path)

        assert set(loaded_acts.keys()) == {0, 1}
        np.testing.assert_array_almost_equal(loaded_acts[0], activations[0])
        np.testing.assert_array_almost_equal(loaded_acts[1], activations[1])

        assert manifest["experiment_id"] == exp_id
        assert manifest["n_samples"] == 3
        assert manifest["hidden_dimension"] == 4
        assert len(manifest["prompts"]) == 3


# =====================================================================
# Mock Model Hidden-State Extraction
# =====================================================================

class TestExtractHiddenStatesMock:

    def test_mock_extraction_pipeline(self) -> None:
        """Test extraction pipeline end-to-end using a mock model and tokenizer."""
        # Create mock tokenizer
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<eos>"
        tokenizer.chat_template = None
        tokenizer.padding_side = "right"

        # Mock tokenization returning dict
        def mock_tokenize(prompts, **kwargs):
            batch_size = len(prompts)
            seq_len = 5
            return {
                "input_ids": torch.ones((batch_size, seq_len), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
            }
        tokenizer.side_effect = mock_tokenize

        # Create mock model returning hidden states
        hidden_dim = 8
        seq_len = 5

        class MockOutputs:
            def __init__(self, batch_size):
                # Tuple of 3 hidden states (layers 0, 1, 2)
                self.hidden_states = (
                    torch.randn((batch_size, seq_len, hidden_dim)),
                    torch.randn((batch_size, seq_len, hidden_dim)),
                    torch.randn((batch_size, seq_len, hidden_dim)),
                )

        model = MagicMock()
        model.parameters.return_value = iter([torch.zeros(1, device="cpu", dtype=torch.float32)])

        def mock_forward(**kwargs):
            b_size = kwargs["input_ids"].shape[0]
            return MockOutputs(b_size)
        model.side_effect = mock_forward

        bundle = ModelBundle(model=model, tokenizer=tokenizer, metadata={"name": "mock"})

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        config = ExtractionConfig(layers=[0, 2], token_position="last_token", batch_size=2)

        acts, meta = extract_hidden_states(bundle, prompts, config)

        assert set(acts.keys()) == {0, 2}
        assert acts[0].shape == (3, hidden_dim)
        assert acts[2].shape == (3, hidden_dim)
        assert len(meta) == 3
