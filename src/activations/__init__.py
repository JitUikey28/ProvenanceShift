"""Activation extraction, representation storage, candidate directions, and persona analysis."""

from src.activations.direction import (
    bootstrap_confidence_interval,
    compute_cohens_d,
    compute_mean_difference_direction,
    compute_projection_metrics,
    evaluate_classifier,
    generate_random_direction,
    project_representations,
    shuffle_labels,
    train_linear_classifier,
)
from src.activations.extractor import (
    ExtractionConfig,
    extract_hidden_states,
    pool_hidden_states,
    resolve_layer_indices,
)
from src.activations.schemas import ActivationMetadata
from src.activations.storage import load_activations, save_activations

__all__ = [
    "ActivationMetadata",
    "ExtractionConfig",
    "extract_hidden_states",
    "pool_hidden_states",
    "resolve_layer_indices",
    "save_activations",
    "load_activations",
    "compute_mean_difference_direction",
    "project_representations",
    "compute_cohens_d",
    "compute_projection_metrics",
    "bootstrap_confidence_interval",
    "train_linear_classifier",
    "evaluate_classifier",
    "generate_random_direction",
    "shuffle_labels",
]
