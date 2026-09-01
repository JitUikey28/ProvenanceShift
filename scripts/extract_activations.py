#!/usr/bin/env python3
"""
Extract hidden-state representations from a model on a prompt dataset.

Usage:
    python scripts/extract_activations.py \\
        --config configs/model.yaml \\
        --prompt-file data/prompts/persona_pilot.json \\
        --experiment-id ACT-001 \\
        --layers all \\
        --pooling last_token \\
        --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract activations from a language model for a prompt dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model.yaml",
        help="Path to model config YAML (default: configs/model.yaml).",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="data/prompts/persona_pilot.json",
        help="Path to prompt dataset JSON file (default: data/prompts/persona_pilot.json).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Unique experiment identifier (auto-generated if omitted).",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layer selection: 'all', 'first_middle_last', 'spaced', or comma-separated ints (e.g. '0,4,8,12').",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["last_token", "mean_pool"],
        default="last_token",
        help="Token pooling strategy (default: last_token).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for extraction (default: 4).",
    )
    parser.add_argument(
        "--storage-dtype",
        type=str,
        choices=["float32", "float16"],
        default="float32",
        help="Dtype for stored NumPy arrays (default: float32).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/raw",
        help="Base directory to save results (default: results/raw).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.activations.extractor import ExtractionConfig, extract_hidden_states
    from src.activations.storage import save_activations
    from src.models.loader import load_model, load_model_config
    from src.models.results import generate_experiment_id
    from src.utils.logging import get_logger
    from src.utils.reproducibility import capture_environment, set_seed

    experiment_id = args.experiment_id or generate_experiment_id(prefix="ACT")
    logger = get_logger(experiment_id=experiment_id)

    logger.info("=" * 60)
    logger.info("ACTIVATION EXTRACTION RUN")
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info("=" * 60)

    # 1. Parse layers argument
    if args.layers in {"all", "first_middle_last", "spaced"}:
        layer_spec = args.layers
    else:
        try:
            layer_spec = [int(x.strip()) for x in args.layers.split(",")]
        except ValueError:
            logger.error(f"Invalid layer spec: {args.layers}")
            sys.exit(1)

    # 2. Load model config & set seed
    model_config = load_model_config(Path(args.config))
    set_seed(model_config.seed)

    # 3. Load prompt dataset
    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        logger.error(f"Prompt file not found: {prompt_path}")
        sys.exit(1)
    with open(prompt_path, "r", encoding="utf-8") as fh:
        prompt_items = json.load(fh)
    logger.info(f"Loaded {len(prompt_items)} prompt items from {prompt_path}")

    # Extract messages or prompt strings
    prompts_to_feed = [item.get("messages", item.get("base_task", "")) for item in prompt_items]

    # 4. Load model
    logger.info(f"Loading model '{model_config.model_name}'...")
    bundle = load_model(model_config)

    # 5. Extraction config
    extract_cfg = ExtractionConfig(
        layers=layer_spec,
        token_position=args.pooling,
        batch_size=args.batch_size,
        storage_dtype=args.storage_dtype,
    )

    # 6. Extract activations
    activations_by_layer, sample_meta = extract_hidden_states(
        bundle=bundle,
        prompts=prompts_to_feed,
        config=extract_cfg,
    )

    # 7. Merge token metadata into prompt items
    for i, meta in enumerate(sample_meta):
        if i < len(prompt_items):
            prompt_items[i]["token_metadata"] = meta

    # 8. Save activations and manifest
    save_dir = Path(args.output_dir) / experiment_id / "activations"
    env_snap = capture_environment(seed=model_config.seed, repo_dir=_project_root)

    save_activations(
        output_dir=save_dir,
        experiment_id=experiment_id,
        activations_by_layer=activations_by_layer,
        prompt_items=prompt_items,
        extraction_config={
            "layers": layer_spec,
            "token_position": args.pooling,
            "batch_size": args.batch_size,
            "storage_dtype": args.storage_dtype,
        },
        model_metadata=bundle.metadata,
        extra_metadata={"environment": env_snap.to_dict()},
    )

    logger.info("=" * 60)
    logger.info(f"EXTRACTION COMPLETE: {save_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
