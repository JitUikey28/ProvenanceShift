#!/usr/bin/env python3
"""
Run a single generation (or batch) through a configured causal language model.

This is the primary entry point for Phase 2 of ProvenanceShift.  It loads a
model from a YAML config, generates text for one or more prompts, and writes
structured results to ``results/raw/<experiment_id>/``.

Usage examples
--------------
Single prompt::

    python scripts/run_generation.py \\
        --config configs/model.yaml \\
        --prompt "Explain what machine learning is."

With explicit experiment ID and seed::

    python scripts/run_generation.py \\
        --config configs/model.yaml \\
        --prompt "Explain what machine learning is." \\
        --experiment-id GEN-001 \\
        --seed 42

Batch from a JSON file::

    python scripts/run_generation.py \\
        --config configs/model.yaml \\
        --prompt-file data/prompts/example_prompts.json \\
        --experiment-id GEN-BATCH-001
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on sys.path so that `src` is importable.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run generation through a configured causal language model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model YAML config (e.g. configs/model.yaml).",
    )

    # Prompt sources (mutually exclusive)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        type=str,
        help="Single prompt string.",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=str,
        help="Path to a JSON file containing a list of prompt strings.",
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed from the config.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens from the config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing experiment directory.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # --- Imports (deferred to keep startup fast) ---
    from src.models.loader import load_model_config, load_model
    from src.models.generation import GenerationConfig, generate, generate_batch
    from src.models.results import (
        ResultWriter,
        build_result_record,
        generate_experiment_id,
    )
    from src.utils.reproducibility import set_seed, capture_environment
    from src.utils.logging import get_logger

    # --- Load config ---
    config = load_model_config(Path(args.config))

    # --- Experiment ID ---
    experiment_id = args.experiment_id or generate_experiment_id()
    logger = get_logger(experiment_id=experiment_id)

    logger.info(f"Experiment ID: {experiment_id}")
    logger.info("=" * 60)

    # --- Report model before download ---
    logger.info(f"Model:                {config.model_name}")
    logger.info(f"Revision:             {config.revision or 'latest'}")
    logger.info(f"Device:               {config.device}")
    logger.info(f"Dtype:                {config.dtype}")
    logger.info(f"Trust remote code:    {config.trust_remote_code}")
    logger.info(f"Quantization 4-bit:   {config.load_in_4bit}")
    logger.info(f"Quantization 8-bit:   {config.load_in_8bit}")
    logger.info(f"CPU fallback:         {config.allow_cpu_fallback}")
    logger.info("=" * 60)

    # --- Seed ---
    seed = args.seed if args.seed is not None else config.seed
    logger.info(f"Setting seed: {seed}")
    set_seed(seed)

    # --- Load model ---
    logger.info("Loading model... (this may download on first run)")
    bundle = load_model(config)

    logger.info("Model metadata:")
    for k, v in bundle.metadata.items():
        logger.info(f"  {k}: {v}")

    # --- Build generation config ---
    gen_dict = dict(config.generation)
    gen_dict["seed"] = seed
    if args.max_new_tokens is not None:
        gen_dict["max_new_tokens"] = args.max_new_tokens
    gen_config = GenerationConfig.from_dict(gen_dict)

    logger.info(f"Generation config: {gen_config.to_dict()}")

    # --- Load prompts ---
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompt_file = Path(args.prompt_file)
        if not prompt_file.exists():
            logger.error(f"Prompt file not found: {prompt_file}")
            sys.exit(1)
        with open(prompt_file, "r", encoding="utf-8") as fh:
            prompts = json.load(fh)
        if not isinstance(prompts, list):
            logger.error("Prompt file must contain a JSON list of strings.")
            sys.exit(1)
        logger.info(f"Loaded {len(prompts)} prompts from {prompt_file}")

    # --- Set up result writer ---
    writer = ResultWriter(
        base_dir="results/raw",
        experiment_id=experiment_id,
        overwrite=args.overwrite,
    )

    # --- Capture environment ---
    env_snapshot = capture_environment(seed=seed, repo_dir=_project_root)
    timestamp = datetime.now(timezone.utc).isoformat()

    # --- Write metadata ---
    metadata = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "model": bundle.metadata,
        "generation_config": gen_config.to_dict(),
        "prompts_count": len(prompts),
        "environment": env_snapshot.to_dict(),
    }
    writer.write_metadata(metadata)

    # --- Generate ---
    logger.info(f"Generating responses for {len(prompts)} prompt(s)...")

    if len(prompts) == 1:
        result = generate(bundle, prompts[0], gen_config)
        record = build_result_record(
            experiment_id=experiment_id,
            timestamp=timestamp,
            model_metadata=bundle.metadata,
            generation_result=result,
        )
        writer.append_output(record)
        results = [result]
    else:
        results = generate_batch(bundle, prompts, gen_config)
        records = [
            build_result_record(
                experiment_id=experiment_id,
                timestamp=timestamp,
                model_metadata=bundle.metadata,
                generation_result=r,
            )
            for r in results
        ]
        writer.write_outputs(records)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Experiment ID:    {experiment_id}")
    logger.info(f"Prompts:          {len(prompts)}")
    logger.info(f"Results dir:      results/raw/{experiment_id}/")
    logger.info(f"Model load time:  {bundle.load_time_seconds:.1f}s")

    total_gen_time = sum(r.generation_seconds for r in results)
    total_output_tokens = sum(r.output_tokens for r in results)
    logger.info(f"Total gen time:   {total_gen_time:.2f}s")
    logger.info(f"Total output tok: {total_output_tokens}")

    if total_gen_time > 0:
        logger.info(
            f"Avg throughput:   {total_output_tokens / total_gen_time:.1f} tok/s"
        )

    # Print first result for quick inspection
    if results:
        logger.info("-" * 60)
        logger.info("First response (truncated to 500 chars):")
        logger.info(results[0].text[:500])
        logger.info("-" * 60)


if __name__ == "__main__":
    main()
