"""
Text generation — deterministic, reproducible inference for causal LMs.

Handles prompt formatting (plain text or chat template), token generation,
output decoding, and structured result capture.  Does NOT store hidden
states — that belongs to later phases.

Design principles:
    - No global state: model/tokenizer are passed explicitly.
    - Seeds are set before every generation for reproducibility.
    - Only newly generated tokens are decoded (input is sliced off).
    - All metadata (timing, token counts, device, dtype) is recorded.
    - Chat templates are used when the tokenizer provides one.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch

from src.models.loader import ModelBundle
from src.utils.logging import get_logger
from src.utils.reproducibility import set_seed

logger = get_logger()


# ---------------------------------------------------------------------------
# Generation configuration
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Parameters controlling text generation.

    All fields are recorded alongside the output for reproducibility.
    """

    seed: int = 42
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 256
    do_sample: bool = False
    repetition_penalty: float = 1.0

    def validate(self) -> None:
        """Check that generation parameters are valid.

        Raises
        ------
        ValueError
            If any parameter is out of range.
        """
        errors: list[str] = []
        if self.seed < 0:
            errors.append("seed must be non-negative.")
        if self.temperature < 0.0:
            errors.append("temperature must be non-negative.")
        if not (0.0 < self.top_p <= 1.0):
            errors.append("top_p must be in (0, 1].")
        if self.top_k < 0:
            errors.append("top_k must be non-negative (0 = disabled).")
        if self.max_new_tokens < 1:
            errors.append("max_new_tokens must be >= 1.")
        if self.repetition_penalty <= 0.0:
            errors.append("repetition_penalty must be positive.")

        if errors:
            raise ValueError(
                "Generation config validation failed:\n  - "
                + "\n  - ".join(errors)
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        """Construct from a dictionary, ignoring unknown keys."""
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ---------------------------------------------------------------------------
# Generation result
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Structured output from a single generation call.

    Contains the generated text plus all metadata needed for
    reproducibility and analysis.
    """

    # Input
    prompt: str = ""
    input_tokens: int = 0

    # Output
    text: str = ""
    output_tokens: int = 0
    total_tokens: int = 0

    # Timing
    generation_seconds: float = 0.0
    tokens_per_second: float = 0.0

    # Configuration (snapshot)
    seed: int = 42
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 256
    do_sample: bool = False
    repetition_penalty: float = 1.0

    # Hardware
    device: str = ""
    dtype: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationResult":
        """Construct from a dictionary, ignoring unknown keys."""
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(
    tokenizer: Any,
    prompt: Union[str, List[Dict[str, str]]],
) -> str:
    """Format a prompt for the model, using the chat template if available.

    Parameters
    ----------
    tokenizer:
        The tokenizer (may have ``apply_chat_template``).
    prompt:
        Either a plain text string or a list of message dicts
        (``[{"role": "user", "content": "..."}]``).

    Returns
    -------
    str
        The formatted prompt string ready for tokenization.
    """
    # If already a list of messages, try to apply chat template
    if isinstance(prompt, list):
        messages = prompt
    else:
        # Wrap plain text as a single user message
        messages = [{"role": "user", "content": prompt}]

    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return formatted
        except Exception as exc:
            logger.warning(
                f"Chat template failed ({exc}); falling back to plain text."
            )

    # Fallback: concatenate message contents
    if isinstance(prompt, list):
        return "\n".join(msg.get("content", "") for msg in prompt)
    return prompt


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    bundle: ModelBundle,
    prompt: Union[str, List[Dict[str, str]]],
    config: GenerationConfig,
) -> GenerationResult:
    """Generate text from a single prompt.

    Parameters
    ----------
    bundle:
        A ``ModelBundle`` containing model, tokenizer, and metadata.
    prompt:
        Plain text string or list of message dicts.
    config:
        Generation parameters.

    Returns
    -------
    GenerationResult
        Structured result with text, token counts, timing, and metadata.
    """
    config.validate()
    set_seed(config.seed)

    model = bundle.model
    tokenizer = bundle.tokenizer

    # Determine device from model parameters
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Format prompt
    formatted_prompt = format_prompt(tokenizer, prompt)
    raw_prompt = prompt if isinstance(prompt, str) else str(prompt)

    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
    ).to(device)

    input_token_count = inputs["input_ids"].shape[-1]

    # Build generation kwargs
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "repetition_penalty": config.repetition_penalty,
        "use_cache": True,
    }

    # Only pass sampling params when sampling is enabled
    if config.do_sample:
        gen_kwargs["temperature"] = config.temperature if config.temperature > 0 else 1.0
        gen_kwargs["top_p"] = config.top_p
        gen_kwargs["top_k"] = config.top_k
    else:
        # Greedy decoding: temperature/top_p/top_k are not used
        pass

    # Generate
    start_time = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )

    generation_time = time.perf_counter() - start_time

    # Decode only newly generated tokens
    new_token_ids = output_ids[0, input_token_count:]
    output_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    output_token_count = len(new_token_ids)
    total_tokens = input_token_count + output_token_count

    # Compute throughput
    tokens_per_sec = (
        output_token_count / generation_time
        if generation_time > 0 else 0.0
    )

    result = GenerationResult(
        prompt=raw_prompt,
        input_tokens=input_token_count,
        text=output_text,
        output_tokens=output_token_count,
        total_tokens=total_tokens,
        generation_seconds=round(generation_time, 4),
        tokens_per_second=round(tokens_per_sec, 2),
        seed=config.seed,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        repetition_penalty=config.repetition_penalty,
        device=str(device),
        dtype=str(dtype),
    )

    logger.info(
        f"Generated {output_token_count} tokens in {generation_time:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )

    return result


def generate_batch(
    bundle: ModelBundle,
    prompts: List[Union[str, List[Dict[str, str]]]],
    config: GenerationConfig,
) -> List[GenerationResult]:
    """Generate text for a list of prompts, one at a time.

    This is a simple sequential loop — no batched GPU inference.
    Reliable first; optimise later.

    Parameters
    ----------
    bundle:
        A ``ModelBundle`` containing model, tokenizer, and metadata.
    prompts:
        List of prompt strings or message-dict lists.
    config:
        Generation parameters (applied identically to each prompt).

    Returns
    -------
    List[GenerationResult]
    """
    results = []
    for i, prompt in enumerate(prompts):
        logger.info(f"Generating {i + 1}/{len(prompts)}...")
        result = generate(bundle, prompt, config)
        results.append(result)
    return results
