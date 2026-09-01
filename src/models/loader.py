"""
Model loader — safe, configurable loading of Hugging Face causal LMs.

Design principles:
    - Configuration-driven: model choice comes from YAML, not hard-coded.
    - Fail loudly: missing or invalid config raises immediately.
    - Lazy: nothing is downloaded until ``load_model`` is called.
    - Hardware-aware: auto-detects CUDA, respects dtype/quantisation flags.
    - Does NOT assume a specific model architecture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

from src.utils.logging import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Validated model configuration.

    Every field corresponds to a key in ``configs/model.yaml``.
    """

    model_name: Optional[str] = None
    revision: Optional[str] = None
    trust_remote_code: bool = False
    device: str = "auto"
    dtype: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    allow_cpu_fallback: bool = True
    max_memory: Optional[str] = None
    seed: int = 42
    generation: Dict[str, Any] = field(default_factory=lambda: {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 50,
        "do_sample": False,
        "repetition_penalty": 1.0,
    })

    def __post_init__(self) -> None:
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError(
                "Cannot set both load_in_4bit and load_in_8bit to True."
            )

    def resolve_device(self) -> torch.device:
        """Return the concrete ``torch.device`` to use.

        Raises
        ------
        RuntimeError
            If CUDA is requested but unavailable and fallback is disabled.
        """
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.info("CUDA not available — using CPU (auto-detect).")
            return torch.device("cpu")

        if self.device == "cuda" and not torch.cuda.is_available():
            if self.allow_cpu_fallback:
                logger.warning(
                    "CUDA requested but unavailable. "
                    "Falling back to CPU (allow_cpu_fallback=true)."
                )
                return torch.device("cpu")
            raise RuntimeError(
                "CUDA requested but unavailable, and allow_cpu_fallback is "
                "disabled. Set device='auto' or allow_cpu_fallback=true, "
                "or run on a machine with CUDA."
            )

        return torch.device(self.device)

    def resolve_dtype(self, device: Optional[torch.device] = None) -> Optional[torch.dtype]:
        """Return the concrete ``torch.dtype``, or ``None`` for model default.

        Parameters
        ----------
        device:
            If provided, used to select sensible defaults when dtype is "auto".
        """
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": None,
        }
        if self.dtype not in mapping:
            raise ValueError(
                f"Unsupported dtype '{self.dtype}'. "
                f"Choose from: {list(mapping.keys())}"
            )

        if self.dtype == "auto":
            if device is not None and device.type == "cuda":
                logger.info("dtype=auto on CUDA → selecting float16.")
                return torch.float16
            elif device is not None and device.type == "cpu":
                logger.info("dtype=auto on CPU → selecting float32.")
                return torch.float32
            return None

        # Validate bfloat16 compatibility
        resolved = mapping[self.dtype]
        if resolved == torch.bfloat16:
            if device is not None and device.type == "cuda":
                if not torch.cuda.is_bf16_supported():
                    logger.warning(
                        "bfloat16 requested but not supported on this GPU. "
                        "Consider using float16 instead."
                    )

        return resolved

    def get_model_metadata(self, device: torch.device, dtype: Optional[torch.dtype]) -> Dict[str, Any]:
        """Return a metadata dictionary describing this model configuration.

        Parameters
        ----------
        device:
            The resolved device.
        dtype:
            The resolved dtype.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "name": self.model_name,
            "revision": self.revision,
            "dtype": str(dtype) if dtype else "model_default",
            "device": str(device),
            "quantized": self.load_in_4bit or self.load_in_8bit,
            "quantization_bits": 4 if self.load_in_4bit else (8 if self.load_in_8bit else None),
            "trust_remote_code": self.trust_remote_code,
        }


def load_model_config(config_path: Path) -> ModelConfig:
    """Load and validate a model configuration from a YAML file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    ModelConfig

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    ValueError
        If the configuration is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    kwargs = {
        "model_name": raw.get("model_name"),
        "revision": raw.get("revision"),
        "trust_remote_code": raw.get("trust_remote_code", False),
        "device": raw.get("device", "auto"),
        "dtype": raw.get("dtype", "auto"),
        "load_in_4bit": raw.get("load_in_4bit", False),
        "load_in_8bit": raw.get("load_in_8bit", False),
        "allow_cpu_fallback": raw.get("allow_cpu_fallback", True),
        "max_memory": raw.get("max_memory"),
        "seed": raw.get("seed", 42),
    }
    if "generation" in raw and raw["generation"] is not None:
        kwargs["generation"] = raw["generation"]

    return ModelConfig(**kwargs)


# ---------------------------------------------------------------------------
# Model bundle (return type)
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """Container returned by ``load_model``.

    Holds the model, tokenizer, and metadata together so they travel
    as a single unit through the pipeline.
    """

    model: Any  # torch.nn.Module — typed as Any to avoid import
    tokenizer: Any  # PreTrainedTokenizer
    metadata: Dict[str, Any] = field(default_factory=dict)
    load_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config: ModelConfig) -> ModelBundle:
    """Load a Hugging Face causal language model and its tokenizer.

    Parameters
    ----------
    config:
        A validated ``ModelConfig`` instance.

    Returns
    -------
    ModelBundle
        Contains model, tokenizer, metadata dict, and load timing.

    Raises
    ------
    ValueError
        If ``config.model_name`` is ``None`` or empty.
    RuntimeError
        If CUDA is required but unavailable and fallback is disabled.
    """
    if not config.model_name:
        raise ValueError(
            "model_name is required but not set.  "
            "Specify it in configs/model.yaml or pass it explicitly."
        )

    # Import here to avoid slow transformers import at module level.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = config.resolve_device()
    dtype = config.resolve_dtype(device)

    logger.info(
        f"Loading model '{config.model_name}' "
        f"(revision={config.revision}, device={device}, dtype={dtype})"
    )

    if config.trust_remote_code:
        logger.info("trust_remote_code is ENABLED for this model.")

    # Build kwargs for from_pretrained
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    # Quantization (requires bitsandbytes + accelerate)
    if config.load_in_4bit or config.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "Quantization requires bitsandbytes and accelerate. "
                "Install them with: pip install bitsandbytes accelerate"
            ) from exc

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"

        # Apply max_memory constraint if set
        if config.max_memory:
            model_kwargs["max_memory"] = {0: config.max_memory}
    else:
        # Non-quantized: load to resolved device.
        model_kwargs["device_map"] = None

    start_time = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )

    # Ensure pad token exists (many causal LMs lack one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            revision=config.revision,
            **model_kwargs,
        )
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            f"Out of GPU memory loading '{config.model_name}'. "
            f"Try enabling 4-bit quantization (load_in_4bit: true) "
            f"or using a smaller model."
        )

    # Move to device if not using device_map="auto".
    if model_kwargs.get("device_map") is None:
        model = model.to(device)

    model.eval()
    load_time = time.perf_counter() - start_time

    # Build metadata
    metadata = config.get_model_metadata(device, dtype)
    metadata["load_time_seconds"] = round(load_time, 2)

    # Report GPU memory usage
    if device.type == "cuda":
        mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        metadata["gpu_memory_allocated_mb"] = round(mem_allocated, 1)
        metadata["gpu_memory_reserved_mb"] = round(mem_reserved, 1)
        logger.info(
            f"GPU memory: {mem_allocated:.1f} MB allocated, "
            f"{mem_reserved:.1f} MB reserved."
        )

    logger.info(
        f"Model loaded successfully on {device} in {load_time:.1f}s."
    )

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        metadata=metadata,
        load_time_seconds=load_time,
    )
