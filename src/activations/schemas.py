"""
Activation metadata schema.

Defines the metadata structure for stored activation tensors.
Actual extraction logic will be implemented later.

Activations are stored as separate ``.pt`` (PyTorch) files, each
accompanied by a JSON sidecar containing this metadata.  This
separation keeps the metadata human-readable and allows querying
without loading multi-GB tensor files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ActivationMetadata:
    """Metadata for a single saved activation tensor.

    Attributes
    ----------
    model:
        Model identifier (Hugging Face name).
    layer:
        Layer index from which the activation was extracted.
    component:
        Component type (e.g. ``"residual_stream"``, ``"mlp_out"``,
        ``"attn_out"``).
    token_position:
        Which token position(s) the activation corresponds to.
        Can be an integer index, ``"last"``, ``"mean"``, etc.
    pooling_method:
        How multiple token positions were aggregated, if applicable
        (e.g. ``"last_token"``, ``"mean_pool"``, ``None``).
    dtype:
        String representation of the tensor dtype (e.g. ``"float16"``).
    shape:
        Tensor shape as a tuple of ints.
    experiment_id:
        The experiment run that produced this activation.
    prompt_id:
        The prompt that was fed to the model.
    extra:
        Arbitrary additional metadata.
    """

    model: str = ""
    layer: int = 0
    component: str = "residual_stream"
    token_position: str = "last"
    pooling_method: Optional[str] = None
    dtype: str = "float32"
    shape: Tuple[int, ...] = ()
    experiment_id: Optional[str] = None
    prompt_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model is required for ActivationMetadata.")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert tuple to list for JSON serialisation.
        d["shape"] = list(d["shape"])
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActivationMetadata":
        data = dict(data)  # shallow copy
        if "shape" in data:
            data["shape"] = tuple(data["shape"])
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
