"""
System information utilities.

Used by the environment-check script and by experiment metadata capture.
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict

import torch


def get_system_info() -> Dict[str, Any]:
    """Return a dictionary of system and library information.

    This function must never raise — missing information is reported as
    ``None`` or ``"unavailable"``.
    """
    info: Dict[str, Any] = {}

    # Python
    info["python_version"] = sys.version
    info["platform"] = platform.platform()

    # PyTorch
    info["torch_version"] = torch.__version__

    # Transformers
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = "not installed"

    # NumPy
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except ImportError:
        info["numpy_version"] = "not installed"

    # CUDA
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_mb"] = int(props.total_mem / (1024 * 1024))
        info["gpu_count"] = torch.cuda.device_count()
    else:
        info["cuda_version"] = None
        info["gpu_name"] = None
        info["gpu_memory_mb"] = None
        info["gpu_count"] = 0

    # CPU / RAM
    info["cpu"] = platform.processor() or "unknown"
    info["cpu_count"] = platform.machine()
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        info["ram_gb"] = "psutil not installed"

    return info


def print_system_info() -> None:
    """Print a human-readable environment report to stdout."""
    info = get_system_info()
    width = 28
    print("=" * 56)
    print("  ProvenanceShift -- Environment Report")
    print("=" * 56)
    print(f"{'Python:':<{width}} {info['python_version'].split()[0]}")
    print(f"{'Platform:':<{width}} {info['platform']}")
    print(f"{'PyTorch:':<{width}} {info['torch_version']}")
    print(f"{'Transformers:':<{width}} {info['transformers_version']}")
    print(f"{'NumPy:':<{width}} {info['numpy_version']}")
    print("-" * 56)
    print(f"{'CUDA available:':<{width}} {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"{'CUDA version:':<{width}} {info['cuda_version']}")
        print(f"{'GPU:':<{width}} {info['gpu_name']}")
        print(f"{'GPU memory (MB):':<{width}} {info['gpu_memory_mb']}")
        print(f"{'GPU count:':<{width}} {info['gpu_count']}")
    else:
        print(f"{'GPU:':<{width}} none (CPU-only mode)")
    print("-" * 56)
    print(f"{'CPU:':<{width}} {info['cpu']}")
    print(f"{'Architecture:':<{width}} {info['cpu_count']}")
    print(f"{'RAM (GB):':<{width}} {info['ram_gb']}")
    print("=" * 56)
