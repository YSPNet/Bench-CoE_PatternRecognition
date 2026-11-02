"""Unified random seeding and RNG state capture."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    workspace = None
    if deterministic:
        # cuBLAS reads this at CUDA-context initialization, so configure it
        # before importing torch or touching torch.cuda in this process.
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "deterministic execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
            )
    random.seed(seed)
    np.random.seed(seed)
    result: dict[str, Any] = {
        "seed": int(seed),
        "python_seeded": True,
        "numpy_seeded": True,
        "torch_available": False,
        "deterministic_requested": bool(deterministic),
        "determinism_mode": "fail_closed" if deterministic else "disabled",
        "warnings": [],
    }
    try:
        import torch
    except ImportError:
        return result
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    result["torch_available"] = True
    result["cuda_available"] = torch.cuda.is_available()
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("PyTorch deterministic algorithms were not enabled")
        warn_only = (
            bool(torch.is_deterministic_algorithms_warn_only_enabled())
            if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
            else False
        )
        if warn_only:
            raise RuntimeError("PyTorch deterministic algorithms unexpectedly remain warn-only")
        result["torch_deterministic_algorithms"] = True
        result["torch_deterministic_warn_only"] = False
        result["cublas_workspace_config"] = workspace
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda"):
            cuda_backend = torch.backends.cuda
            required = ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp")
            if torch.cuda.is_available() and not all(hasattr(cuda_backend, name) for name in required):
                raise RuntimeError("cannot enforce deterministic SDPA backend with this PyTorch build")
            if all(hasattr(cuda_backend, name) for name in required):
                cuda_backend.enable_flash_sdp(False)
                cuda_backend.enable_mem_efficient_sdp(False)
                cuda_backend.enable_math_sdp(True)
            if hasattr(cuda_backend, "matmul") and hasattr(cuda_backend.matmul, "allow_tf32"):
                cuda_backend.matmul.allow_tf32 = False
            result["scaled_dot_product_attention"] = {
                "flash_enabled": bool(cuda_backend.flash_sdp_enabled())
                if hasattr(cuda_backend, "flash_sdp_enabled")
                else None,
                "memory_efficient_enabled": bool(cuda_backend.mem_efficient_sdp_enabled())
                if hasattr(cuda_backend, "mem_efficient_sdp_enabled")
                else None,
                "math_enabled": bool(cuda_backend.math_sdp_enabled())
                if hasattr(cuda_backend, "math_sdp_enabled")
                else None,
            }
            attention = result["scaled_dot_product_attention"]
            if torch.cuda.is_available() and (
                attention["flash_enabled"] is not False
                or attention["memory_efficient_enabled"] is not False
                or attention["math_enabled"] is not True
            ):
                raise RuntimeError("deterministic SDPA backend policy was not enforced")
        result["reproducibility_scope"] = (
            "fail-closed deterministic kernels within a fixed software/hardware execution geometry; "
            "no cross-geometry bitwise claim"
        )
    else:
        result["torch_deterministic_algorithms"] = bool(
            torch.are_deterministic_algorithms_enabled()
        )
    return result


def capture_rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
