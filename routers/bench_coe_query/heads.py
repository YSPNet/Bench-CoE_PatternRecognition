"""Audited query classification heads."""

from __future__ import annotations

import torch
from torch import nn


class LayerNormLinearHead(nn.Module):
    """The strict minimal frozen-feature head."""

    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_experts)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.layer_norm(features))


class AuditedBenchCoEHead(nn.Module):
    """Hash-audited Bench-CoE candidate head used by vlrb_adapted_v1."""

    def __init__(self, hidden_dim: int, num_experts: int, dropout: float = 0.15):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(hidden_dim, num_experts)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.layer_norm(features)))


def trainable_parameter_names(module: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, parameter in module.named_parameters() if parameter.requires_grad)
