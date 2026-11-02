"""Small frozen-feature heads used by the cost-reduction experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class NonNegativeCostHead(nn.Module):
    """Direct non-negative predictor for C/kappa (not a post-generation lookup)."""

    def __init__(self, hidden_dim: int, num_experts: int, dropout: float = 0.15):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.regressor = nn.Linear(hidden_dim, num_experts)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.regressor(self.dropout(self.layer_norm(features))))


@dataclass(frozen=True)
class TemperatureCalibration:
    temperatures: np.ndarray
    fitted_experts: np.ndarray
    source_role: str

    def transform(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.temperatures):
            raise ValueError("calibration logits/expert dimension mismatch")
        return values / self.temperatures[None, :]


def identity_calibration(num_experts: int, *, source_role: str = "NO_LEGAL_QCAL") -> TemperatureCalibration:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    temperatures = np.ones(num_experts, dtype=np.float64)
    fitted = np.zeros(num_experts, dtype=bool)
    temperatures.flags.writeable = False
    fitted.flags.writeable = False
    return TemperatureCalibration(temperatures, fitted, source_role)


def fit_per_expert_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    observed: np.ndarray,
    *,
    source_role: str,
    minimum_samples: int = 20,
    max_iter: int = 100,
) -> TemperatureCalibration:
    """Fit independent positive temperatures on an explicitly supplied qcal role.

    Experts with too few labels or all-one/all-zero labels retain identity.  The
    caller is responsible for proving this source role is disjoint from fit,
    dev, cert and test; the experiment runner refuses to call this function
    because the frozen repository has no legal qcal split.
    """

    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(observed, dtype=bool)
    if values.ndim != 2 or truth.shape != values.shape or mask.shape != values.shape:
        raise ValueError("calibration array shape mismatch")
    if not np.isfinite(values).all() or not np.isin(truth[mask], (0, 1)).all():
        raise ValueError("calibration logits/targets are invalid")
    temperatures = np.ones(values.shape[1], dtype=np.float64)
    fitted = np.zeros(values.shape[1], dtype=bool)
    for expert in range(values.shape[1]):
        selected = mask[:, expert]
        labels = truth[selected, expert]
        if selected.sum() < minimum_samples or len(np.unique(labels)) < 2:
            continue
        expert_logits = torch.tensor(values[selected, expert], dtype=torch.float64)
        expert_targets = torch.tensor(labels, dtype=torch.float64)
        log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [log_temperature], max_iter=int(max_iter), line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            temperature = log_temperature.clamp(-6.0, 6.0).exp()
            loss = F.binary_cross_entropy_with_logits(
                expert_logits / temperature, expert_targets
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        value = float(log_temperature.detach().clamp(-6.0, 6.0).exp())
        if np.isfinite(value) and value > 0:
            temperatures[expert] = value
            fitted[expert] = True
    temperatures.flags.writeable = False
    fitted.flags.writeable = False
    return TemperatureCalibration(temperatures, fitted, str(source_role))

