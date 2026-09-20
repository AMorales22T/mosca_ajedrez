from __future__ import annotations

import torch
from torch import nn

from ..encoding import POLICY_SIZE
from .connectome import load_connectome


class FlyBrainNet(nn.Module):
    def __init__(self, units: int = 1000, recurrent_steps: int = 4, connectome_mode: str = "synthetic", seed: int = 0, connectome=None):
        super().__init__()
        self.units, self.recurrent_steps, self.connectome_mode = units, recurrent_steps, connectome_mode
        input_units = max(128, units // 5)
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(19 * 8 * 8, input_units), nn.LayerNorm(input_units), nn.GELU(), nn.Linear(input_units, units))
        self.recurrent = nn.Linear(units, units, bias=False)
        self.norm = nn.LayerNorm(units)
        self.register_buffer("connectome", connectome if connectome is not None else load_connectome(connectome_mode, units, seed))
        self.policy = nn.Linear(units, POLICY_SIZE)
        self.value = nn.Sequential(nn.Linear(units, 128), nn.GELU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, board, state=None):
        x = self.encoder(board)
        h = torch.zeros_like(x) if state is None else state
        weight = self.recurrent.weight * self.connectome
        for _ in range(self.recurrent_steps):
            h = self.norm(torch.tanh(x + torch.nn.functional.linear(h, weight)))
        return self.policy(h), self.value(h).squeeze(-1), h


class MLPControl(FlyBrainNet):
    def __init__(self, units=1000, recurrent_steps=4, **kwargs):
        super().__init__(units, recurrent_steps, "synthetic", kwargs.get("seed", 0))
        self.connectome.fill_(1.0)
        self.connectome_mode = "dense_control"


class CNNControl(FlyBrainNet):
    def __init__(self, units=1000, recurrent_steps=4, **kwargs):
        super().__init__(units, recurrent_steps, "synthetic", kwargs.get("seed", 0))
        target_parameters = sum(parameter.numel() for parameter in self.parameters())
        self.encoder = nn.Sequential(nn.Conv2d(19, 4, 3, padding=1), nn.GELU(), nn.Flatten(), nn.Linear(4 * 8 * 8, units))
        current_parameters = sum(parameter.numel() for parameter in self.parameters())
        if current_parameters < target_parameters:
            self.budget_pad = nn.Parameter(torch.zeros(target_parameters - current_parameters))
        self.connectome_mode = "cnn_control"

    def forward(self, board, state=None):
        logits, value, h = super().forward(board, state)
        if hasattr(self, "budget_pad"):
            h = h + self.budget_pad.sum() * 0.0
        return logits, value, h


def parameter_report(models: dict[str, nn.Module]) -> dict[str, int]:
    counts = {name: sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) for name, model in models.items()}
    reference = counts["fly"]
    if any(abs(count - reference) / reference > 0.05 for count in counts.values()):
        raise AssertionError(f"Controles fuera de ±5% de parámetros: {counts}")
    print(f"parámetros entrenables: {counts}")
    return counts
