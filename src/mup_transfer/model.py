from dataclasses import dataclass

import torch
from torch import nn

from mup_transfer import width_multiplier


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int = 8
    output_dim: int = 4
    width: int = 32
    base_width: int = 32
    depth: int = 2
    parametrization: str = "mup"

    def validate(self) -> None:
        if min(self.input_dim, self.output_dim, self.width, self.base_width) <= 0:
            raise ValueError("all dimensions must be positive")
        if self.depth < 1:
            raise ValueError("depth must be at least one")
        if self.parametrization not in {"mup", "standard"}:
            raise ValueError("parametrization must be 'mup' or 'standard'")


class WidthMLP(nn.Module):
    """ReLU MLP with a width-correct μP readout and fan-in initialization."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dimensions = [config.input_dim, *([config.width] * config.depth), config.output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(len(dimensions) - 1)
        )
        self.readout_multiplier = (
            width_multiplier(config.width, config.base_width)
            if config.parametrization == "mup"
            else 1.0
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=layer.in_features**-0.5)
            nn.init.zeros_(layer.bias)

    def forward_with_activations(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = inputs
        activations = []
        for layer in self.layers[:-1]:
            hidden = torch.relu(layer(hidden))
            activations.append(hidden)
        output = self.layers[-1](hidden) * self.readout_multiplier
        return output, activations

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_activations(inputs)[0]
