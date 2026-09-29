from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn


@dataclass
class BranchingPolicyConfig:
    input_dim: int
    hidden_dims: tuple[int, ...] = (128, 128)
    activation: str = "relu"

    @classmethod
    def from_dict(cls, values: dict) -> "BranchingPolicyConfig":
        values = dict(values)
        values["hidden_dims"] = tuple(values.get("hidden_dims", (128, 128)))
        return cls(**values)


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported branching-policy activation: {name}")


class BranchingPolicy(nn.Module):
    """Shared MLP that assigns one score to each candidate variable."""

    def __init__(self, config: BranchingPolicyConfig):
        super().__init__()
        self.config = config

        dimensions: Sequence[int] = (
            config.input_dim,
            *config.hidden_dims,
            1,
        )
        layers: list[nn.Module] = []
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                layers.append(_activation(config.activation))
        self.network = nn.Sequential(*layers)

        self.register_buffer("feature_mean", torch.zeros(config.input_dim))
        self.register_buffer("feature_std", torch.ones(config.input_dim))

    def set_normalization(
        self,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ) -> None:
        feature_mean = torch.as_tensor(feature_mean, dtype=torch.float32)
        feature_std = torch.as_tensor(feature_std, dtype=torch.float32).clamp_min(1e-8)
        if feature_mean.shape != (self.config.input_dim,):
            raise ValueError(
                f"Expected feature_mean shape ({self.config.input_dim},), "
                f"got {tuple(feature_mean.shape)}"
            )
        if feature_std.shape != (self.config.input_dim,):
            raise ValueError(
                f"Expected feature_std shape ({self.config.input_dim},), "
                f"got {tuple(feature_std.shape)}"
            )
        self.feature_mean.copy_(feature_mean)
        self.feature_std.copy_(feature_std)

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        candidate_features = candidate_features.to(
            device=self.feature_mean.device,
            dtype=torch.float32,
        )
        normalized = (candidate_features - self.feature_mean) / self.feature_std
        return self.network(normalized).squeeze(-1)

    def save(self, path: str | Path, feature_names: list[str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
                "feature_names": list(feature_names),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> tuple["BranchingPolicy", list[str]]:
        checkpoint = torch.load(
            Path(path),
            map_location=map_location,
            weights_only=False,
        )
        policy = cls(BranchingPolicyConfig.from_dict(checkpoint["config"]))
        policy.load_state_dict(checkpoint["state_dict"])
        policy.to(map_location)
        policy.eval()
        return policy, list(checkpoint.get("feature_names", []))