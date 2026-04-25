from __future__ import annotations

import torch
from torch import nn


class PolicyMLP(nn.Module):
    """Score one candidate action given state and action features."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        input_dim = state_dim + action_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_features: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        """Return one scalar score for each candidate action."""
        x = torch.cat([state_features, action_features], dim=-1)
        return self.network(x).squeeze(-1)
