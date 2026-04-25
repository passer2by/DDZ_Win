from __future__ import annotations

from pathlib import Path
from typing import Optional

from ddz.agent import TurnView
from ddz.cards import sort_cards
from ddz.generator import generate_legal_plays
from ddz.patterns import Pattern, identify_pattern
from training.dataset import encode_action_features, encode_state_features, feature_sizes, TrainingSample


class ModelPolicyAgent:
    """Use a trained policy checkpoint to score legal actions during play."""

    def __init__(self, checkpoint_path: str = "training/checkpoints/policy_mlp_best.pt") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self._torch = None
        self._model = None
        self._device = None
        self._load()

    def choose_play(self, view: TurnView) -> Optional[list[str]]:
        """Pick the legal action with the highest model score."""
        legal_patterns = generate_legal_plays(view.hand, view.last_play.cards if view.last_play else None)
        if not legal_patterns:
            return None

        sample = _build_inference_sample(view=view, legal_patterns=legal_patterns)
        state_vector = self._torch.tensor(encode_state_features(sample), dtype=self._torch.float32, device=self._device)
        action_matrix = self._torch.tensor(
            [encode_action_features(pattern.cards) for pattern in legal_patterns],
            dtype=self._torch.float32,
            device=self._device,
        )
        state_matrix = state_vector.unsqueeze(0).repeat(action_matrix.size(0), 1)

        self._model.eval()
        with self._torch.no_grad():
            scores = self._model(state_matrix, action_matrix)
        best_index = int(scores.argmax().item())
        return list(legal_patterns[best_index].cards)

    def _load(self) -> None:
        """Load torch lazily so the rest of the project can run without it."""
        import torch

        from training.model import PolicyMLP

        state_dim, action_dim = feature_sizes()
        checkpoint = torch.load(self.checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        hidden_dim = int(checkpoint.get("hidden_dim", 256))
        model = PolicyMLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        model.load_state_dict(checkpoint["model_state_dict"])

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        self._torch = torch


def _build_inference_sample(view: TurnView, legal_patterns: list[Pattern]) -> TrainingSample:
    """Mirror the training sample structure for model inference."""
    role = "landlord" if view.player == view.landlord else "farmer"
    last_pattern = identify_pattern(view.last_play.cards) if view.last_play is not None else None
    return TrainingSample(
        player=view.player,
        landlord=view.landlord,
        role=role,
        hand=sort_cards(view.hand),
        hand_counts=list(view.hand_counts),
        last_play_cards=[] if view.last_play is None else list(view.last_play.cards),
        last_play_player=None if view.last_play is None else view.last_play.player,
        play_history=[
            {
                "player": record.player,
                "cards": list(record.cards),
                "is_pass": record.is_pass,
            }
            for record in view.play_history
        ],
        legal_actions=[list(pattern.cards) for pattern in legal_patterns],
        chosen_action=list(legal_patterns[0].cards),
        winner=view.landlord,
        did_win=False,
        last_play_kind=None if last_pattern is None else last_pattern.kind,
        last_play_main_rank=None if last_pattern is None else last_pattern.main_rank,
    )
