from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch
from torch import nn
from torch.optim import AdamW

from training.dataset import (
    build_action_label,
    encode_action_features,
    encode_state_features,
    feature_sizes,
    load_samples_from_jsonl,
)
from training.model import PolicyMLP


@dataclass(frozen=True)
class TrainConfig:
    """Minimal training configuration for the first policy baseline."""

    data_path: str = "training/data/heuristic_self_play.jsonl"
    model_path: str = "training/checkpoints/policy_mlp.pt"
    epochs: int = 3
    learning_rate: float = 1e-3
    hidden_dim: int = 256


def train_policy(config: TrainConfig) -> None:
    """Train a small MLP to imitate heuristic action choices."""
    samples = load_samples_from_jsonl(config.data_path)
    if not samples:
        raise ValueError(f"No samples found at {config.data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim, action_dim = feature_sizes()
    model = PolicyMLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=config.hidden_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    progress_interval = max(1, len(samples) // 20)

    print(
        f"[train] device={device} samples={len(samples)} epochs={config.epochs} "
        f"hidden_dim={config.hidden_dim} lr={config.learning_rate}"
    )

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        total_loss = 0.0
        total_correct = 0
        for sample_index, sample in enumerate(samples, start=1):
            state_vector = torch.tensor(encode_state_features(sample), dtype=torch.float32, device=device)
            action_matrix = torch.tensor(
                [encode_action_features(action) for action in sample.legal_actions],
                dtype=torch.float32,
                device=device,
            )
            state_matrix = state_vector.unsqueeze(0).repeat(action_matrix.size(0), 1)
            label = torch.tensor([build_action_label(sample)], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(state_matrix, action_matrix).unsqueeze(0)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_correct += int(logits.argmax(dim=1).item() == label.item())

            if sample_index % progress_interval == 0 or sample_index == len(samples):
                running_loss = total_loss / sample_index
                running_accuracy = total_correct / sample_index
                progress = sample_index / len(samples)
                print(
                    f"[train] epoch={epoch}/{config.epochs} step={sample_index}/{len(samples)} "
                    f"progress={progress:.0%} loss={running_loss:.4f} accuracy={running_accuracy:.4f}"
                )

        avg_loss = total_loss / len(samples)
        accuracy = total_correct / len(samples)
        epoch_seconds = time.perf_counter() - epoch_start
        print(
            f"[train] epoch_done={epoch}/{config.epochs} loss={avg_loss:.4f} "
            f"accuracy={accuracy:.4f} seconds={epoch_seconds:.1f}"
        )

    model_path = Path(config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": config.hidden_dim,
            "model_state_dict": model.state_dict(),
        },
        model_path,
    )
    print(f"[train] saved_model={model_path}")


def main() -> None:
    train_policy(TrainConfig())


if __name__ == "__main__":
    main()
