from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
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
    best_model_path: str = "training/checkpoints/policy_mlp_best.pt"
    epochs: int = 5
    learning_rate: float = 1e-3
    hidden_dim: int = 256
    validation_fraction: float = 0.2
    shuffle_seed: int = 42


def train_policy(config: TrainConfig) -> None:
    """Train a small MLP to imitate heuristic action choices."""
    samples = load_samples_from_jsonl(config.data_path)
    if not samples:
        raise ValueError(f"No samples found at {config.data_path}")

    train_samples, validation_samples = _split_samples(
        samples=samples,
        validation_fraction=config.validation_fraction,
        shuffle_seed=config.shuffle_seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim, action_dim = feature_sizes()
    model = PolicyMLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=config.hidden_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    progress_interval = max(1, len(train_samples) // 20)
    best_validation_accuracy = float("-inf")

    print(
        f"[train] device={device} train_samples={len(train_samples)} validation_samples={len(validation_samples)} "
        f"epochs={config.epochs} hidden_dim={config.hidden_dim} lr={config.learning_rate}"
    )

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        total_loss = 0.0
        total_correct = 0
        for sample_index, sample in enumerate(train_samples, start=1):
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

            if sample_index % progress_interval == 0 or sample_index == len(train_samples):
                running_loss = total_loss / sample_index
                running_accuracy = total_correct / sample_index
                progress = sample_index / len(train_samples)
                print(
                    f"[train] epoch={epoch}/{config.epochs} step={sample_index}/{len(train_samples)} "
                    f"progress={progress:.0%} loss={running_loss:.4f} accuracy={running_accuracy:.4f}"
                )

        avg_loss = total_loss / len(train_samples)
        accuracy = total_correct / len(train_samples)
        validation_loss, validation_accuracy = _evaluate_samples(
            model=model,
            samples=validation_samples,
            criterion=criterion,
            device=device,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        print(
            f"[train] epoch_done={epoch}/{config.epochs} loss={avg_loss:.4f} "
            f"accuracy={accuracy:.4f} val_loss={validation_loss:.4f} "
            f"val_accuracy={validation_accuracy:.4f} seconds={epoch_seconds:.1f}"
        )

        if validation_accuracy >= best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            _save_checkpoint(
                model=model,
                output_path=Path(config.best_model_path),
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=config.hidden_dim,
            )
            print(
                f"[train] best_model_updated path={config.best_model_path} "
                f"val_accuracy={validation_accuracy:.4f}"
            )

    _save_checkpoint(
        model=model,
        output_path=Path(config.model_path),
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
    )
    print(f"[train] saved_model={config.model_path}")


def _split_samples(
    samples: list,
    validation_fraction: float,
    shuffle_seed: int,
) -> tuple[list, list]:
    """Split samples into train and validation subsets."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0.0, 1.0).")

    shuffled = list(samples)
    random.Random(shuffle_seed).shuffle(shuffled)

    validation_size = int(len(shuffled) * validation_fraction)
    if validation_fraction > 0.0 and validation_size == 0 and len(shuffled) > 1:
        validation_size = 1
    if validation_size >= len(shuffled):
        validation_size = len(shuffled) - 1

    validation_samples = shuffled[:validation_size]
    train_samples = shuffled[validation_size:]
    return train_samples, validation_samples


def _evaluate_samples(
    model: PolicyMLP,
    samples: list,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the current model on a held-out sample list."""
    if not samples:
        return 0.0, 0.0

    total_loss = 0.0
    total_correct = 0
    model.eval()
    with torch.no_grad():
        for sample in samples:
            state_vector = torch.tensor(encode_state_features(sample), dtype=torch.float32, device=device)
            action_matrix = torch.tensor(
                [encode_action_features(action) for action in sample.legal_actions],
                dtype=torch.float32,
                device=device,
            )
            state_matrix = state_vector.unsqueeze(0).repeat(action_matrix.size(0), 1)
            label = torch.tensor([build_action_label(sample)], dtype=torch.long, device=device)

            logits = model(state_matrix, action_matrix).unsqueeze(0)
            loss = criterion(logits, label)
            total_loss += float(loss.item())
            total_correct += int(logits.argmax(dim=1).item() == label.item())
    model.train()
    return total_loss / len(samples), total_correct / len(samples)


def _save_checkpoint(
    model: PolicyMLP,
    output_path: Path,
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
) -> None:
    """Persist one model checkpoint to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "model_state_dict": model.state_dict(),
        },
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a policy model from exported self-play data.")
    parser.add_argument(
        "--data-path",
        default="training/data/heuristic_self_play.jsonl",
        help="Path to the exported JSONL dataset.",
    )
    parser.add_argument(
        "--model-path",
        default="training/checkpoints/policy_mlp.pt",
        help="Where to save the final checkpoint.",
    )
    parser.add_argument(
        "--best-model-path",
        default="training/checkpoints/policy_mlp_best.pt",
        help="Where to save the best validation checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for the MLP.")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for validation.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Shuffle seed for train/validation split.")
    args = parser.parse_args()

    train_policy(
        TrainConfig(
            data_path=args.data_path,
            model_path=args.model_path,
            best_model_path=args.best_model_path,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            validation_fraction=args.validation_fraction,
            shuffle_seed=args.shuffle_seed,
        )
    )


if __name__ == "__main__":
    main()
