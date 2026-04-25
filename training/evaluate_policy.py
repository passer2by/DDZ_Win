from __future__ import annotations

import argparse
from dataclasses import dataclass

from ddz.agent import HeuristicAgent
from ddz.simulator import simulate_game
from training.policy_agent import ModelPolicyAgent


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for a quick model-vs-heuristic benchmark."""

    checkpoint_path: str = "training/checkpoints/policy_mlp_best.pt"
    num_games: int = 10
    seed: int = 42


def evaluate_policy(config: EvaluationConfig) -> None:
    """Compare one model-controlled seat against two heuristic opponents."""
    model_agent = ModelPolicyAgent(config.checkpoint_path)
    total_wins = 0
    landlord_wins = 0
    farmer_wins = 0
    seat_wins = [0, 0, 0]

    for index in range(config.num_games):
        model_seat = index % 3
        landlord = index % 3
        agents = [HeuristicAgent(), HeuristicAgent(), HeuristicAgent()]
        agents[model_seat] = model_agent
        result = simulate_game(agents=agents, seed=config.seed + index, landlord=landlord)
        did_win = result.winner == model_seat
        if did_win:
            total_wins += 1
            seat_wins[model_seat] += 1
            if model_seat == result.landlord:
                landlord_wins += 1
            else:
                farmer_wins += 1
        print(
            f"[eval] game={index + 1}/{config.num_games} model_seat=P{model_seat} "
            f"landlord=P{landlord} winner=P{result.winner} did_win={did_win} "
            f"running_win_rate={total_wins / (index + 1):.4f}"
        )

    print(
        f"[eval] games={config.num_games} model_wins={total_wins} "
        f"win_rate={total_wins / config.num_games:.4f}"
    )
    print(f"[eval] landlord_wins={landlord_wins} farmer_wins={farmer_wins} seat_wins={seat_wins}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy against heuristic agents.")
    parser.add_argument(
        "--checkpoint-path",
        default="training/checkpoints/policy_mlp_best.pt",
        help="Checkpoint to load for evaluation.",
    )
    parser.add_argument("--num-games", type=int, default=10, help="Number of evaluation games to run.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for evaluation games.")
    args = parser.parse_args()

    evaluate_policy(
        EvaluationConfig(
            checkpoint_path=args.checkpoint_path,
            num_games=args.num_games,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
