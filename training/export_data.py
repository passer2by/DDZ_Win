from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ddz.agent import HeuristicAgent, TurnView
from ddz.cards import normalize_cards, sort_cards
from ddz.generator import generate_legal_plays
from ddz.simulator import deal_new_game
from ddz.state import HistoryRecord, Play
from training.dataset import TrainingSample, build_training_sample, write_samples_to_jsonl


@dataclass(frozen=True)
class ExportSummary:
    """Summary of one export run."""

    games: int
    samples: int
    output_path: str


def export_self_play_data(
    num_games: int,
    output_path: str | Path,
    seed: Optional[int] = None,
) -> ExportSummary:
    """Generate supervised-learning samples from heuristic self-play."""
    agents = [HeuristicAgent(), HeuristicAgent(), HeuristicAgent()]
    all_samples: list[TrainingSample] = []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] start games={num_games} output={output_path}")

    for index in range(num_games):
        game_seed = None if seed is None else seed + index
        landlord = index % 3
        game_samples = _play_one_game_for_export(agents=agents, seed=game_seed, landlord=landlord)
        all_samples.extend(game_samples)
        print(
            f"[export] game={index + 1}/{num_games} "
            f"landlord=P{landlord} samples_this_game={len(game_samples)} total_samples={len(all_samples)}"
        )

    write_samples_to_jsonl(all_samples, output_path)
    print(f"[export] done games={num_games} total_samples={len(all_samples)}")
    return ExportSummary(games=num_games, samples=len(all_samples), output_path=str(output_path))


def _play_one_game_for_export(
    agents: list[HeuristicAgent],
    seed: Optional[int],
    landlord: int,
) -> list[TrainingSample]:
    """Run one game and return all decision samples from that game."""
    deal = deal_new_game(seed=seed, landlord=landlord)
    hands = [list(hand) for hand in deal.hands]
    current_player = deal.landlord
    lead_player = deal.landlord
    last_play: Optional[Play] = None
    consecutive_passes = 0
    play_history: list[HistoryRecord] = []
    pending_samples: list[tuple[TurnView, list[list[str]], list[str]]] = []

    while True:
        hand_counts = [len(hand) for hand in hands]
        legal_patterns = generate_legal_plays(hands[current_player], last_play.cards if last_play else None)
        legal_actions = [list(pattern.cards) for pattern in legal_patterns]
        if not legal_actions:
            play_history.append(HistoryRecord(player=current_player, cards=[], is_pass=True))
            consecutive_passes += 1
            if consecutive_passes == 2:
                current_player = lead_player
                last_play = None
                consecutive_passes = 0
            else:
                current_player = (current_player + 1) % 3
            continue

        view = TurnView(
            player=current_player,
            landlord=deal.landlord,
            hand=sort_cards(hands[current_player]),
            hand_counts=hand_counts,
            last_play=last_play,
            play_history=tuple(
                HistoryRecord(player=record.player, cards=list(record.cards), is_pass=record.is_pass)
                for record in play_history
            ),
        )
        chosen_action = agents[current_player].choose_play(view)
        if chosen_action is None:
            chosen_action = legal_actions[0]

        normalized = normalize_cards(chosen_action)
        legal_card_sets = {tuple(action) for action in legal_actions}
        if tuple(normalized) not in legal_card_sets:
            normalized = legal_actions[0]

        pending_samples.append((view, legal_actions, list(normalized)))
        for card in normalized:
            hands[current_player].remove(card)
        last_play = Play(cards=normalized, player=current_player)
        lead_player = current_player
        consecutive_passes = 0
        play_history.append(HistoryRecord(player=current_player, cards=list(normalized), is_pass=False))

        if not hands[current_player]:
            winner = current_player
            return [
                build_training_sample(view=sample_view, legal_actions=legal, chosen_action=chosen, winner=winner)
                for sample_view, legal, chosen in pending_samples
            ]

        current_player = (current_player + 1) % 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Export heuristic self-play data for policy training.")
    parser.add_argument("--num-games", type=int, default=200, help="Number of self-play games to export.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("training") / "data" / "heuristic_self_play.jsonl",
        help="Path to the output JSONL file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducible exports.")
    args = parser.parse_args()

    summary = export_self_play_data(
        num_games=args.num_games,
        output_path=args.output_path,
        seed=args.seed,
    )
    print("games:", summary.games)
    print("samples:", summary.samples)
    print("output:", summary.output_path)


if __name__ == "__main__":
    main()
