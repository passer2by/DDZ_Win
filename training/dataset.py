from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any, Iterable

from ddz.cards import RANK_ORDER
from ddz.agent import TurnView
from ddz.patterns import identify_pattern
from ddz.state import HistoryRecord


@dataclass(frozen=True)
class TrainingSample:
    """One supervised-learning sample extracted from self-play."""

    player: int
    landlord: int
    role: str
    hand: list[str]
    hand_counts: list[int]
    last_play_cards: list[str]
    last_play_player: int | None
    play_history: list[dict[str, Any]]
    legal_actions: list[list[str]]
    chosen_action: list[str]
    winner: int
    did_win: bool
    last_play_kind: str | None
    last_play_main_rank: str | None


PATTERN_ORDER = [
    "single",
    "pair",
    "triple",
    "triple_single",
    "triple_pair",
    "straight",
    "pair_straight",
    "plane",
    "plane_single",
    "plane_pair",
    "four_two_single",
    "four_two_pair",
    "bomb",
    "rocket",
]

RANK_INDEX = {rank: index for index, rank in enumerate(RANK_ORDER)}
PATTERN_INDEX = {pattern: index for index, pattern in enumerate(PATTERN_ORDER)}


def build_training_sample(
    view: TurnView,
    legal_actions: Iterable[list[str]],
    chosen_action: list[str],
    winner: int,
) -> TrainingSample:
    """Convert one simulator decision point into a serializable sample."""
    role = "landlord" if view.player == view.landlord else "farmer"
    last_pattern = identify_pattern(view.last_play.cards) if view.last_play is not None else None
    did_win = winner == view.landlord if role == "landlord" else winner != view.landlord
    return TrainingSample(
        player=view.player,
        landlord=view.landlord,
        role=role,
        hand=list(view.hand),
        hand_counts=list(view.hand_counts),
        last_play_cards=[] if view.last_play is None else list(view.last_play.cards),
        last_play_player=None if view.last_play is None else view.last_play.player,
        play_history=[_history_record_to_dict(record) for record in view.play_history],
        legal_actions=[list(action) for action in legal_actions],
        chosen_action=list(chosen_action),
        winner=winner,
        did_win=did_win,
        last_play_kind=None if last_pattern is None else last_pattern.kind,
        last_play_main_rank=None if last_pattern is None else last_pattern.main_rank,
    )


def write_samples_to_jsonl(samples: Iterable[TrainingSample], output_path: str | Path) -> None:
    """Write training samples to a JSONL file for easy inspection and reuse."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def load_samples_from_jsonl(input_path: str | Path) -> list[TrainingSample]:
    """Load training samples from a JSONL file."""
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as handle:
        return [TrainingSample(**json.loads(line)) for line in handle if line.strip()]


def encode_state_features(sample: TrainingSample) -> list[float]:
    """Encode one sample into a fixed-length state feature vector."""
    features: list[float] = []
    features.extend(_count_cards(sample.hand))
    features.extend(_encode_role(sample.role))
    features.extend(_encode_last_play(sample))
    features.extend(_normalize_counts(sample.hand_counts))
    features.extend(_encode_history_summary(sample))
    return features


def encode_action_features(action: list[str]) -> list[float]:
    """Encode one candidate action into a fixed-length action feature vector."""
    pattern = identify_pattern(action)
    if pattern is None:
        raise ValueError(f"Cannot encode invalid action: {action}")

    features: list[float] = []
    features.extend(_count_cards(action))
    features.extend(_one_hot(PATTERN_INDEX[pattern.kind], len(PATTERN_ORDER)))
    features.extend(_one_hot(RANK_INDEX[pattern.main_rank], len(RANK_ORDER)))
    features.append(pattern.length / 20.0)
    features.append(1.0 if pattern.kind == "bomb" else 0.0)
    features.append(1.0 if pattern.kind == "rocket" else 0.0)
    return features


def build_action_label(sample: TrainingSample) -> int:
    """Return the chosen-action index inside the legal action list."""
    chosen = tuple(sample.chosen_action)
    for index, action in enumerate(sample.legal_actions):
        if tuple(action) == chosen:
            return index
    raise ValueError("Chosen action was not found inside legal_actions.")


def feature_sizes() -> tuple[int, int]:
    """Return state and action feature dimensions."""
    dummy_sample = TrainingSample(
        player=0,
        landlord=0,
        role="landlord",
        hand=[],
        hand_counts=[0, 0, 0],
        last_play_cards=[],
        last_play_player=None,
        play_history=[],
        legal_actions=[],
        chosen_action=[],
        winner=0,
        did_win=False,
        last_play_kind=None,
        last_play_main_rank=None,
    )
    return len(encode_state_features(dummy_sample)), len(encode_action_features(["3"]))


def _history_record_to_dict(record: HistoryRecord) -> dict[str, Any]:
    """Convert one history record into a plain dictionary."""
    return {
        "player": record.player,
        "cards": list(record.cards),
        "is_pass": record.is_pass,
    }


def _count_cards(cards: list[str]) -> list[float]:
    counts = [0.0] * len(RANK_ORDER)
    for card in cards:
        counts[RANK_INDEX[card]] += 1.0
    return counts


def _encode_role(role: str) -> list[float]:
    return [1.0, 0.0] if role == "landlord" else [0.0, 1.0]


def _encode_last_play(sample: TrainingSample) -> list[float]:
    features: list[float] = [1.0 if sample.last_play_cards else 0.0]
    features.extend(_count_cards(sample.last_play_cards))

    kind_vector = [0.0] * len(PATTERN_ORDER)
    if sample.last_play_kind is not None:
        kind_vector[PATTERN_INDEX[sample.last_play_kind]] = 1.0
    features.extend(kind_vector)

    rank_vector = [0.0] * len(RANK_ORDER)
    if sample.last_play_main_rank is not None:
        rank_vector[RANK_INDEX[sample.last_play_main_rank]] = 1.0
    features.extend(rank_vector)
    return features


def _normalize_counts(hand_counts: list[int]) -> list[float]:
    return [count / 20.0 for count in hand_counts]


def _encode_history_summary(sample: TrainingSample) -> list[float]:
    seen_counts = _count_cards([card for record in sample.play_history for card in record["cards"]])
    passes = sum(1 for record in sample.play_history if record["is_pass"])
    return seen_counts + [passes / 100.0, len(sample.play_history) / 100.0, 1.0 if sample.did_win else 0.0]


def _one_hot(index: int, size: int) -> list[float]:
    vector = [0.0] * size
    vector[index] = 1.0
    return vector
