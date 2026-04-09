from __future__ import annotations

from collections import Counter
from typing import Iterable, List

RANK_ORDER = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "BJ", "RJ"]
RANK_VALUE = {rank: index + 3 for index, rank in enumerate(RANK_ORDER)}
SEQUENCE_RANKS = RANK_ORDER[:12]


def normalize_cards(cards: Iterable[str]) -> List[str]:
    normalized = [card.strip().upper() for card in cards if card.strip()]
    invalid = [card for card in normalized if card not in RANK_VALUE]
    if invalid:
        raise ValueError(f"Invalid cards: {', '.join(invalid)}")
    return sort_cards(normalized)


def sort_cards(cards: Iterable[str]) -> List[str]:
    return sorted(cards, key=lambda card: (RANK_VALUE[card], card))


def card_counter(cards: Iterable[str]) -> Counter[str]:
    return Counter(cards)
