from __future__ import annotations

from typing import Optional

from ddz.patterns import Pattern, find_patterns_from_hand
from ddz.rules import compare_patterns


def generate_legal_plays(hand: list[str], last_play_cards: Optional[list[str]]) -> list[Pattern]:
    candidates = find_patterns_from_hand(hand)
    if last_play_cards is None:
        return candidates

    from ddz.patterns import identify_pattern

    target = identify_pattern(last_play_cards)
    if target is None:
        raise ValueError("Last play is not a valid pattern.")

    legal = [pattern for pattern in candidates if compare_patterns(pattern, target) > 0]
    return legal
