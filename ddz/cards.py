from __future__ import annotations

from collections import Counter
import random
from typing import Iterable, List

# 斗地主内部统一使用这些 rank 字符串表示牌面。
# 顺序既决定了排序结果，也决定了比较大小时使用的数值。
RANK_ORDER = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "BJ", "RJ"]
# 3 -> 3, 4 -> 4, ... , RJ -> 17。后续所有比大小都依赖这张映射表。
RANK_VALUE = {rank: index + 3 for index, rank in enumerate(RANK_ORDER)}
# 顺子/连对/飞机只能使用 3~A，不能包含 2 和大小王。
SEQUENCE_RANKS = RANK_ORDER[:12]
# 一副标准斗地主牌：3~2 各四张，加大小王。
FULL_DECK = [rank for rank in RANK_ORDER[:-2] for _ in range(4)] + ["BJ", "RJ"]


def normalize_cards(cards: Iterable[str]) -> List[str]:
    """Validate, normalize, and sort raw card tokens."""
    # 统一清洗输入，避免外部传入大小写不一致或多余空白。
    normalized = [card.strip().upper() for card in cards if card.strip()]
    invalid = [card for card in normalized if card not in RANK_VALUE]
    if invalid:
        raise ValueError(f"Invalid cards: {', '.join(invalid)}")
    return sort_cards(normalized)


def sort_cards(cards: Iterable[str]) -> List[str]:
    """Return cards sorted by Dou Dizhu rank order."""
    # 先按牌力排序；第二关键字保留稳定、可预期的字符串顺序。
    return sorted(cards, key=lambda card: (RANK_VALUE[card], card))


def card_counter(cards: Iterable[str]) -> Counter[str]:
    """Count how many times each rank appears in a card collection."""
    # Counter 在判型时很常用，用来判断每个点数出现了几次。
    return Counter(cards)


def build_deck() -> List[str]:
    """Create a fresh standard 54-card Dou Dizhu deck."""
    # 返回新列表，避免调用方意外修改模块级常量 FULL_DECK。
    return list(FULL_DECK)


def shuffled_deck(seed: int | None = None) -> List[str]:
    """Return a shuffled deck, optionally reproducible via seed."""
    # 使用局部 Random，这样传 seed 时可复现，且不会污染全局随机状态。
    deck = build_deck()
    rng = random.Random(seed)
    rng.shuffle(deck)
    return deck
