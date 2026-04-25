from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

from ddz.cards import RANK_VALUE, SEQUENCE_RANKS, card_counter, normalize_cards


PATTERN_STRENGTH = {
    # strength 只用于跨牌型时的粗粒度优先级。
    # 普通牌型默认同级，炸弹高一档，王炸最高。
    "single": 1,
    "pair": 1,
    "triple": 1,
    "triple_single": 1,
    "triple_pair": 1,
    "straight": 1,
    "pair_straight": 1,
    "plane": 1,
    "plane_single": 1,
    "plane_pair": 1,
    "four_two_single": 1,
    "four_two_pair": 1,
    "bomb": 2,
    "rocket": 3,
}


@dataclass(frozen=True)
class Pattern:
    # kind: 牌型名称
    # main_rank: 比大小时真正拿来比较的“主点数”
    # cards: 组成该牌型的具体牌，已经排好序
    # length: 牌张数，比较时很多牌型要求长度一致
    kind: str
    main_rank: str
    cards: list[str]
    length: int

    @property
    def strength(self) -> int:
        """Return the coarse priority tier used during sorting."""
        return PATTERN_STRENGTH[self.kind]


def identify_pattern(cards: Iterable[str]) -> Optional[Pattern]:
    """Recognize the exact Dou Dizhu pattern formed by the given cards."""
    # Every rule check below tries to answer one question:
    # "If these cards are played together, what exact pattern do they form?"
    normalized = normalize_cards(cards)
    length = len(normalized)
    counts = card_counter(normalized)
    # ranks 只保留去重后的点数，并按实际牌力升序排列。
    # 后面顺子、连对、飞机都依赖这个顺序。
    ranks = sorted(counts, key=lambda rank: RANK_VALUE[rank])

    # 王炸在斗地主里是最高牌型，必须优先识别。
    if length == 2 and counts.get("BJ") == 1 and counts.get("RJ") == 1:
        return Pattern("rocket", "RJ", normalized, length)

    if length == 1:
        return Pattern("single", normalized[0], normalized, length)

    if length == 2 and len(counts) == 1:
        return Pattern("pair", normalized[0], normalized, length)

    if length == 3 and len(counts) == 1:
        return Pattern("triple", normalized[0], normalized, length)

    if length == 4:
        if len(counts) == 1:
            return Pattern("bomb", normalized[0], normalized, length)
        # 4 张牌里如果不是炸弹，那只有“三带一”这一种合法组合。
        if sorted(counts.values()) == [1, 3]:
            main_rank = next(rank for rank, count in counts.items() if count == 3)
            return Pattern("triple_single", main_rank, normalized, length)

    if length == 5 and sorted(counts.values()) == [2, 3]:
        main_rank = next(rank for rank, count in counts.items() if count == 3)
        return Pattern("triple_pair", main_rank, normalized, length)

    if length == 6 and 4 in counts.values():
        main_rank = next(rank for rank, count in counts.items() if count == 4)
        return Pattern("four_two_single", main_rank, normalized, length)

    if length == 8 and sorted(counts.values()) == [2, 2, 4]:
        main_rank = next(rank for rank, count in counts.items() if count == 4)
        return Pattern("four_two_pair", main_rank, normalized, length)

    if length >= 5 and _is_consecutive(ranks) and all(count == 1 for count in counts.values()):
        return Pattern("straight", ranks[-1], normalized, length)

    if length >= 6 and length % 2 == 0 and _is_consecutive(ranks) and all(count == 2 for count in counts.values()):
        return Pattern("pair_straight", ranks[-1], normalized, length)

    if length >= 6 and length % 3 == 0 and _is_consecutive(ranks) and all(count == 3 for count in counts.values()):
        return Pattern("plane", ranks[-1], normalized, length)

    if length >= 8:
        # 飞机带单的判断比裸飞机更复杂，所以拆到辅助函数里做。
        plane_single = _identify_plane_with_wings(counts, wing_size=1)
        if plane_single is not None:
            return Pattern("plane_single", plane_single, normalized, length)

    if length >= 10:
        plane_pair = _identify_plane_with_wings(counts, wing_size=2)
        if plane_pair is not None:
            return Pattern("plane_pair", plane_pair, normalized, length)

    return None


def find_patterns_from_hand(cards: Iterable[str]) -> list[Pattern]:
    """Enumerate every legal pattern that can be formed from one hand."""
    hand = normalize_cards(cards)
    counts = card_counter(hand)
    sorted_ranks = sorted(counts, key=lambda rank: RANK_VALUE[rank])
    unique_patterns: dict[tuple[str, tuple[str, ...]], Pattern] = {}

    def add_pattern(kind: str, main_rank: str, pattern_cards: list[str]) -> None:
        normalized = normalize_cards(pattern_cards)
        pattern = Pattern(kind, main_rank, normalized, len(normalized))
        unique_patterns[(pattern.kind, tuple(pattern.cards))] = pattern

    for rank in sorted_ranks:
        add_pattern("single", rank, [rank])
        if counts[rank] >= 2:
            add_pattern("pair", rank, [rank, rank])
        if counts[rank] >= 3:
            add_pattern("triple", rank, [rank, rank, rank])
        if counts[rank] == 4:
            add_pattern("bomb", rank, [rank, rank, rank, rank])

    if counts.get("BJ", 0) and counts.get("RJ", 0):
        add_pattern("rocket", "RJ", ["BJ", "RJ"])

    triple_ranks = [rank for rank in sorted_ranks if counts[rank] >= 3]
    pair_ranks = [rank for rank in sorted_ranks if counts[rank] >= 2]
    bomb_ranks = [rank for rank in sorted_ranks if counts[rank] == 4]

    for triple_rank in triple_ranks:
        for single_rank in sorted_ranks:
            if single_rank != triple_rank:
                add_pattern("triple_single", triple_rank, [triple_rank, triple_rank, triple_rank, single_rank])
        for pair_rank in pair_ranks:
            if pair_rank != triple_rank:
                add_pattern("triple_pair", triple_rank, [triple_rank, triple_rank, triple_rank, pair_rank, pair_rank])

    for bomb_rank in bomb_ranks:
        remainder = counts.copy()
        remainder[bomb_rank] -= 4
        if remainder[bomb_rank] == 0:
            del remainder[bomb_rank]

        for singles in _select_card_multisets(remainder, 2):
            add_pattern("four_two_single", bomb_rank, [bomb_rank, bomb_rank, bomb_rank, bomb_rank, *singles])

        pair_choices = [rank for rank, count in remainder.items() if count >= 2]
        for pair_ranks_choice in _select_rank_combinations(pair_choices, 2):
            wing_cards = [rank for rank in pair_ranks_choice for _ in range(2)]
            add_pattern("four_two_pair", bomb_rank, [bomb_rank, bomb_rank, bomb_rank, bomb_rank, *wing_cards])

    straight_ranks = [rank for rank in SEQUENCE_RANKS if counts.get(rank, 0) >= 1]
    for run in _consecutive_runs(straight_ranks):
        if len(run) >= 5:
            add_pattern("straight", run[-1], list(run))

    pair_straight_ranks = [rank for rank in SEQUENCE_RANKS if counts.get(rank, 0) >= 2]
    for run in _consecutive_runs(pair_straight_ranks):
        if len(run) >= 3:
            add_pattern("pair_straight", run[-1], [rank for rank in run for _ in range(2)])

    plane_ranks = [rank for rank in SEQUENCE_RANKS if counts.get(rank, 0) >= 3]
    for run in _consecutive_runs(plane_ranks):
        plane_cards = [rank for rank in run for _ in range(3)]
        add_pattern("plane", run[-1], plane_cards)

        remainder = counts.copy()
        for rank in run:
            remainder[rank] -= 3
            if remainder[rank] == 0:
                del remainder[rank]

        for singles in _select_card_multisets(remainder, len(run)):
            add_pattern("plane_single", run[-1], plane_cards + list(singles))

        pair_choices = [rank for rank, count in remainder.items() if count >= 2]
        for pair_ranks_choice in _select_rank_combinations(pair_choices, len(run)):
            wing_cards = [rank for rank in pair_ranks_choice for _ in range(2)]
            add_pattern("plane_pair", run[-1], plane_cards + wing_cards)

    return sorted(
        unique_patterns.values(),
        key=lambda pattern: (
            pattern.strength,
            pattern.length,
            RANK_VALUE[pattern.main_rank],
            pattern.cards,
        ),
    )


def _is_consecutive(ranks: list[str]) -> bool:
    """Return whether rank labels form a valid consecutive sequence."""
    # 这里的输入是“去重点数列表”，例如 ["5", "6", "7"]。
    if len(ranks) < 2:
        return False
    if any(rank not in SEQUENCE_RANKS for rank in ranks):
        return False
    values = [RANK_VALUE[rank] for rank in ranks]
    return all(values[index] + 1 == values[index + 1] for index in range(len(values) - 1))


def _identify_plane_with_wings(counts: Counter[str], wing_size: int) -> Optional[str]:
    """Detect a plane-with-wings pattern and return its highest triple rank."""
    # A plane is one or more consecutive triples plus "wings".
    # wing_size == 1 means single-card wings; wing_size == 2 means pair wings.
    triple_ranks = sorted(
        [rank for rank, count in counts.items() if count >= 3 and rank in SEQUENCE_RANKS],
        key=lambda rank: RANK_VALUE[rank],
    )
    if len(triple_ranks) < 2:
        return None

    for run in _consecutive_runs(triple_ranks):
        plane_size = len(run)
        required_length = plane_size * (3 + wing_size)
        # 例如两连飞机带两单，总长度必须是 2 * (3 + 1) = 8。
        if sum(counts.values()) != required_length:
            continue

        remainder = counts.copy()
        # 先把连续三张主体从计数里扣掉，剩余部分再检查是否满足“翅膀”条件。
        for rank in run:
            remainder[rank] -= 3
            if remainder[rank] == 0:
                del remainder[rank]

        if wing_size == 1 and sum(remainder.values()) == plane_size:
            return run[-1]

        if wing_size == 2:
            # 飞机带对要求剩余部分恰好是同样数量的对子。
            pair_ranks = [rank for rank, count in remainder.items() if count == 2]
            if len(pair_ranks) == plane_size and sum(remainder.values()) == plane_size * 2:
                return run[-1]

    return None


def _consecutive_runs(ranks: list[str]) -> list[list[str]]:
    """Generate all consecutive runs of length at least two from sorted ranks."""
    # We generate every consecutive run so later checks can try the longest
    # sequence first, but still fall back to shorter valid planes if needed.
    runs: list[list[str]] = []
    current_run: list[str] = []

    for rank in ranks:
        if not current_run or RANK_VALUE[rank] == RANK_VALUE[current_run[-1]] + 1:
            current_run.append(rank)
        else:
            if len(current_run) >= 2:
                runs.extend(_all_subruns(current_run))
            current_run = [rank]

    if len(current_run) >= 2:
        runs.extend(_all_subruns(current_run))

    # 让更长、结尾更大的连续段优先被尝试，能减少误判并更贴近出牌习惯。
    runs.sort(key=lambda run: (len(run), RANK_VALUE[run[-1]]), reverse=True)
    return runs


def _all_subruns(run: list[str]) -> list[list[str]]:
    """Expand one consecutive run into every valid subrun of length >= 2."""
    # 例如 ["5", "6", "7", "8"] 会生成
    # ["5","6"], ["5","6","7"], ["5","6","7","8"], ["6","7"], ...
    subruns: list[list[str]] = []
    for start in range(len(run)):
        for end in range(start + 2, len(run) + 1):
            subruns.append(run[start:end])
    return subruns


def _select_card_multisets(counts: Counter[str], total_cards: int) -> list[tuple[str, ...]]:
    """Choose every multiset of cards of a fixed size from a rank counter."""
    if total_cards == 0:
        return [()]

    ranks = sorted((rank for rank, count in counts.items() if count > 0), key=lambda rank: RANK_VALUE[rank])
    results: list[tuple[str, ...]] = []

    def backtrack(index: int, remaining: int, current: list[str]) -> None:
        if remaining == 0:
            results.append(tuple(current))
            return
        if index >= len(ranks):
            return

        rank = ranks[index]
        max_take = min(counts[rank], remaining)
        for take in range(max_take + 1):
            if take:
                current.extend([rank] * take)
            backtrack(index + 1, remaining - take, current)
            if take:
                del current[-take:]

    backtrack(0, total_cards, [])
    return results


def _select_rank_combinations(ranks: list[str], size: int) -> list[tuple[str, ...]]:
    """Choose distinct ranks without repetition."""
    if size == 0:
        return [()]
    results: list[tuple[str, ...]] = []

    def backtrack(start: int, current: list[str]) -> None:
        if len(current) == size:
            results.append(tuple(current))
            return
        for index in range(start, len(ranks)):
            current.append(ranks[index])
            backtrack(index + 1, current)
            current.pop()

    backtrack(0, [])
    return results
