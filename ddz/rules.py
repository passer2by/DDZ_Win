from __future__ import annotations

from typing import Iterable, Optional

from ddz.cards import RANK_VALUE
from ddz.patterns import Pattern, identify_pattern


def can_beat(candidate_cards: Iterable[str], last_cards: Optional[Iterable[str]]) -> bool:
    """Check whether raw candidate cards can legally beat the previous play."""
    # 对外暴露的便捷函数：传原始牌列表即可判断能否压过上家。
    candidate = identify_pattern(candidate_cards)
    if candidate is None:
        return False
    if last_cards is None:
        return True
    target = identify_pattern(last_cards)
    if target is None:
        raise ValueError("Last play is not a valid pattern.")
    return compare_patterns(candidate, target) > 0


def compare_patterns(candidate: Pattern, target: Pattern) -> int:
    """Compare two recognized patterns using Dou Dizhu beating rules."""
    # 返回 1 / 0 / -1，含义分别是：
    # candidate > target / 相等 / 无法压过 target。
    #
    # 斗地主的核心例外规则只有两条：
    # 1. 王炸最大
    # 2. 炸弹可以压所有非炸弹牌型
    if candidate.kind == "rocket":
        return 1 if target.kind != "rocket" else 0
    if target.kind == "rocket":
        return -1

    if candidate.kind == "bomb" and target.kind != "bomb":
        return 1
    if target.kind == "bomb" and candidate.kind != "bomb":
        return -1

    # 普通牌型必须“同类型且同长度”才能比较主点数。
    if candidate.kind != target.kind or candidate.length != target.length:
        return -1

    candidate_value = RANK_VALUE[candidate.main_rank]
    target_value = RANK_VALUE[target.main_rank]
    if candidate_value > target_value:
        return 1
    if candidate_value == target_value:
        return 0
    return -1
