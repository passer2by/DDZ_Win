from __future__ import annotations

from typing import Optional

from ddz.patterns import Pattern, find_patterns_from_hand
from ddz.rules import compare_patterns


def generate_legal_plays(hand: list[str], last_play_cards: Optional[list[str]]) -> list[Pattern]:
    """List every legal pattern the current hand may play this turn."""
    # 生成当前手牌在当前回合下的所有合法出法。
    # 如果是领出（last_play_cards is None），所有可识别牌型都合法；
    # 如果是跟牌，则只保留能压过上家的那些牌型。
    candidates = find_patterns_from_hand(hand)
    if last_play_cards is None:
        return candidates

    from ddz.patterns import identify_pattern

    target = identify_pattern(last_play_cards)
    if target is None:
        raise ValueError("Last play is not a valid pattern.")

    # compare_patterns 已经封装了炸弹 / 王炸等特殊规则。
    legal = [pattern for pattern in candidates if compare_patterns(pattern, target) > 0]
    return legal
