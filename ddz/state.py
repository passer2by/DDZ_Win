from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Play:
    # cards 是已经打出的具体牌组，player 表示是谁出的这手牌。
    cards: list[str]
    player: int = 0


@dataclass(frozen=True)
class HistoryRecord:
    """One historical action in chronological order, including passes."""

    player: int
    cards: list[str]
    is_pass: bool


@dataclass
class GameState:
    # 这是策略层看到的“单回合决策输入”，不是完整对局状态。
    # 它故意保持轻量，方便 CLI、模拟器、后续训练代码复用。
    my_hand: list[str]
    last_play: Optional[Play]
    # my_role 目前只区分 landlord / farmer，用于决定进攻或配合倾向。
    my_role: str
    current_player: int = 0
    # 这些剩余张数是启发式策略的重要上下文：
    # 谁快跑完、队友是否快出完，都会影响是否要压牌或保炸弹。
    teammate_cards_left: Optional[int] = None
    left_enemy_cards_left: Optional[int] = None
    right_enemy_cards_left: Optional[int] = None
    play_history: tuple[HistoryRecord, ...] = ()
