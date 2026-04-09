from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Play:
    cards: list[str]
    player: int = 0


@dataclass
class GameState:
    my_hand: list[str]
    last_play: Optional[Play]
    my_role: str
    current_player: int = 0
    teammate_cards_left: Optional[int] = None
    left_enemy_cards_left: Optional[int] = None
    right_enemy_cards_left: Optional[int] = None
