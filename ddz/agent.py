from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from ddz.cards import sort_cards
from ddz.generator import generate_legal_plays
from ddz.state import GameState, HistoryRecord, Play
from ddz.strategy import recommend_play


@dataclass(frozen=True)
class TurnView:
    # TurnView is the minimal information an agent needs to decide a move.
    # 这是模拟器传给 Agent 的“观察视角”，只包含当前玩家合法决策需要的信息。
    player: int
    landlord: int
    hand: list[str]
    hand_counts: list[int]
    last_play: Optional[Play]
    play_history: tuple[HistoryRecord, ...] = ()


class Agent(Protocol):
    # 任何实现了 choose_play(view) 的对象，都可以接入模拟器对战。
    def choose_play(self, view: TurnView) -> Optional[list[str]]:
        """Choose a list of cards to play, or None to pass."""
        ...


class HeuristicAgent:
    def choose_play(self, view: TurnView) -> Optional[list[str]]:
        """Adapt the heuristic recommender to the simulator agent interface."""
        # Reuse the current rule-based recommender as a drop-in player.
        role = "landlord" if view.player == view.landlord else "farmer"
        teammate_cards_left = None
        if role == "farmer":
            # 农民的“队友”是另外一个非地主玩家。
            teammate = next(index for index in range(3) if index not in {view.player, view.landlord})
            teammate_cards_left = view.hand_counts[teammate]

        # TurnView -> GameState，相当于把模拟器视角转换成策略模块需要的输入格式。
        state = GameState(
            my_hand=sort_cards(view.hand),
            last_play=view.last_play,
            my_role=role,
            current_player=view.player,
            teammate_cards_left=teammate_cards_left,
            left_enemy_cards_left=view.hand_counts[(view.player + 1) % 3],
            right_enemy_cards_left=view.hand_counts[(view.player + 2) % 3],
            play_history=view.play_history,
        )
        recommendation = recommend_play(state)
        if recommendation is None:
            return None
        return recommendation.pattern.cards


class FirstLegalAgent:
    def choose_play(self, view: TurnView) -> Optional[list[str]]:
        """Pick the first legal move, mainly for tests and smoke simulations."""
        # This agent is intentionally simple and is mainly useful for testing.
        # 它不做任何策略判断，只拿“合法列表中的第一手”。
        legal_plays = generate_legal_plays(view.hand, view.last_play.cards if view.last_play else None)
        if not legal_plays:
            return None
        return legal_plays[0].cards
