from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ddz.cards import normalize_cards, sort_cards
from ddz.generator import generate_legal_plays
from ddz.simulator import DealResult, deal_new_game
from ddz.state import HistoryRecord, Play


@dataclass(frozen=True)
class EnvObservation:
    # 给外部调用方看的环境快照。
    # 这里直接暴露了三家手牌，适合训练/调试；如果以后做真实博弈，
    # 可能需要改成“只暴露当前玩家可见信息”的版本。
    current_player: int
    landlord: int
    hands: list[list[str]]
    hand_counts: list[int]
    last_play: Optional[Play]
    lead_player: int
    consecutive_passes: int
    winner: Optional[int]
    play_history: tuple[HistoryRecord, ...]


@dataclass(frozen=True)
class StepResult:
    # 类似强化学习环境常见的 step() 返回值。
    observation: EnvObservation
    reward: list[float]
    done: bool
    info: dict[str, object]


class DoudizhuEnv:
    def __init__(self) -> None:
        """Initialize an empty environment that must be reset before use."""
        # These fields are the mutable game state used by reset/step.
        self._deal: Optional[DealResult] = None
        self._hands: list[list[str]] = []
        self._current_player = 0
        self._lead_player = 0
        self._last_play: Optional[Play] = None
        self._consecutive_passes = 0
        self._winner: Optional[int] = None
        self._play_history: list[HistoryRecord] = []

    def reset(self, seed: int | None = None, landlord: int | None = None) -> EnvObservation:
        """Start a new game and return the first observation snapshot."""
        # reset() starts a fresh game and returns the first observable state.
        self._deal = deal_new_game(seed=seed, landlord=landlord)
        self._hands = [list(hand) for hand in self._deal.hands]
        self._current_player = self._deal.landlord
        self._lead_player = self._deal.landlord
        self._last_play = None
        self._consecutive_passes = 0
        self._winner = None
        self._play_history = []
        return self.observation()

    def observation(self) -> EnvObservation:
        """Return the current public snapshot of the environment state."""
        if self._deal is None:
            raise RuntimeError("Environment has not been reset.")
        # We expose a snapshot object instead of the raw mutable fields.
        return EnvObservation(
            current_player=self._current_player,
            landlord=self._deal.landlord,
            hands=[sort_cards(hand) for hand in self._hands],
            hand_counts=[len(hand) for hand in self._hands],
            last_play=self._last_play,
            lead_player=self._lead_player,
            consecutive_passes=self._consecutive_passes,
            winner=self._winner,
            play_history=tuple(self._play_history),
        )

    def legal_actions(self) -> list[Optional[list[str]]]:
        """List all actions available to the current player right now."""
        self._ensure_ready()
        assert self._deal is not None

        legal_patterns = generate_legal_plays(
            self._hands[self._current_player],
            self._last_play.cards if self._last_play else None,
        )
        actions = [pattern.cards for pattern in legal_patterns]
        # Passing is only legal when following someone else's play.
        # 领出阶段不能“过”，因为这一轮必须有人先把牌带起来。
        if self._last_play is not None:
            return [None, *actions]
        return actions

    def step(self, action: Optional[list[str]]) -> StepResult:
        """Apply one action, advance the game, and return the new state."""
        self._ensure_ready()
        assert self._deal is not None

        if self._winner is not None:
            raise RuntimeError("Game is already finished.")

        legal_actions = self.legal_actions()
        normalized_action = None if action is None else normalize_cards(action)
        if not self._contains_action(legal_actions, normalized_action):
            raise ValueError("Action is not legal in the current state.")

        info: dict[str, object] = {"player": self._current_player}
        rewards = [0.0, 0.0, 0.0]

        if normalized_action is None:
            self._consecutive_passes += 1
            info["is_pass"] = True
            self._play_history.append(HistoryRecord(player=self._current_player, cards=[], is_pass=True))
            # After two passes, the previous lead wins the trick and starts fresh.
            # 例如 A 出牌，B 过，C 过，那么下一手重新轮到 A 自由领出。
            if self._consecutive_passes == 2:
                self._current_player = self._lead_player
                self._last_play = None
                self._consecutive_passes = 0
            else:
                self._current_player = (self._current_player + 1) % 3
            return StepResult(observation=self.observation(), reward=rewards, done=False, info=info)

        self._play_cards(self._current_player, normalized_action)
        self._last_play = Play(cards=normalized_action, player=self._current_player)
        self._play_history.append(
            HistoryRecord(player=self._current_player, cards=list(normalized_action), is_pass=False)
        )
        # 本手一旦成功打出，当前玩家就成为新的 lead_player。
        self._lead_player = self._current_player
        self._consecutive_passes = 0
        info["is_pass"] = False
        info["cards"] = normalized_action

        if not self._hands[self._current_player]:
            self._winner = self._current_player
            rewards = self._terminal_rewards(self._winner, self._deal.landlord)
            return StepResult(observation=self.observation(), reward=rewards, done=True, info=info)

        self._current_player = (self._current_player + 1) % 3
        return StepResult(observation=self.observation(), reward=rewards, done=False, info=info)

    def _ensure_ready(self) -> None:
        """Raise an error if reset() has not been called yet."""
        if self._deal is None:
            raise RuntimeError("Environment has not been reset.")

    def _play_cards(self, player: int, cards: list[str]) -> None:
        """Remove the chosen cards from one player's hand in place."""
        # Remove exact card instances from the selected player's hand.
        for card in cards:
            self._hands[player].remove(card)

    @staticmethod
    def _contains_action(actions: list[Optional[list[str]]], candidate: Optional[list[str]]) -> bool:
        """Check whether a normalized candidate action is in the legal set."""
        # 把 list 归一化成 tuple 后再比较，便于放入 set 做精确匹配。
        normalized_actions = {None if action is None else tuple(action) for action in actions}
        normalized_candidate = None if candidate is None else tuple(candidate)
        return normalized_candidate in normalized_actions

    @staticmethod
    def _terminal_rewards(winner: int, landlord: int) -> list[float]:
        """Return simple terminal rewards for landlord-versus-farmers outcome."""
        # First reward version:
        # landlord wins => landlord +1, both farmers -1
        # farmers win => landlord -1, both farmers +1
        if winner == landlord:
            return [1.0 if index == landlord else -1.0 for index in range(3)]
        return [-1.0 if index == landlord else 1.0 for index in range(3)]
