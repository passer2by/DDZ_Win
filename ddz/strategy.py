from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ddz.cards import FULL_DECK, RANK_ORDER, RANK_VALUE, card_counter, sort_cards
from ddz.generator import generate_legal_plays
from ddz.patterns import Pattern
from ddz.state import GameState


@dataclass(frozen=True)
class Recommendation:
    # 策略层最终给出的建议对象：
    # 出什么牌、这手牌打多少分、为什么推荐它。
    pattern: Pattern
    score: float
    reasons: list[str]


@dataclass(frozen=True)
class ThreatMemory:
    """Simple inferred memory about premium hidden power still outside our hand."""

    rocket_possible: bool
    bomb_ranks: tuple[str, ...]

    @property
    def premium_possible(self) -> bool:
        """Return whether opponents may still hold any bomb or rocket."""
        return self.rocket_possible or bool(self.bomb_ranks)


@dataclass(frozen=True)
class PlayerThreatMemory:
    """Seat-specific inferred premium threat for one other player."""

    player: int
    cards_remaining: Optional[int]
    rocket_possible: bool
    bomb_ranks: tuple[str, ...]
    shown_ranks: tuple[str, ...]

    @property
    def premium_possible(self) -> bool:
        """Return whether this player may still hold any bomb or rocket."""
        return self.rocket_possible or bool(self.bomb_ranks)


def recommend_play(state: GameState) -> Optional[Recommendation]:
    """Return the highest-scoring legal move for the given game state."""
    # Strategy works in two steps:
    # 1. generate every legal move for the current position
    # 2. score them and keep the highest-scoring one
    legal_plays = generate_legal_plays(state.my_hand, state.last_play.cards if state.last_play else None)
    if not legal_plays:
        return None

    recommendations = [_score_play(state, play) for play in legal_plays]
    recommendations.sort(
        key=lambda item: (
            item.score,
            item.pattern.strength,
            item.pattern.length,
            # 当前分数相同的情况下，倾向保留更小的主点数，
            # reverse=True 后这里取负值，等价于“优先更小的 main_rank”。
            -RANK_VALUE[item.pattern.main_rank],
        ),
        reverse=True,
    )
    return recommendations[0]


def _score_play(state: GameState, pattern: Pattern) -> Recommendation:
    """Assign a heuristic score and explanation to one legal candidate play."""
    # 这是当前项目的核心启发式策略：
    # 给每个合法出法打一个静态分，再选分数最高的。
    #
    # 它的优点是简单、可解释、容易调参；
    # 它的缺点是没有做多步搜索，因此更像 MVP 策略而不是强 AI。
    score = 0.0
    reasons: list[str] = []
    threat_memory = _infer_threat_memory(state)
    player_memories = _infer_player_threat_memories(state)
    next_player_memory = player_memories.get((state.current_player + 1) % 3)

    # Larger plays usually reduce the number of future turns we need.
    score += pattern.length * 6
    reasons.append(f"Play {pattern.length} cards at once to reduce future turns.")

    # Different pattern types have different strategic values.
    kind_bonus_map = {
        # 这里体现了当前作者对牌型价值的主观偏好：
        # 更完整、一次能走更多牌的组合通常分更高；
        # 炸弹和王炸默认扣分，表示“除非必要先别交”。
        "single": 4,
        "pair": 7,
        "triple": 10,
        "triple_single": 14,
        "triple_pair": 16,
        "straight": 20,
        "pair_straight": 24,
        "plane": 28,
        "plane_single": 30,
        "plane_pair": 32,
        "four_two_single": 18,
        "four_two_pair": 20,
        "bomb": -10,
        "rocket": -16,
    }
    score += kind_bonus_map[pattern.kind]

    if pattern.kind in {"straight", "pair_straight", "plane", "plane_single", "plane_pair"}:
        reasons.append("Prefer finished combinations because they usually improve hand structure.")
    elif pattern.kind in {"four_two_single", "four_two_pair"}:
        reasons.append("This turns a heavy four-of-a-kind into an efficient tempo play.")
    elif pattern.kind in {"bomb", "rocket"}:
        reasons.append("This is a premium resource, so the strategy saves it unless pressure is high.")

    remaining_hand = _remaining_hand(state.my_hand, pattern.cards)
    remaining_groups = len(card_counter(remaining_hand))
    # Fewer disconnected rank groups usually means the hand is easier to finish.
    # 这里把“剩余牌的分散程度”当成一个非常粗糙的手牌质量指标。
    # group 越多，说明后续可能越难连成组合，所以要扣分。
    score -= remaining_groups * 1.5

    # A、2、大小王这类高牌是关键资源，轻易打掉通常不划算。
    high_card_penalty = sum(2 for card in pattern.cards if RANK_VALUE[card] >= RANK_VALUE["A"])
    if high_card_penalty:
        score -= high_card_penalty * 3
        reasons.append("The move spends high-value cards, so it gets a conservative penalty.")

    if pattern.kind in {"bomb", "rocket"}:
        if threat_memory.premium_possible and not _opponent_is_dangerous(state):
            score -= 6
            reasons.append("Unseen bomb or rocket may still exist, so premium force is conserved.")
        elif not threat_memory.premium_possible:
            score += 8
            reasons.append("No unseen bomb or rocket remains, so premium force is safer to cash.")
    elif pattern.length >= 5 and not threat_memory.premium_possible:
        score += 6
        reasons.append("No unseen bomb or rocket remains, so long combinations are safer to push.")

    if next_player_memory is not None and pattern.length >= 5:
        if next_player_memory.premium_possible and not _opponent_is_dangerous(state):
            score -= 3
            reasons.append("Next player may still hide premium force, so the push stays slightly cautious.")
        elif not next_player_memory.premium_possible:
            score += 2
            reasons.append("Next player has little premium threat left, so tempo plays become a bit safer.")

    # 对手只剩很少牌时，策略会明显转向“拦截优先”。
    if pattern.kind in {"bomb", "rocket"} and _opponent_is_dangerous(state):
        score += 25
        reasons.append("An opponent is close to going out, so blocking value becomes much higher.")
    elif _opponent_is_dangerous(state) and pattern.length >= 2:
        score += 10
        reasons.append("Pressure matters more because an opponent has very few cards left.")

    if state.my_role == "landlord":
        # 地主没有队友，默认会更主动地推进出牌节奏。
        score += 4
        reasons.append("Landlord play is a bit more proactive about cashing out the hand.")
    elif state.my_role == "farmer" and state.teammate_cards_left is not None and state.teammate_cards_left <= 3:
        if pattern.kind in {"single", "pair"}:
            # 农民队友快走完时，不希望自己用小牌把出牌权硬抢回来。
            score -= 8
            reasons.append("As farmer, avoid stealing tempo when your teammate is nearly out.")

    return Recommendation(pattern=pattern, score=round(score, 2), reasons=reasons[:3])


def _remaining_hand(hand: list[str], played_cards: list[str]) -> list[str]:
    """Return the sorted cards left after removing one chosen play from hand."""
    # Remove one matching card at a time so duplicate ranks are handled correctly.
    remaining = list(hand)
    for card in played_cards:
        remaining.remove(card)
    return sort_cards(remaining)


def _opponent_is_dangerous(state: GameState) -> bool:
    """Check whether any tracked opponent is close to going out."""
    # 只要任意一个敌方玩家剩余牌数 <= 2，就认为局势进入高压阶段。
    enemy_counts = [count for count in [state.left_enemy_cards_left, state.right_enemy_cards_left] if count is not None]
    return any(count <= 2 for count in enemy_counts)


def _infer_threat_memory(state: GameState) -> ThreatMemory:
    """Infer whether unseen bombs or a rocket may still remain in opponents' hands."""
    seen_cards = list(state.my_hand)
    for record in state.play_history:
        seen_cards.extend(record.cards)

    unseen_counts = card_counter(FULL_DECK)
    for card in seen_cards:
        unseen_counts[card] -= 1

    rocket_possible = unseen_counts.get("BJ", 0) == 1 and unseen_counts.get("RJ", 0) == 1
    bomb_ranks = tuple(rank for rank in RANK_ORDER[:-2] if unseen_counts.get(rank, 0) == 4)
    return ThreatMemory(rocket_possible=rocket_possible, bomb_ranks=bomb_ranks)


def _infer_player_threat_memories(state: GameState) -> dict[int, PlayerThreatMemory]:
    """Infer seat-specific premium threats for the two players around us."""
    histories_by_player: dict[int, list[str]] = {}
    for record in state.play_history:
        histories_by_player.setdefault(record.player, []).extend(record.cards)

    unseen_counts = card_counter(FULL_DECK)
    for card in state.my_hand:
        unseen_counts[card] -= 1
    for record in state.play_history:
        for card in record.cards:
            unseen_counts[card] -= 1

    players = {
        (state.current_player + 1) % 3: state.left_enemy_cards_left,
        (state.current_player + 2) % 3: state.right_enemy_cards_left,
    }
    memories: dict[int, PlayerThreatMemory] = {}
    for player, cards_remaining in players.items():
        shown_cards = histories_by_player.get(player, [])
        shown_ranks = tuple(sorted(set(shown_cards), key=lambda rank: RANK_VALUE[rank]))
        can_hold_premium = cards_remaining is None or cards_remaining >= 2
        rocket_possible = (
            can_hold_premium
            and unseen_counts.get("BJ", 0) == 1
            and unseen_counts.get("RJ", 0) == 1
        )
        bomb_ranks = tuple(
            rank
            for rank in RANK_ORDER[:-2]
            if unseen_counts.get(rank, 0) == 4 and (cards_remaining is None or cards_remaining >= 4)
        )
        memories[player] = PlayerThreatMemory(
            player=player,
            cards_remaining=cards_remaining,
            rocket_possible=rocket_possible,
            bomb_ranks=bomb_ranks,
            shown_ranks=shown_ranks,
        )
    return memories
