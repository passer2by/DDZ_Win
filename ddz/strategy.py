from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ddz.cards import RANK_VALUE, card_counter, sort_cards
from ddz.generator import generate_legal_plays
from ddz.patterns import Pattern
from ddz.state import GameState


@dataclass(frozen=True)
class Recommendation:
    pattern: Pattern
    score: float
    reasons: list[str]


def recommend_play(state: GameState) -> Optional[Recommendation]:
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
            -RANK_VALUE[item.pattern.main_rank],
        ),
        reverse=True,
    )
    return recommendations[0]


def _score_play(state: GameState, pattern: Pattern) -> Recommendation:
    score = 0.0
    reasons: list[str] = []

    # Larger plays usually reduce the number of future turns we need.
    score += pattern.length * 6
    reasons.append(f"Play {pattern.length} cards at once to reduce future turns.")

    # Different pattern types have different strategic values.
    kind_bonus_map = {
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
    score -= remaining_groups * 1.5

    high_card_penalty = sum(2 for card in pattern.cards if RANK_VALUE[card] >= RANK_VALUE["A"])
    if high_card_penalty:
        score -= high_card_penalty * 3
        reasons.append("The move spends high-value cards, so it gets a conservative penalty.")

    if pattern.kind in {"bomb", "rocket"} and _opponent_is_dangerous(state):
        score += 25
        reasons.append("An opponent is close to going out, so blocking value becomes much higher.")
    elif _opponent_is_dangerous(state) and pattern.length >= 2:
        score += 10
        reasons.append("Pressure matters more because an opponent has very few cards left.")

    if state.my_role == "landlord":
        score += 4
        reasons.append("Landlord play is a bit more proactive about cashing out the hand.")
    elif state.my_role == "farmer" and state.teammate_cards_left is not None and state.teammate_cards_left <= 3:
        if pattern.kind in {"single", "pair"}:
            score -= 8
            reasons.append("As farmer, avoid stealing tempo when your teammate is nearly out.")

    return Recommendation(pattern=pattern, score=round(score, 2), reasons=reasons[:3])


def _remaining_hand(hand: list[str], played_cards: list[str]) -> list[str]:
    # Remove one matching card at a time so duplicate ranks are handled correctly.
    remaining = list(hand)
    for card in played_cards:
        remaining.remove(card)
    return sort_cards(remaining)


def _opponent_is_dangerous(state: GameState) -> bool:
    enemy_counts = [count for count in [state.left_enemy_cards_left, state.right_enemy_cards_left] if count is not None]
    return any(count <= 2 for count in enemy_counts)
