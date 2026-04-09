from ddz.cards import normalize_cards
from ddz.state import GameState, Play
from ddz.strategy import recommend_play


def test_recommend_bomb_when_opponent_is_critical() -> None:
    state = GameState(
        my_hand=normalize_cards("4 4 4 4 7 8".split()),
        last_play=Play(cards=["3", "3"]),
        my_role="farmer",
        left_enemy_cards_left=1,
    )
    recommendation = recommend_play(state)
    assert recommendation is not None
    assert recommendation.pattern.kind == "bomb"
