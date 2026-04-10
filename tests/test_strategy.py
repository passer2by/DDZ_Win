from ddz.cards import normalize_cards
from ddz.state import GameState, HistoryRecord, Play
from ddz.strategy import _infer_player_threat_memories, _infer_threat_memory, recommend_play


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


def test_infer_threat_memory_tracks_unseen_bombs_and_rocket() -> None:
    state = GameState(
        my_hand=normalize_cards("BJ 7 7".split()),
        last_play=None,
        my_role="landlord",
        play_history=(
            HistoryRecord(player=1, cards=["RJ"], is_pass=False),
            HistoryRecord(player=2, cards=["4", "4"], is_pass=False),
        ),
    )

    memory = _infer_threat_memory(state)
    assert memory.rocket_possible is False
    assert "4" not in memory.bomb_ranks
    assert "5" in memory.bomb_ranks


def test_memory_makes_long_premium_play_safer_when_no_enemy_threat_remains() -> None:
    base_state = GameState(
        my_hand=normalize_cards("4 4 4 4 7 7".split()),
        last_play=None,
        my_role="landlord",
    )
    remembered_state = GameState(
        my_hand=normalize_cards("4 4 4 4 7 7".split()),
        last_play=None,
        my_role="landlord",
        play_history=(
            HistoryRecord(player=1, cards=["BJ", "RJ"], is_pass=False),
            HistoryRecord(player=2, cards=["3", "3", "3", "3"], is_pass=False),
            HistoryRecord(player=1, cards=["5", "5", "5", "5"], is_pass=False),
            HistoryRecord(player=2, cards=["6", "6", "6", "6"], is_pass=False),
            HistoryRecord(player=1, cards=["7", "7"], is_pass=False),
            HistoryRecord(player=2, cards=["8", "8", "8", "8"], is_pass=False),
            HistoryRecord(player=1, cards=["9", "9", "9", "9"], is_pass=False),
            HistoryRecord(player=2, cards=["10", "10", "10", "10"], is_pass=False),
            HistoryRecord(player=1, cards=["J", "J", "J", "J"], is_pass=False),
            HistoryRecord(player=2, cards=["Q", "Q", "Q", "Q"], is_pass=False),
            HistoryRecord(player=1, cards=["K", "K", "K", "K"], is_pass=False),
            HistoryRecord(player=2, cards=["A", "A", "A", "A"], is_pass=False),
            HistoryRecord(player=1, cards=["2", "2", "2", "2"], is_pass=False),
        ),
    )

    base_recommendation = recommend_play(base_state)
    remembered_recommendation = recommend_play(remembered_state)

    assert base_recommendation is not None
    assert remembered_recommendation is not None
    assert remembered_recommendation.pattern.kind == "four_two_single"
    assert remembered_recommendation.score > base_recommendation.score


def test_infer_player_threat_memories_is_seat_specific() -> None:
    state = GameState(
        my_hand=normalize_cards("BJ 7 7".split()),
        last_play=None,
        my_role="landlord",
        current_player=0,
        left_enemy_cards_left=4,
        right_enemy_cards_left=1,
        play_history=(
            HistoryRecord(player=1, cards=["4", "4"], is_pass=False),
            HistoryRecord(player=2, cards=["RJ"], is_pass=False),
        ),
    )

    memories = _infer_player_threat_memories(state)
    left_memory = memories[1]
    right_memory = memories[2]

    assert left_memory.cards_remaining == 4
    assert right_memory.cards_remaining == 1
    assert left_memory.rocket_possible is False
    assert right_memory.rocket_possible is False
    assert "5" in left_memory.bomb_ranks
    assert right_memory.bomb_ranks == ()
