from ddz.rules import can_beat


def test_bomb_beats_non_bomb() -> None:
    assert can_beat(["9", "9", "9", "9"], ["A"])


def test_rocket_beats_bomb() -> None:
    assert can_beat(["BJ", "RJ"], ["9", "9", "9", "9"])


def test_different_shape_cannot_beat() -> None:
    assert not can_beat(["6", "6", "6", "7", "7", "7", "9", "9", "J", "J"], ["8", "8", "8"])
