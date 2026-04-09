from ddz.patterns import identify_pattern


def test_identify_four_with_two_singles() -> None:
    pattern = identify_pattern(["7", "7", "7", "7", "9", "J"])
    assert pattern is not None
    assert pattern.kind == "four_two_single"
    assert pattern.main_rank == "7"


def test_identify_four_with_two_pairs() -> None:
    pattern = identify_pattern(["8", "8", "8", "8", "4", "4", "6", "6"])
    assert pattern is not None
    assert pattern.kind == "four_two_pair"
    assert pattern.main_rank == "8"


def test_identify_plane_with_single_wings() -> None:
    pattern = identify_pattern(["4", "4", "4", "5", "5", "5", "8", "9"])
    assert pattern is not None
    assert pattern.kind == "plane_single"
    assert pattern.main_rank == "5"


def test_identify_plane_with_pair_wings() -> None:
    pattern = identify_pattern(["6", "6", "6", "7", "7", "7", "9", "9", "J", "J"])
    assert pattern is not None
    assert pattern.kind == "plane_pair"
    assert pattern.main_rank == "7"
