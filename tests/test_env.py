from ddz.env import DoudizhuEnv


def test_env_reset_sets_expected_opening_state() -> None:
    env = DoudizhuEnv()
    observation = env.reset(seed=13, landlord=2)
    assert observation.current_player == 2
    assert observation.landlord == 2
    assert observation.last_play is None
    assert observation.hand_counts == [17, 17, 20]


def test_env_legal_actions_include_pass_when_following() -> None:
    env = DoudizhuEnv()
    env.reset(seed=1, landlord=0)
    first_action = env.legal_actions()[0]
    result = env.step(first_action)
    assert not result.done
    legal_actions = env.legal_actions()
    assert legal_actions[0] is None


def test_env_terminal_reward_matches_farmer_win() -> None:
    env = DoudizhuEnv()
    env.reset(seed=2, landlord=0)
    env._hands = [["3"], ["4"], ["5"]]  # type: ignore[attr-defined]
    env._current_player = 1  # type: ignore[attr-defined]
    env._lead_player = 1  # type: ignore[attr-defined]
    env._last_play = None  # type: ignore[attr-defined]

    result = env.step(["4"])
    assert result.done
    assert result.observation.winner == 1
    assert result.reward == [-1.0, 1.0, 1.0]


def test_env_observation_tracks_play_history() -> None:
    env = DoudizhuEnv()
    env.reset(seed=1, landlord=0)
    first_action = env.legal_actions()[0]
    result = env.step(first_action)

    assert len(result.observation.play_history) == 1
    assert result.observation.play_history[0].player == 0
    assert result.observation.play_history[0].is_pass is False
