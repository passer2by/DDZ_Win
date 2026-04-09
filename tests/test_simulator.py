from ddz.agent import FirstLegalAgent, HeuristicAgent
from ddz.simulator import deal_new_game, simulate_game, simulate_many_games


def test_deal_produces_expected_hand_sizes() -> None:
    deal = deal_new_game(seed=7, landlord=1)
    assert len(deal.hands) == 3
    assert len(deal.bottom_cards) == 3
    assert [len(hand) for hand in deal.hands] == [17, 20, 17]


def test_simulate_game_finishes_with_empty_winner_hand() -> None:
    result = simulate_game(agents=[FirstLegalAgent(), FirstLegalAgent(), FirstLegalAgent()], seed=3, landlord=0)
    assert result.winner in {0, 1, 2}
    assert result.final_hands[result.winner] == []
    assert len(result.turns) > 0


def test_simulate_many_games_returns_summary() -> None:
    summary = simulate_many_games(5, agents=[HeuristicAgent(), HeuristicAgent(), HeuristicAgent()], seed=11)
    assert summary["games"] == 5
    assert summary["landlord_wins"] + summary["farmer_wins"] == 5
