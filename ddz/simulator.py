from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ddz.agent import Agent, HeuristicAgent, TurnView
from ddz.cards import normalize_cards, shuffled_deck, sort_cards
from ddz.generator import generate_legal_plays
from ddz.state import Play


@dataclass(frozen=True)
class TurnRecord:
    player: int
    cards: list[str]
    is_pass: bool


@dataclass(frozen=True)
class DealResult:
    hands: list[list[str]]
    bottom_cards: list[str]
    landlord: int


@dataclass(frozen=True)
class SimulationResult:
    winner: int
    landlord: int
    turns: list[TurnRecord]
    final_hands: list[list[str]]


def deal_new_game(seed: int | None = None, landlord: int | None = None) -> DealResult:
    # Standard Dou Dizhu dealing:
    # 17 cards for each player and 3 bottom cards for the landlord.
    deck = shuffled_deck(seed)
    hands = [sort_cards(deck[index * 17 : (index + 1) * 17]) for index in range(3)]
    bottom_cards = sort_cards(deck[51:])
    landlord_index = 0 if landlord is None else landlord
    hands[landlord_index] = sort_cards(hands[landlord_index] + bottom_cards)
    return DealResult(hands=hands, bottom_cards=bottom_cards, landlord=landlord_index)


def simulate_game(
    agents: list[Agent] | None = None,
    seed: int | None = None,
    landlord: int | None = None,
) -> SimulationResult:
    # This is the simplest full-game loop in the project.
    # It lets three agents play until one hand becomes empty.
    players = agents or [HeuristicAgent(), HeuristicAgent(), HeuristicAgent()]
    if len(players) != 3:
        raise ValueError("Exactly three agents are required.")

    deal = deal_new_game(seed=seed, landlord=landlord)
    hands = [list(hand) for hand in deal.hands]
    current_player = deal.landlord
    lead_player = deal.landlord
    last_play: Optional[Play] = None
    consecutive_passes = 0
    turns: list[TurnRecord] = []

    while True:
        hand_counts = [len(hand) for hand in hands]
        chosen_cards = _choose_play(
            agent=players[current_player],
            player=current_player,
            landlord=deal.landlord,
            hands=hands,
            hand_counts=hand_counts,
            last_play=last_play,
        )

        if chosen_cards is None:
            turns.append(TurnRecord(player=current_player, cards=[], is_pass=True))
            consecutive_passes += 1
            # Two passes after a lead means the trick is cleared
            # and the lead player starts a fresh round.
            if consecutive_passes == 2:
                current_player = lead_player
                last_play = None
                consecutive_passes = 0
                continue
            current_player = (current_player + 1) % 3
            continue

        _remove_cards(hands[current_player], chosen_cards)
        normalized = normalize_cards(chosen_cards)
        last_play = Play(cards=normalized, player=current_player)
        lead_player = current_player
        consecutive_passes = 0
        turns.append(TurnRecord(player=current_player, cards=normalized, is_pass=False))

        if not hands[current_player]:
            return SimulationResult(
                winner=current_player,
                landlord=deal.landlord,
                turns=turns,
                final_hands=[sort_cards(hand) for hand in hands],
            )

        current_player = (current_player + 1) % 3


def simulate_many_games(
    num_games: int,
    agents: list[Agent] | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    # Batch simulation is the bridge to evaluation and later training.
    landlord_wins = 0
    farmer_wins = 0
    per_player_wins = [0, 0, 0]
    results: list[SimulationResult] = []

    for index in range(num_games):
        game_seed = None if seed is None else seed + index
        result = simulate_game(agents=agents, seed=game_seed, landlord=index % 3)
        results.append(result)
        per_player_wins[result.winner] += 1
        if result.winner == result.landlord:
            landlord_wins += 1
        else:
            farmer_wins += 1

    return {
        "games": num_games,
        "landlord_wins": landlord_wins,
        "farmer_wins": farmer_wins,
        "per_player_wins": per_player_wins,
        "results": results,
    }


def _choose_play(
    agent: Agent,
    player: int,
    landlord: int,
    hands: list[list[str]],
    hand_counts: list[int],
    last_play: Optional[Play],
) -> Optional[list[str]]:
    legal_plays = generate_legal_plays(hands[player], last_play.cards if last_play else None)
    if not legal_plays:
        return None

    view = TurnView(
        player=player,
        landlord=landlord,
        hand=sort_cards(hands[player]),
        hand_counts=hand_counts,
        last_play=last_play,
    )
    chosen = agent.choose_play(view)
    if chosen is None:
        return None

    normalized = normalize_cards(chosen)
    legal_card_sets = {tuple(pattern.cards) for pattern in legal_plays}
    # If an agent returns an illegal move, fall back to a safe legal action
    # so simulation can continue instead of crashing.
    if tuple(normalized) not in legal_card_sets:
        return legal_plays[0].cards
    return normalized


def _remove_cards(hand: list[str], cards: list[str]) -> None:
    for card in cards:
        hand.remove(card)
