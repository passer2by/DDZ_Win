from __future__ import annotations

from ddz.simulator import simulate_many_games


def main() -> None:
    summary = simulate_many_games(num_games=20, seed=42)
    print("games:", summary["games"])
    print("landlord_wins:", summary["landlord_wins"])
    print("farmer_wins:", summary["farmer_wins"])
    print("per_player_wins:", summary["per_player_wins"])


if __name__ == "__main__":
    main()
