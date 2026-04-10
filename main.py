from __future__ import annotations

from ddz.cards import normalize_cards
from ddz.state import GameState, Play
from ddz.strategy import recommend_play


def main() -> None:
    """Run the interactive CLI and print one heuristic recommendation."""
    print("=== Dou Dizhu Decision Helper MVP ===")
    hand = _read_cards("Your hand (space separated, e.g. 3 3 4 4 5 6 BJ RJ): ")
    last_raw = input("Last play (press Enter if you lead this round): ").strip()
    role = input("Your role (landlord/farmer, default farmer): ").strip().lower() or "farmer"
    left_enemy = _read_optional_int("Left opponent cards remaining (optional): ")
    right_enemy = _read_optional_int("Right opponent cards remaining (optional): ")
    teammate = _read_optional_int("Teammate cards remaining (optional for farmer): ")

    last_play = Play(cards=_read_cards_from_text(last_raw)) if last_raw else None
    state = GameState(
        my_hand=hand,
        last_play=last_play,
        my_role=role,
        left_enemy_cards_left=left_enemy,
        right_enemy_cards_left=right_enemy,
        teammate_cards_left=teammate,
    )

    recommendation = recommend_play(state)
    if recommendation is None:
        print("No legal play found. Suggested action: pass.")
        return

    # 当前 CLI 只是一个很轻量的调试入口：
    # 输入你的手牌和局面，输出启发式策略的推荐结果与原因。
    print("\nRecommended play:", " ".join(recommendation.pattern.cards))
    print("Pattern:", recommendation.pattern.kind)
    print("Score:", recommendation.score)
    print("Reasons:")
    for index, reason in enumerate(recommendation.reasons, start=1):
        print(f"{index}. {reason}")


def _read_cards(prompt: str) -> list[str]:
    """Prompt once and parse the response as a normalized card list."""
    return _read_cards_from_text(input(prompt))


def _read_cards_from_text(raw_text: str) -> list[str]:
    """Parse one line of text into normalized card tokens."""
    # 同时兼容空格和逗号分隔，便于手输。
    return normalize_cards(raw_text.replace(",", " ").split())


def _read_optional_int(prompt: str) -> int | None:
    """Prompt for an optional integer and return None when left blank."""
    raw = input(prompt).strip()
    return int(raw) if raw else None


if __name__ == "__main__":
    main()
