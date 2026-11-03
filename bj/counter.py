"""Shoe tracking: Hi-Lo running count, true count, and remaining composition."""
from __future__ import annotations

import math
from typing import Dict

from .cards import card_name

HI_LO = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1}


class Shoe:
    def __init__(self, decks: int = 6):
        self.decks = decks
        self.reset()

    def reset(self) -> None:
        self.remaining: Dict[int, int] = {v: 4 * self.decks for v in range(1, 10)}
        self.remaining[10] = 16 * self.decks
        self.running_count = 0
        self.cards_seen = 0

    def see(self, card: int) -> None:
        if self.remaining[card] == 0:
            raise ValueError(f"No {card_name(card)}s left in the shoe. Check your input, or reset.")
        self.remaining[card] -= 1
        self.running_count += HI_LO[card]
        self.cards_seen += 1

    def unsee(self, card: int) -> None:
        """Put a card back, e.g. after a misread."""
        self.remaining[card] += 1
        self.running_count -= HI_LO[card]
        self.cards_seen -= 1

    @property
    def cards_left(self) -> int:
        return sum(self.remaining.values())

    @property
    def decks_left(self) -> float:
        return self.cards_left / 52

    @property
    def true_count(self) -> float:
        # Don't let the count explode in the last few cards.
        return self.running_count / max(self.decks_left, 0.25)

    def probabilities(self) -> Dict[int, float]:
        left = self.cards_left
        if left == 0:
            raise ValueError("The shoe is empty. Reset it.")
        return {v: n / left for v, n in self.remaining.items()}


def bet_units(true_count: float, spread: int = 8) -> int:
    """Simple Hi-Lo ramp: bet the floored true count in units, 1 to `spread`."""
    return max(1, min(spread, math.floor(true_count)))


def player_edge(true_count: float, house_edge: float) -> float:
    """Each true count point is worth about +0.5% to the player."""
    return -house_edge + 0.5 * true_count
