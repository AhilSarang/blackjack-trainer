"""Card parsing and hand evaluation.

Cards are represented as ints 1..10: 1 is an ace, 10 is any ten-value card
(10, J, Q, K). Suits don't matter in blackjack, so they're accepted and dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

_RANKS = {
    "A": 1, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "10": 10, "T": 10, "J": 10, "Q": 10, "K": 10,
}
_SUITS = set("HDCS♠♥♦♣")


def parse_card(token: str) -> int:
    """Parse '10', 'K', 'Kh', 'A♠', 't' ... into a card value 1..10."""
    t = token.strip().upper()
    if len(t) > 1 and t[-1] in _SUITS:
        t = t[:-1]
    if t not in _RANKS:
        raise ValueError(f"Not a card: {token!r} (use A, 2-10, J, Q, K)")
    return _RANKS[t]


def parse_cards(tokens: List[str]) -> List[int]:
    return [parse_card(t) for t in tokens]


def card_name(value: int) -> str:
    return "A" if value == 1 else str(value)


@dataclass
class Hand:
    cards: List[int] = field(default_factory=list)
    from_split: bool = False
    doubled: bool = False
    surrendered: bool = False
    done: bool = False

    @property
    def hard_total(self) -> int:
        return sum(self.cards)

    @property
    def soft(self) -> bool:
        """True if an ace is currently counted as 11."""
        return 1 in self.cards and self.hard_total + 10 <= 21

    @property
    def total(self) -> int:
        return self.hard_total + 10 if self.soft else self.hard_total

    @property
    def is_pair(self) -> bool:
        return len(self.cards) == 2 and self.cards[0] == self.cards[1]

    @property
    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and self.total == 21 and not self.from_split

    @property
    def busted(self) -> bool:
        return self.hard_total > 21

    def describe(self) -> str:
        cards = " ".join(card_name(c) for c in self.cards)
        if self.is_blackjack:
            return f"{cards}  (blackjack)"
        if self.busted:
            return f"{cards}  (bust, {self.total})"
        kind = "soft" if self.soft else "hard"
        return f"{cards}  ({kind} {self.total})"
