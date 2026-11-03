from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    decks: int = 6
    hit_soft_17: bool = True          # H17: dealer hits soft 17 (most common in the US)
    double_after_split: bool = True   # DAS
    late_surrender: bool = True
    dealer_peeks: bool = True         # dealer checks for blackjack under an A or 10
    blackjack_pays: float = 1.5       # 3:2. Use 1.2 for 6:5 tables.

    def house_edge(self) -> float:
        """Approximate house edge in percent for a basic-strategy player."""
        edge = 0.42  # baseline: 6 decks, S17, DAS, no surrender
        edge += {1: -0.48, 2: -0.19, 4: -0.06, 6: 0.0, 8: 0.02}.get(self.decks, 0.0)
        if self.hit_soft_17:
            edge += 0.22
        if not self.double_after_split:
            edge += 0.14
        if self.late_surrender:
            edge -= 0.07
        if self.blackjack_pays < 1.5:
            edge += 1.39
        return edge

    def describe(self) -> str:
        parts = [
            f"{self.decks} deck{'s' if self.decks != 1 else ''}",
            "H17" if self.hit_soft_17 else "S17",
            "DAS" if self.double_after_split else "no DAS",
            "late surrender" if self.late_surrender else "no surrender",
            "3:2" if self.blackjack_pays >= 1.5 else "6:5",
        ]
        return ", ".join(parts)
