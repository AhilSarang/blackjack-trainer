"""Probabilities and expected values from the current shoe composition.

Draws use the remaining shoe's card frequencies and treat them as fixed for the
rest of the hand. With dozens of cards left, that's within a fraction of a
percent of exact. When the dealer peeks, everything is conditioned on the
dealer not having blackjack, which is the situation you're deciding in.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

from .rules import Rules

BUST = 22  # key used for "dealer busts" in outcome tables
CARDS = range(1, 11)


def _total(hard: int, has_ace: bool) -> Tuple[int, bool]:
    if has_ace and hard + 10 <= 21:
        return hard + 10, True
    return hard, False


class Odds:
    def __init__(self, probs: Dict[int, float], up: int, rules: Rules):
        self.p = probs
        self.up = up
        self.rules = rules
        self.dealer = self._dealer_outcomes()
        self._ev_hit = lru_cache(maxsize=None)(self._ev_hit_uncached)

    # ---- dealer ---------------------------------------------------------

    def _dealer_outcomes(self) -> Dict[int, float]:
        """Probability of each dealer final total (17-21) or BUST."""
        p, h17 = self.p, self.rules.hit_soft_17
        memo: Dict[Tuple[int, bool], Dict[int, float]] = {}

        def finish(hard: int, has_ace: bool) -> Dict[int, float]:
            total, soft = _total(hard, has_ace)
            if total > 21:
                return {BUST: 1.0}
            if total > 17 or (total == 17 and not (soft and h17)):
                return {total: 1.0}
            key = (hard, has_ace)
            if key not in memo:
                out: Dict[int, float] = {}
                for c in CARDS:
                    if p[c]:
                        for t, q in finish(hard + c, has_ace or c == 1).items():
                            out[t] = out.get(t, 0.0) + p[c] * q
                memo[key] = out
            return memo[key]

        # The hole card: if the dealer peeked and had no blackjack, it can't
        # be the card that would have made one.
        excluded = None
        if self.rules.dealer_peeks:
            excluded = {1: 10, 10: 1}.get(self.up)
        weights = {c: (0.0 if c == excluded else p[c]) for c in CARDS}
        norm = sum(weights.values())
        out: Dict[int, float] = {}
        for c in CARDS:
            if weights[c]:
                for t, q in finish(self.up + c, self.up == 1 or c == 1).items():
                    out[t] = out.get(t, 0.0) + weights[c] / norm * q
        return out

    @property
    def dealer_bust(self) -> float:
        return self.dealer.get(BUST, 0.0)

    # ---- player ---------------------------------------------------------

    def ev_stand(self, total: int) -> float:
        if total > 21:
            return -1.0
        ev = 0.0
        for t, q in self.dealer.items():
            if t == BUST or t < total:
                ev += q
            elif t > total:
                ev -= q
        return ev

    def _ev_hit_uncached(self, hard: int, has_ace: bool) -> float:
        """EV of taking one card, then playing hit/stand optimally."""
        ev = 0.0
        for c in CARDS:
            if not self.p[c]:
                continue
            nh, na = hard + c, has_ace or c == 1
            total, _ = _total(nh, na)
            if total > 21:
                ev -= self.p[c]
            else:
                ev += self.p[c] * max(self.ev_stand(total), self._ev_hit(nh, na))
        return ev

    def ev_hit(self, hard: int, has_ace: bool) -> float:
        return self._ev_hit(hard, has_ace)

    def ev_double(self, hard: int, has_ace: bool) -> float:
        ev = 0.0
        for c in CARDS:
            total, _ = _total(hard + c, has_ace or c == 1)
            ev += self.p[c] * self.ev_stand(total)
        return 2 * ev

    def bust_if_hit(self, hard: int) -> float:
        # Aces count as 1 in `hard`, so a hand only busts when its hard total does.
        return sum(self.p[c] for c in CARDS if hard + c > 21)
