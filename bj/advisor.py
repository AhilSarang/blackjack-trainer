"""One call that turns (hand, dealer upcard, shoe) into everything the overlay shows.

This is the function the vision pipeline will call each time the table state changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .cards import Hand
from .counter import Shoe, bet_units, player_edge
from .odds import Odds
from .rules import Rules
from .strategy import DOUBLE, HIT, STAND, SURRENDER, Decision, decide, take_insurance


@dataclass
class Advice:
    decision: Decision
    bust_if_hit: float
    dealer_bust: float
    ev: Dict[str, float]          # per-action EV in units of the bet; split isn't modeled
    insurance: Optional[bool]     # None when the dealer isn't showing an ace


@dataclass
class CountStatus:
    running_count: int
    true_count: float
    decks_left: float
    bet_units: int
    edge_pct: float


def advise(hand: Hand, up: int, rules: Rules, probs: Dict[int, float],
           true_count: Optional[float]) -> Advice:
    decision = decide(hand, up, rules, true_count)
    odds = Odds(probs, up, rules)
    has_ace = 1 in hand.cards
    two_cards = len(hand.cards) == 2

    ev = {STAND: odds.ev_stand(hand.total), HIT: odds.ev_hit(hand.hard_total, has_ace)}
    if two_cards and (not hand.from_split or rules.double_after_split):
        ev[DOUBLE] = odds.ev_double(hand.hard_total, has_ace)
    if rules.late_surrender and two_cards and not hand.from_split:
        ev[SURRENDER] = -0.5

    return Advice(
        decision=decision,
        bust_if_hit=odds.bust_if_hit(hand.hard_total),
        dealer_bust=odds.dealer_bust,
        ev=ev,
        insurance=take_insurance(up, true_count) if up == 1 else None,
    )


def count_status(shoe: Shoe, rules: Rules, spread: int = 8) -> CountStatus:
    tc = shoe.true_count
    return CountStatus(
        running_count=shoe.running_count,
        true_count=tc,
        decks_left=shoe.decks_left,
        bet_units=bet_units(tc, spread),
        edge_pct=player_edge(tc, rules.house_edge()),
    )
