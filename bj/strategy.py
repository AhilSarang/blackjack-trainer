"""Basic strategy (4-8 decks) plus Hi-Lo count deviations.

Table codes:
  H  hit                S  stand               P  split
  D  double, else hit   Ds double, else stand
  Rh surrender, else hit    Rs surrender, else stand    Rp surrender, else split
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .cards import Hand
from .rules import Rules

HIT, STAND, DOUBLE, SPLIT, SURRENDER = "HIT", "STAND", "DOUBLE", "SPLIT", "SURRENDER"

#               dealer: 2  3  4  5  6  7  8  9  10  A
_HARD_H17 = {
    17: "S  S  S  S  S  S  S  S  S  Rs",
    16: "S  S  S  S  S  H  H  Rh Rh Rh",
    15: "S  S  S  S  S  H  H  H  Rh Rh",
    14: "S  S  S  S  S  H  H  H  H  H",
    13: "S  S  S  S  S  H  H  H  H  H",
    12: "H  H  S  S  S  H  H  H  H  H",
    11: "D  D  D  D  D  D  D  D  D  D",
    10: "D  D  D  D  D  D  D  D  H  H",
    9:  "H  D  D  D  D  H  H  H  H  H",
}
_SOFT_H17 = {
    20: "S  S  S  S  S  S  S  S  S  S",
    19: "S  S  S  S  Ds S  S  S  S  S",
    18: "Ds Ds Ds Ds Ds S  S  H  H  H",
    17: "H  D  D  D  D  H  H  H  H  H",
    16: "H  H  D  D  D  H  H  H  H  H",
    15: "H  H  D  D  D  H  H  H  H  H",
    14: "H  H  H  D  D  H  H  H  H  H",
    13: "H  H  H  D  D  H  H  H  H  H",
}
# Keyed by the value of one card in the pair. 5s are played as hard 10.
_PAIRS_H17 = {
    1:  "P  P  P  P  P  P  P  P  P  P",
    10: "S  S  S  S  S  S  S  S  S  S",
    9:  "P  P  P  P  P  S  P  P  S  S",
    8:  "P  P  P  P  P  P  P  P  P  Rp",
    7:  "P  P  P  P  P  P  H  H  H  H",
    6:  "P  P  P  P  P  H  H  H  H  H",
    4:  "H  H  H  P  P  H  H  H  H  H",
    3:  "P  P  P  P  P  P  H  H  H  H",
    2:  "P  P  P  P  P  P  H  H  H  H",
}
# Where S17 basic strategy differs from H17: (table, total, upcard) -> code
_S17_CHANGES = {
    ("hard", 17, 1): "S",
    ("hard", 15, 1): "H",
    ("hard", 11, 1): "H",
    ("soft", 19, 6): "S",
    ("soft", 18, 2): "S",
    ("pair", 8, 1): "P",
}

# Hi-Lo play deviations (Illustrious 18, multi-deck):
# (table, total, upcard) -> (index, code if TC >= index, code if TC < index)
_DEVIATIONS_H17 = {
    ("hard", 16, 10): (0, "S", "H"),
    ("hard", 15, 10): (4, "S", "H"),
    ("hard", 16, 9): (5, "S", "H"),
    ("hard", 13, 2): (-1, "S", "H"),
    ("hard", 13, 3): (-2, "S", "H"),
    ("hard", 12, 2): (3, "S", "H"),
    ("hard", 12, 3): (2, "S", "H"),
    ("hard", 12, 4): (0, "S", "H"),
    ("hard", 12, 5): (-2, "S", "H"),
    ("hard", 12, 6): (-1, "S", "H"),
    ("hard", 10, 10): (4, "D", "H"),
    ("hard", 10, 1): (3, "D", "H"),
    ("hard", 9, 2): (1, "D", "H"),
    ("hard", 9, 7): (3, "D", "H"),
    ("pair", 10, 5): (5, "P", "S"),
    ("pair", 10, 6): (4, "P", "S"),
}
_S17_DEVIATION_CHANGES = {
    ("hard", 10, 1): (4, "D", "H"),
    ("hard", 11, 1): (1, "D", "H"),
}

# Hi-Lo surrender deviations ("Fab 4"): (total, upcard) -> surrender if TC >= index
_SURRENDER_H17 = {(14, 10): 3, (15, 9): 2, (15, 10): 0, (15, 1): -1}
_SURRENDER_S17 = {(14, 10): 3, (15, 9): 2, (15, 10): 0, (15, 1): 1}

INSURANCE_INDEX = 3

_UPCARD_COLUMN = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 1: 9}


def _parse(table: Dict[int, str]) -> Dict[int, Tuple[str, ...]]:
    return {k: tuple(v.split()) for k, v in table.items()}


_TABLES = {"hard": _parse(_HARD_H17), "soft": _parse(_SOFT_H17), "pair": _parse(_PAIRS_H17)}


def _lookup(table: str, total: int, up: int, rules: Rules) -> str:
    if not rules.hit_soft_17 and (table, total, up) in _S17_CHANGES:
        return _S17_CHANGES[(table, total, up)]
    rows = _TABLES[table]
    if table == "hard":
        if total >= 18:
            return "S"
        if total <= 8:
            return "H"
    if table == "soft":
        if total >= 20:
            return "S"
        if total <= 12:  # A,A once you can't split again
            return "H"
    return rows[total][_UPCARD_COLUMN[up]]


def _deviation(table: str, total: int, up: int, rules: Rules):
    key = (table, total, up)
    if not rules.hit_soft_17 and key in _S17_DEVIATION_CHANGES:
        return _S17_DEVIATION_CHANGES[key]
    return _DEVIATIONS_H17.get(key)


@dataclass
class Decision:
    action: str            # HIT / STAND / DOUBLE / SPLIT / SURRENDER
    basic_action: str      # what basic strategy alone says
    deviation: bool        # True when the count changed the play
    note: str = ""


def _resolve(code: str, can_double: bool, can_surrender: bool) -> str:
    if code.startswith("R"):
        if can_surrender:
            return SURRENDER
        code = code[1:].upper()  # Rh -> H, Rs -> S, Rp -> P
    if code == "D":
        return DOUBLE if can_double else HIT
    if code == "Ds":
        return DOUBLE if can_double else STAND
    return {"H": HIT, "S": STAND, "P": SPLIT}[code]


def decide(hand: Hand, up: int, rules: Rules, true_count: Optional[float] = None) -> Decision:
    """Best play for `hand` against dealer upcard `up`.

    Pass `true_count=None` for pure basic strategy; pass a number to apply
    Hi-Lo deviations.
    """
    if len(hand.cards) < 2:
        raise ValueError("Need at least two cards to advise.")
    if hand.total >= 21:
        return Decision(STAND, STAND, False)

    basic = _decide(hand, up, rules, None)
    if true_count is None:
        return Decision(basic, basic, False)
    counted = _decide(hand, up, rules, true_count)
    return Decision(counted, basic, counted != basic)


def _decide(hand: Hand, up: int, rules: Rules, tc: Optional[float]) -> str:
    two_cards = len(hand.cards) == 2
    can_double = two_cards and (not hand.from_split or rules.double_after_split)
    can_surrender = rules.late_surrender and two_cards and not hand.from_split

    # 1. Count-based surrender decisions take priority.
    if can_surrender and tc is not None and not hand.soft and not hand.is_pair:
        table = _SURRENDER_H17 if rules.hit_soft_17 else _SURRENDER_S17
        index = table.get((hand.total, up))
        if index is not None:
            if tc >= index:
                return SURRENDER
            can_surrender = False

    # 2. Pairs.
    if hand.is_pair and hand.cards[0] != 5:
        value = hand.cards[0]
        code = _lookup("pair", value, up, rules)
        dev = _deviation("pair", value, up, rules) if tc is not None else None
        if dev:
            index, above, below = dev
            code = above if tc >= index else below
        if code in ("P", "Rp"):
            return _resolve(code, can_double, can_surrender)
        # Not splitting: fall through and play it as a normal total.

    # 3. Soft totals.
    if hand.soft:
        return _resolve(_lookup("soft", hand.total, up, rules), can_double, can_surrender)

    # 4. Hard totals. Surrender is decided before any play deviation.
    code = _lookup("hard", hand.total, up, rules)
    if code.startswith("R") and can_surrender:
        return SURRENDER
    code = code[1:].upper() if code.startswith("R") else code
    dev = _deviation("hard", hand.total, up, rules) if tc is not None else None
    if dev:
        index, above, below = dev
        code = above if tc >= index else below
    return _resolve(code, can_double, can_surrender)


def take_insurance(up: int, true_count: Optional[float]) -> bool:
    """Insurance is only worth it with an ace up and TC >= +3."""
    return up == 1 and true_count is not None and true_count >= INSURANCE_INDEX
