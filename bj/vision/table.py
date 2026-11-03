"""Turns noisy per-frame card sightings into a stable round: who has what, counted once.

Rules of thumb:
- A card is counted only after it's been seen in `confirm_frames` frames, so a
  one-frame misread never touches the count.
- Once counted, a card stays in the round even if a hand covers it for a while.
- A card joins a hand only once it has come to rest (it's counted as soon as it's
  read, though), so cards moving through the dealer's hand don't land anywhere.
- The dealer's cards sit in one row, level with the dealer's upcard (the first card
  to settle above the line). Cards below that row are yours, even when your hit
  cards fan up past the line. Cards above it are in the dealer's hand or on the
  discard pile, and don't belong to either hand yet.
- Once a round is decided (dealer on 17+, or you bust), hands are frozen so the
  dealer's sweep can't reshuffle them, and the round ends as soon as those cards
  are gone. An undecided round ends after the table has been empty for
  `clear_seconds`. The count carries over; the shoe only resets when you say so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..cards import Hand, parse_card
from ..counter import Shoe


@dataclass
class TableCard:
    code: str
    value: int
    x: float
    y: float
    zone: str = ""          # "dealer" or "player" once the card has come to rest
    rest_x: float = 0.0     # where the card has been sitting, and for how many updates
    rest_y: float = 0.0
    rest_count: int = 0
    settled: bool = False
    hits: int = 1
    last_seen: float = 0.0
    counted: bool = False
    ignored: bool = False   # misread you removed; tracked but never counted
    order: int = 0
    box: Optional[tuple] = None   # (x1, y1, x2, y2) around the corners last seen


@dataclass
class Table:
    shoe: Shoe
    divider: float = 0.45           # fraction of frame height; your spot is below it
    confirm_frames: int = 5
    clear_seconds: float = 2.0      # empty table ends an undecided round
    decided_clear_seconds: float = 0.8
    candidate_timeout: float = 0.6
    match_radius: float = 0.08      # fraction of the frame diagonal
    still_radius: float = 0.05      # moving less than this (fraction of diagonal) = resting;
                                    # generous, since a card's centre jumps when only one
                                    # of its two corners is read
    row_tolerance: float = 0.05     # how far (fraction of height) a dealer card can sit
                                    # from the upcard's level and still be in the row
    settle_frames: int = 5          # updates at rest before a card joins a hand
    cards: List[TableCard] = field(default_factory=list)
    frozen: bool = False            # round decided: hands no longer change
    _next_order: int = 0
    rounds_played: int = 0

    # ---- per-frame update ----------------------------------------------------

    def update(self, sightings, now: float, frame_w: int, frame_h: int) -> List[str]:
        """Feed one frame of detections. Returns human-readable events."""
        events: List[str] = []
        diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
        radius = self.match_radius * diag
        matched = set()
        newly_settled: List[TableCard] = []

        for s in sorted(sightings, key=lambda s: -s.conf):
            card = self._match(s, radius, matched)
            if card is None:
                card = TableCard(s.code, parse_card(s.code), s.x, s.y, last_seen=now, hits=0)
                self.cards.append(card)
            matched.add(id(card))
            card.x, card.y, card.last_seen = s.x, s.y, now
            boxes = getattr(s, "boxes", None)
            if boxes:
                card.box = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                            max(b[2] for b in boxes), max(b[3] for b in boxes))
            card.hits += 1
            if ((card.x - card.rest_x) ** 2 + (card.y - card.rest_y) ** 2) ** 0.5 < self.still_radius * diag:
                card.rest_count += 1
            else:
                card.rest_x, card.rest_y, card.rest_count = card.x, card.y, 1
            if not card.counted and not card.ignored and card.hits >= self.confirm_frames:
                self.shoe.see(card.value)
                card.counted = True
                card.order = self._next_order
                self._next_order += 1
            if (card.counted and not card.settled and not self.frozen
                    and card.rest_count >= self.settle_frames):
                card.settled = True
                newly_settled.append(card)

        # Forget unconfirmed one-off detections.
        self.cards = [c for c in self.cards
                      if c.counted or c.ignored or now - c.last_seen < self.candidate_timeout]

        if not self.frozen:
            self._assign_hands(frame_w, frame_h)
            for c in newly_settled:
                events.append(f"{'Dealer' if c.zone == 'dealer' else 'You'}: {c.code}")
            if self.decided:
                self.frozen = True

        # The round's cards are gone: it's over.
        counted = self.counted_cards
        if counted:
            gone_for = now - max(c.last_seen for c in counted)
            if gone_for >= (self.decided_clear_seconds if self.frozen else self.clear_seconds):
                self.end_round()
                events.append("Round over")
        return events

    def _assign_hands(self, frame_w: int, frame_h: int) -> None:
        settled = sorted((c for c in self.cards if c.counted and c.settled), key=lambda c: c.order)
        upcard = next((c for c in settled if c.y < self.divider * frame_h), None)
        tolerance = self.row_tolerance * frame_h
        # If the "upcard" was really one of your hit cards (the real upcard was missed),
        # the dealer's row shows up as several resting cards above it: move up to them.
        # One card above is just the dealer holding a card, so it doesn't count.
        for _ in range(3):
            above = [c for c in settled if upcard is not None and c.y < upcard.y - tolerance]
            if len(above) < 2:
                break
            upcard = above[0]
        for c in settled:
            if upcard is None or c.y > upcard.y + tolerance:
                c.zone = "player"
            elif c.y >= upcard.y - tolerance:
                c.zone = "dealer"
            else:
                c.zone = ""     # held up by the dealer, or on the discard pile

    def _match(self, s, radius: float, matched: set) -> Optional[TableCard]:
        same = [c for c in self.cards if c.code == s.code and id(c) not in matched]
        if not same:
            return None
        dist = lambda c: ((c.x - s.x) ** 2 + (c.y - s.y) ** 2) ** 0.5
        nearest = min(same, key=dist)
        if dist(nearest) <= radius:
            return nearest
        # The same card still on its way (being dealt or slid): don't count it twice.
        # A card that has already settled stays put, so a far-away match is another
        # copy of it from the shoe (multi-deck), not the same card.
        moving = [c for c in same if (c.counted or c.ignored) and not c.settled]
        return min(moving, key=dist) if moving else None

    # ---- round state -----------------------------------------------------------

    @property
    def decided(self) -> bool:
        you, dealer = self.player_hand, self.dealer_hand
        return you.busted or (len(dealer.cards) >= 2 and dealer.total >= 17)

    @property
    def counted_cards(self) -> List[TableCard]:
        return sorted((c for c in self.cards if c.counted), key=lambda c: c.order)

    def zone_cards(self, zone: str) -> List[int]:
        return [c.value for c in self.counted_cards if c.zone == zone]

    @property
    def player_hand(self) -> Hand:
        return Hand(self.zone_cards("player"))

    @property
    def dealer_hand(self) -> Hand:
        return Hand(self.zone_cards("dealer"))

    def end_round(self) -> None:
        self.cards = []
        self.frozen = False
        self.rounds_played += 1

    def remove_last(self) -> Optional[str]:
        """Undo the most recently counted card (for a misread)."""
        counted = self.counted_cards
        if not counted:
            return None
        card = counted[-1]
        self.shoe.unsee(card.value)
        card.counted, card.ignored = False, True
        card.zone = ""
        return card.code
