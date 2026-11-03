"""Multi-panel heads-up display drawn over the video.

Panels: recommendation with per-action EV bars, hand and dealer-outcome odds,
count gauge with shoe penetration and bet sizing, remaining-shoe composition,
and a decision history strip. Cards get corner brackets, a label, and their
Hi-Lo value.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ..advisor import advise, count_status
from ..counter import HI_LO
from ..odds import BUST, Odds
from ..rules import Rules
from ..strategy import DOUBLE, HIT, SPLIT, STAND, SURRENDER
from .table import Table

# BGR palette
BG = (18, 16, 14)
EDGE = (90, 84, 78)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
ACCENT = (255, 190, 60)        # cyan-blue
GOOD = (110, 220, 120)
WARN = (40, 200, 255)
BAD = (80, 80, 245)
RED_SUIT = (70, 70, 230)
ACTION_COLOR = {HIT: WARN, STAND: GOOD, DOUBLE: ACCENT, SPLIT: (230, 120, 220), SURRENDER: BAD}
RESULT_COLOR = {"YOU WIN": GOOD, "BLACKJACK!": GOOD, "DEALER WINS": BAD, "BUST": BAD, "PUSH": TEXT}

FONT, BOLD = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX


class Canvas:
    def __init__(self, img: np.ndarray):
        self.img = img
        self.s = img.shape[1] / 1920

    def px(self, v: float) -> int:
        return int(round(v * self.s))

    def text(self, t, x, y, size=0.6, color=TEXT, bold=False, thick=1):
        cv2.putText(self.img, t, (int(x), int(y)), BOLD if bold else FONT, size * self.s,
                    color, max(1, int(round(thick * self.s))), cv2.LINE_AA)

    def text_w(self, t, size=0.6, bold=False, thick=1) -> int:
        return cv2.getTextSize(t, BOLD if bold else FONT, size * self.s, max(1, int(round(thick * self.s))))[0][0]

    def rect(self, x1, y1, x2, y2, color, thick=-1):
        cv2.rectangle(self.img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick if thick < 0 else max(1, int(thick)))

    def shade(self, x1, y1, x2, y2, alpha=0.72):
        h, w = self.img.shape[:2]
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            roi = self.img[y1:y2, x1:x2]
            roi[:] = (roi * (1 - alpha) + np.array(BG) * alpha).astype(np.uint8)

    def panel(self, x, y, w, h, title):
        """Dark glass panel with a title bar and accent corner marks."""
        self.shade(x, y, x + w, y + h)
        self.rect(x, y, x + w, y + h, EDGE, 1)
        k = self.px(14)
        for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            cv2.line(self.img, (int(cx), int(cy)), (int(cx + dx * k), int(cy)), ACCENT, max(1, self.px(2)), cv2.LINE_AA)
            cv2.line(self.img, (int(cx), int(cy)), (int(cx), int(cy + dy * k)), ACCENT, max(1, self.px(2)), cv2.LINE_AA)
        self.text(title.upper(), x + self.px(14), y + self.px(26), 0.5, ACCENT, thick=1)
        cv2.line(self.img, (int(x + self.px(14)), int(y + self.px(36))), (int(x + w - self.px(14)), int(y + self.px(36))),
                 EDGE, 1, cv2.LINE_AA)
        return x + self.px(14), y + self.px(36)

    def bar(self, x, y, w, h, frac, color, back=(60, 56, 52)):
        self.rect(x, y, x + w, y + h, back)
        self.rect(x, y, x + w * max(0.0, min(1.0, frac)), y + h, color)


# ---- panels ---------------------------------------------------------------------

def _recommendation(c: Canvas, x, y, w, headline, color, advice, deviation_note):
    h = c.px(300 if advice else 140)
    ix, iy = c.panel(x, y, w, h, "Recommendation")
    c.text(headline, ix, iy + c.px(62), 1.9, color, bold=True, thick=3)
    if deviation_note:
        c.text(deviation_note, ix, iy + c.px(92), 0.5, WARN)
    if not advice:
        return y + h
    best = max(advice.ev, key=advice.ev.get)
    row = iy + c.px(118)
    c.text("EXPECTED VALUE PER UNIT BET", ix, row, 0.42, MUTED)
    bw = w - c.px(28) - c.px(210)
    for action, ev in advice.ev.items():
        row += c.px(34)
        col = ACTION_COLOR.get(action, TEXT) if action == best else MUTED
        c.text(action.title(), ix, row, 0.55, TEXT if action == best else MUTED, bold=action == best)
        bx, mid = ix + c.px(120), ix + c.px(120) + bw / 2
        c.rect(bx, row - c.px(14), bx + bw, row - c.px(2), (52, 48, 45))
        cv2.line(c.img, (int(mid), int(row - c.px(17))), (int(mid), int(row + c.px(1))), EDGE, 1)
        end = mid + (bw / 2) * max(-1.0, min(1.0, ev))
        c.rect(min(mid, end), row - c.px(14), max(mid, end), row - c.px(2), col)
        c.text(f"{ev * 100:+.1f}%", bx + bw + c.px(12), row, 0.5, col if action == best else MUTED)
    return y + h


def _hand(c: Canvas, x, y, w, table: Table, rules: Rules, advice):
    you, dealer = table.player_hand, table.dealer_hand
    h = c.px(272)
    ix, iy = c.panel(x, y, w, h, "Hand analysis")
    kind = "SOFT" if you.soft else "HARD"
    c.text("YOU", ix, iy + c.px(32), 0.45, MUTED)
    c.text(f"{you.total}" if you.cards else "--", ix, iy + c.px(78), 1.6, TEXT, bold=True, thick=2)
    if you.cards:
        c.text(kind, ix + c.px(92), iy + c.px(78), 0.5, MUTED)
    dx = ix + w // 2
    c.text("DEALER", dx, iy + c.px(32), 0.45, MUTED)
    c.text(f"{dealer.total}" if dealer.cards else "--", dx, iy + c.px(78), 1.6, TEXT, bold=True, thick=2)
    if advice:
        c.text(f"Bust if hit  {advice.bust_if_hit:.0%}", ix, iy + c.px(112), 0.5, TEXT)
        c.text(f"Dealer bust  {advice.dealer_bust:.0%}", dx, iy + c.px(112), 0.5, TEXT)
    # Dealer final-total distribution.
    if dealer.cards:
        c.text("DEALER FINISHES ON", ix, iy + c.px(144), 0.42, MUTED)
        odds = Odds(table.shoe.probabilities(), dealer.cards[0], rules)
        keys = [17, 18, 19, 20, 21, BUST]
        slot = (w - c.px(28)) / len(keys)
        base = iy + c.px(200)
        for i, k in enumerate(keys):
            p = odds.dealer.get(k, 0.0)
            bx = ix + i * slot
            bh = c.px(44) * min(1.0, p / 0.5)
            c.rect(bx + c.px(6), base - bh, bx + slot - c.px(6), base, GOOD if k == BUST else ACCENT)
            c.text("BUST" if k == BUST else str(k), bx + c.px(8), base + c.px(18), 0.4, MUTED)
            c.text(f"{p:.0%}", bx + c.px(8), base - bh - c.px(6), 0.4, TEXT)
    return y + h


def _count(c: Canvas, x, y, w, table: Table, rules: Rules, spread: int):
    cs = count_status(table.shoe, rules, spread)
    h = c.px(322)
    ix, iy = c.panel(x, y, w, h, "Hi-Lo count")
    c.text("RUNNING", ix, iy + c.px(32), 0.45, MUTED)
    c.text(f"{cs.running_count:+d}", ix, iy + c.px(80), 1.6, TEXT, bold=True, thick=2)
    tx = ix + w // 2 - c.px(10)
    c.text("TRUE", tx, iy + c.px(32), 0.45, MUTED)
    tc_col = GOOD if cs.true_count >= 2 else (BAD if cs.true_count <= -1 else TEXT)
    c.text(f"{cs.true_count:+.1f}", tx, iy + c.px(80), 1.6, tc_col, bold=True, thick=2)

    # Gauge -6..+6
    gx, gy, gw = ix, iy + c.px(112), w - c.px(28)
    for i in range(60):
        t = i / 59
        col = tuple(int(a + (b - a) * t) for a, b in zip(BAD, GOOD))
        c.rect(gx + gw * i / 60, gy, gx + gw * (i + 1) / 60 + 1, gy + c.px(10), col)
    mark = gx + gw * (max(-6, min(6, cs.true_count)) + 6) / 12
    cv2.drawMarker(c.img, (int(mark), int(gy - c.px(4))), TEXT, cv2.MARKER_TRIANGLE_DOWN, c.px(16), max(1, c.px(2)))
    for v in (-6, -3, 0, 3, 6):
        vx = gx + gw * (v + 6) / 12
        c.text(f"{v:+d}" if v else "0", vx - c.px(10), gy + c.px(30), 0.38, MUTED)

    edge_col = GOOD if cs.edge_pct > 0 else MUTED
    c.text("PLAYER EDGE", ix, iy + c.px(176), 0.42, MUTED)
    c.text(f"{cs.edge_pct:+.2f}%", ix, iy + c.px(204), 0.8, edge_col, bold=True)
    c.text("SHOE DEALT", tx, iy + c.px(176), 0.42, MUTED)
    total = 52 * table.shoe.decks
    dealt = (total - table.shoe.cards_left) / total
    c.bar(tx, iy + c.px(190), w // 2 - c.px(26), c.px(12), dealt, ACCENT)
    c.text(f"{dealt:.0%}  ({cs.decks_left:.1f} decks left)", tx, iy + c.px(220), 0.4, MUTED)

    c.text("NEXT BET", ix, iy + c.px(244), 0.42, MUTED)
    high = cs.bet_units >= 2
    label = f"{cs.bet_units} UNITS - BET HIGH" if high else "1 UNIT - MINIMUM"
    chip_x = ix
    for i in range(min(cs.bet_units, 8)):
        cx = int(chip_x + c.px(14) + i * c.px(24))
        cv2.circle(c.img, (cx, int(iy + c.px(254))), c.px(10), GOOD if high else MUTED, -1, cv2.LINE_AA)
        cv2.circle(c.img, (cx, int(iy + c.px(254))), c.px(10), BG, max(1, c.px(2)), cv2.LINE_AA)
    c.text(label, ix + c.px(14) + min(cs.bet_units, 8) * c.px(24) + c.px(6), iy + c.px(261), 0.55,
           GOOD if high else TEXT, bold=high)
    return y + h


def _shoe(c: Canvas, x, y, w, table: Table):
    h = c.px(190)
    ix, iy = c.panel(x, y, w, h, "Remaining shoe")
    shoe = table.shoe
    full = {v: 4 * shoe.decks for v in range(1, 10)}
    full[10] = 16 * shoe.decks
    expected = shoe.cards_left / (52 * shoe.decks)
    ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    slot = (w - c.px(28)) / len(ranks)
    base, top = iy + c.px(118), iy + c.px(20)
    ref_y = base - (base - top) * expected
    for i, r in enumerate(ranks):
        frac = shoe.remaining[r] / full[r]
        bx = ix + i * slot
        bar_top = base - (base - top) * frac
        rich = frac > expected + 0.005
        col = ACCENT if rich else (90, 86, 82)
        c.rect(bx + c.px(5), bar_top, bx + slot - c.px(5), base, col)
        name = "A" if r == 1 else ("T" if r == 10 else str(r))
        c.text(name, bx + slot / 2 - c.px(6), base + c.px(22), 0.45, TEXT)
        hl = HI_LO[r]
        c.text(f"{hl:+d}" if hl else "0", bx + slot / 2 - c.px(10), base + c.px(42), 0.35,
               GOOD if hl > 0 else (BAD if hl < 0 else MUTED))
    cv2.line(c.img, (int(ix), int(ref_y)), (int(ix + w - c.px(28)), int(ref_y)), TEXT, 1, cv2.LINE_AA)
    c.text("expected", ix + w - c.px(28) - c.text_w("expected", 0.35), ref_y - c.px(5), 0.35, MUTED)
    return y + h


def _history(c: Canvas, x, y, w, log_lines: List[str]):
    rows = log_lines[-4:]
    h = c.px(44 + 28 * max(1, len(rows)))
    ix, iy = c.panel(x, y, w, h, "Decision log")
    if not rows:
        c.text("Waiting for the first decision...", ix, iy + c.px(28), 0.45, MUTED)
    for i, line in enumerate(rows):
        col = TEXT if i == len(rows) - 1 else MUTED
        c.text(line, ix, iy + c.px(28) + i * c.px(28), 0.45, col)
    return y + h


# ---- cards ------------------------------------------------------------------------

def _cards(c: Canvas, table: Table):
    k = c.px(16)
    for card in table.cards:
        if card.ignored:
            continue
        x1, y1, x2, y2 = card.box if card.box else (card.x - 40, card.y - 40, card.x + 40, card.y + 40)
        pad = c.px(10)
        x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
        if not card.counted:
            col = MUTED
        elif card.zone == "dealer":
            col = WARN
        elif card.zone == "player":
            col = GOOD
        else:
            col = TEXT
        t = max(1, c.px(3))
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(c.img, (int(cx), int(cy)), (int(cx + dx * k), int(cy)), col, t, cv2.LINE_AA)
            cv2.line(c.img, (int(cx), int(cy)), (int(cx), int(cy + dy * k)), col, t, cv2.LINE_AA)
        rank, suit = card.code[:-1], card.code[-1]
        hl = HI_LO[card.value]
        tag = f"{hl:+d}" if hl else "0"
        tw = c.text_w(rank + suit, 0.7, bold=True, thick=2) + c.text_w("  " + tag, 0.5) + c.px(16)
        lx, ly = x1, y1 - c.px(8)
        c.rect(lx, ly - c.px(28), lx + tw, ly, (24, 22, 20))
        c.rect(lx, ly - c.px(28), lx + c.px(4), ly, col)
        c.text(rank, lx + c.px(10), ly - c.px(7), 0.7, TEXT, bold=True, thick=2)
        sx = lx + c.px(10) + c.text_w(rank, 0.7, bold=True, thick=2)
        c.text(suit, sx, ly - c.px(7), 0.7, RED_SUIT if suit in "HD" else TEXT, bold=True, thick=2)
        c.text("  " + tag, sx + c.text_w(suit, 0.7, bold=True, thick=2), ly - c.px(8), 0.5,
               GOOD if hl > 0 else (BAD if hl < 0 else MUTED))


def _divider(c: Canvas, table: Table):
    h, w = c.img.shape[:2]
    y = int(table.divider * h)
    dash = c.px(18)
    for x in range(0, w, dash * 2):
        cv2.line(c.img, (x, y), (min(w, x + dash), y), (200, 200, 200), 1, cv2.LINE_AA)
    x = c.px(24 + 560 + 24)
    c.text("DEALER ZONE", x, y - c.px(10), 0.45, WARN)
    c.text("PLAYER ZONE", x, y + c.px(26), 0.45, GOOD)


# ---- entry point -------------------------------------------------------------------

def draw_hud(frame: np.ndarray, table: Table, rules: Rules, spread: int, headline: str, color,
             log_lines: List[str], flash: Optional[str], fps: float, hints: bool) -> np.ndarray:
    out = frame.copy()
    c = Canvas(out)
    _divider(c, table)
    _cards(c, table)

    you, dealer = table.player_hand, table.dealer_hand
    advice, note = None, ""
    if (len(you.cards) >= 2 and dealer.cards and len(dealer.cards) < 2
            and not you.busted and you.total < 21):
        advice = advise(you, dealer.cards[0], rules, table.shoe.probabilities(), table.shoe.true_count)
        if advice.decision.deviation:
            note = f"COUNT DEVIATION - basic strategy says {advice.decision.basic_action}"
    color = RESULT_COLOR.get(headline, ACTION_COLOR.get(headline, color))

    h, w = out.shape[:2]
    m, lw, rw = c.px(24), c.px(560), c.px(470)
    ly = _recommendation(c, m, m, lw, headline, color, advice, note)
    _hand(c, m, ly + c.px(14), lw, table, rules, advice)
    ry = _count(c, w - m - rw, m, rw, table, rules, spread)
    _shoe(c, w - m - rw, ry + c.px(14), rw, table)

    hist_w = c.px(900)
    hist_lines = [_short(l) for l in log_lines]
    hh = c.px(44 + 28 * max(1, len(hist_lines[-4:])))
    _history(c, (w - hist_w) / 2, h - m - hh - (c.px(40) if hints else 0), hist_w, hist_lines)

    if flash:
        tw = c.text_w(flash, 0.7, bold=True)
        fx, fy = (w - tw) / 2, c.px(60)
        c.shade(fx - c.px(16), fy - c.px(34), fx + tw + c.px(16), fy + c.px(14), 0.8)
        c.rect(fx - c.px(16), fy - c.px(34), fx - c.px(12), fy + c.px(14), GOOD)
        c.text(flash, fx, fy, 0.7, GOOD, bold=True)
    if hints:
        c.shade(0, h - c.px(34), w, h, 0.8)
        c.text(f"Q quit   N new round   R new shoe   X remove last card   W/S move line   C camera     {fps:.0f} fps",
               m, h - c.px(11), 0.45, MUTED)
    return out


def _short(line: str) -> str:
    """Compact a DecisionLog line for the history strip."""
    line = " ".join(line.split()).replace(" -> ", "  >  ")
    return line.split(" Count ")[0]
