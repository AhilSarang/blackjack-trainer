"""Type cards in as they're dealt and get live advice plus the count.

    python3 -m bj                       interactive session
    python3 -m bj advise 10 6 vs 10     one-off advice (add --tc 3 to apply the count)
"""
from __future__ import annotations

import argparse
import copy
import shlex
import sys
from typing import List, Optional

from .advisor import advise, count_status
from .cards import Hand, card_name, parse_cards
from .counter import Shoe
from .rules import Rules
from .strategy import DOUBLE, HIT, SPLIT, STAND, SURRENDER

_TTY = sys.stdout.isatty()


def _style(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def bold(t: str) -> str:
    return _style(t, "1")


def dim(t: str) -> str:
    return _style(t, "2")


def green(t: str) -> str:
    return _style(t, "1;32")


def yellow(t: str) -> str:
    return _style(t, "1;33")


def red(t: str) -> str:
    return _style(t, "1;31")


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def signed_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


HELP = """
Cards: A 2-10 J Q K (suits optional: Kh, 10s). Everything you enter is counted.

  deal 10 6 vs 10    start a round: your two cards, then the dealer's upcard
  hit 5   (h)        add a card to your current hand
  double 5 (d)       double down; the card is your last one
  split              split a pair, then 'hit' to give each hand its second card
  stand   (s)        stand on the current hand
  surrender (r)      surrender the current hand
  seen 2 K 9  (+)    other cards you saw (other players, burn cards)
  dealer 7 9         dealer's hole card and any draws; settles the round
  next    (n)        end the round without entering the dealer's cards
  count   (c)        show the count       shoe   remaining cards by rank
  undo    (u)        undo the last command
  reset              new shoe (shuffle)    rules  show table rules
  help    (?)        this help             quit   (q) exit
"""


class Session:
    def __init__(self, rules: Rules, spread: int, use_count: bool):
        self.rules = rules
        self.spread = spread
        self.use_count = use_count
        self.shoe = Shoe(rules.decks)
        self.history: List[tuple] = []
        self._new_round()

    # ---- state ------------------------------------------------------------

    def _new_round(self) -> None:
        self.hands: List[Hand] = []
        self.active = 0
        self.up: Optional[int] = None

    def _snapshot(self) -> None:
        self.history.append(copy.deepcopy((self.shoe, self.hands, self.active, self.up)))
        del self.history[:-50]

    def undo(self) -> None:
        if not self.history:
            print("Nothing to undo.")
            return
        self.shoe, self.hands, self.active, self.up = self.history.pop()
        print(dim("Undone."))
        self.show_table()

    def _see(self, cards: List[int]) -> None:
        for c in cards:
            self.shoe.see(c)

    @property
    def hand(self) -> Optional[Hand]:
        if self.hands and self.active < len(self.hands):
            return self.hands[self.active]
        return None

    def _advance(self) -> None:
        """Mark finished hands done and move to the next one that needs a decision."""
        for h in self.hands:
            if h.busted or (h.total == 21 and len(h.cards) >= 2):
                h.done = True
        while self.active < len(self.hands) and self.hands[self.active].done:
            self.active += 1

    # ---- commands -----------------------------------------------------------

    def deal(self, args: List[str]) -> None:
        if "vs" in args:
            i = args.index("vs")
            mine, theirs = args[:i], args[i + 1:]
        else:
            mine, theirs = args[:-1], args[-1:]
        if len(mine) != 2 or len(theirs) != 1:
            print("Usage: deal <card> <card> vs <dealer upcard>   e.g. deal 10 6 vs 10")
            return
        cards, up = parse_cards(mine), parse_cards(theirs)[0]
        self._new_round()
        self._see(cards + [up])
        self.hands = [Hand(cards)]
        self.up = up
        if self.hands[0].is_blackjack:
            print(green("Blackjack!") + " Enter the dealer's cards with 'dealer ...' to settle.")
            self.hands[0].done = True
        if up == 1:
            adv_tc = self.shoe.true_count if self.use_count else None
            take = adv_tc is not None and adv_tc >= 3
            msg = green("TAKE insurance") if take else "Decline insurance"
            print(f"Dealer shows an ace. {msg} (take it only at TC +3 or higher).")
        self._advance()
        self.show_table()

    def hit(self, args: List[str], double: bool = False) -> None:
        h = self._require_hand()
        if not h:
            return
        if len(args) != 1:
            print(f"Usage: {'double' if double else 'hit'} <card>")
            return
        if double and len(h.cards) != 2:
            print("You can only double on your first two cards.")
            return
        card = parse_cards(args)[0]
        self.shoe.see(card)
        h.cards.append(card)
        if double:
            h.doubled = True
            h.done = True
        if h.from_split and h.cards[0] == 1 and len(h.cards) == 2:
            h.done = True  # split aces get one card each
        if h.busted:
            print(red(f"Bust: {h.describe()}"))
        self._advance()
        self.show_table()

    def split(self) -> None:
        h = self._require_hand()
        if not h:
            return
        if not h.is_pair:
            print("You can only split a pair.")
            return
        a, b = Hand([h.cards[0]], from_split=True), Hand([h.cards[1]], from_split=True)
        self.hands[self.active:self.active + 1] = [a, b]
        self.show_table()

    def finish(self, surrender: bool = False) -> None:
        h = self._require_hand()
        if not h:
            return
        if surrender:
            if len(h.cards) != 2 or h.from_split or not self.rules.late_surrender:
                print("Surrender is only allowed on your first two cards.")
                return
            h.surrendered = True
        h.done = True
        self._advance()
        self.show_table()

    def seen(self, args: List[str]) -> None:
        self._see(parse_cards(args))
        self.show_count()

    def dealer(self, args: List[str]) -> None:
        if self.up is None:
            print("No round in progress. Use 'seen' to count other cards.")
            return
        cards = parse_cards(args)
        self._see(cards)
        d = Hand([self.up] + cards)
        print(f"Dealer: {d.describe()}")
        dealer_bj = d.is_blackjack
        net = 0.0
        for i, h in enumerate(self.hands, 1):
            bet = 2 if h.doubled else 1
            if h.surrendered:
                result, win = "surrendered", -0.5
            elif h.is_blackjack:
                result, win = ("push", 0.0) if dealer_bj else ("blackjack", self.rules.blackjack_pays)
            elif dealer_bj:
                result, win = "lose (dealer blackjack)", -1.0
            elif h.busted:
                result, win = "bust", -bet
            elif d.busted or h.total > d.total:
                result, win = "win", bet
            elif h.total < d.total:
                result, win = "lose", -bet
            else:
                result, win = "push", 0.0
            net += win
            color = green if win > 0 else red if win < 0 else dim
            label = f"Hand {i}: " if len(self.hands) > 1 else "You: "
            print(f"  {label}{h.describe()} → {color(result)} ({win:+g} units)")
        print(f"  Round: {net:+g} units")
        self._new_round()
        self.show_count()

    def next_round(self) -> None:
        self._new_round()
        print(dim("Round cleared."))
        self.show_count()

    def reset(self) -> None:
        self.shoe.reset()
        self._new_round()
        print(f"New shoe: {self.rules.decks} decks, count is 0.")

    def _require_hand(self) -> Optional[Hand]:
        h = self.hand
        if h is None:
            print("No hand in play. Start one with: deal 10 6 vs 10")
        return h

    # ---- display -----------------------------------------------------------

    def show_table(self) -> None:
        if self.up is None:
            return
        for i, h in enumerate(self.hands):
            marker = "▶" if i == self.active else " "
            label = f"Hand {i + 1}" if len(self.hands) > 1 else "You"
            state = ""
            if h.surrendered:
                state = dim("  surrendered")
            elif h.doubled:
                state = dim("  doubled")
            elif h.done and not h.busted and not h.is_blackjack:
                state = dim("  stood")
            print(f"{marker} {label}: {h.describe()}{state}")
        print(f"  Dealer shows: {card_name(self.up)}")

        h = self.hand
        if h is None:
            print(dim("  All hands done. Enter the dealer's cards: dealer <hole card> [draws...]"))
        elif len(h.cards) < 2:
            print(dim("  Give this split hand its second card: hit <card>"))
        else:
            self.show_advice(h)
        self.show_count()

    def show_advice(self, h: Hand) -> None:
        tc = self.shoe.true_count if self.use_count else None
        a = advise(h, self.up, self.rules, self.shoe.probabilities(), tc)
        d = a.decision
        color = {HIT: yellow, STAND: green, DOUBLE: green, SPLIT: green, SURRENDER: red}[d.action]
        line = f"  → {color(d.action)}"
        if d.deviation:
            line += dim(f"   (count deviation; basic strategy says {d.basic_action})")
        print(line)
        print(f"    Bust if you hit: {pct(a.bust_if_hit)}   Dealer busts: {pct(a.dealer_bust)}")
        evs = "  ".join(f"{k.lower()} {signed_pct(v)}" for k, v in a.ev.items())
        if h.is_pair:
            evs += "  (split EV not calculated yet)"
        print(dim(f"    EV  {evs}"))

    def show_count(self) -> None:
        s = count_status(self.shoe, self.rules, self.spread)
        edge = green(f"{s.edge_pct:+.1f}%") if s.edge_pct > 0 else f"{s.edge_pct:+.1f}%"
        print(dim("  ─ ") + f"Running {s.running_count:+d}   True {s.true_count:+.1f}   "
              f"Decks left {s.decks_left:.1f}   Next bet {bold(str(s.bet_units))} unit"
              f"{'s' if s.bet_units != 1 else ''}   Edge ≈ {edge}")

    def show_shoe(self) -> None:
        r = self.shoe.remaining
        full = {v: 4 * self.rules.decks for v in range(1, 10)}
        full[10] = 16 * self.rules.decks
        for v in [1] + list(range(2, 11)):
            name = "A" if v == 1 else ("10s" if v == 10 else str(v))
            print(f"  {name:>3}: {r[v]:3d} / {full[v]}")
        print(f"  {self.shoe.cards_left} cards left")

    # ---- loop ----------------------------------------------------------------

    def run(self) -> None:
        print(bold("Blackjack trainer") + dim(f"  ({self.rules.describe()})"))
        print(dim("Type 'help' for commands. Start a round with: deal 10 6 vs 10"))
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not line:
                continue
            try:
                parts = shlex.split(line)
            except ValueError:
                parts = line.split()
            cmd, args = parts[0].lower(), parts[1:]
            if cmd in ("q", "quit", "exit"):
                return
            try:
                self.dispatch(cmd, args)
            except ValueError as e:
                print(red(str(e)))

    def dispatch(self, cmd: str, args: List[str]) -> None:
        mutating = {"deal", "hit", "h", "double", "d", "split", "p", "stand", "s",
                    "surrender", "r", "seen", "+", "dealer", "next", "n", "reset"}
        if cmd in mutating:
            self._snapshot()
        try:
            self._dispatch(cmd, args)
        except ValueError:
            if cmd in mutating:
                self.shoe, self.hands, self.active, self.up = self.history.pop()
            raise

    def _dispatch(self, cmd: str, args: List[str]) -> None:
        if cmd == "deal":
            self.deal(args)
        elif cmd in ("hit", "h"):
            self.hit(args)
        elif cmd in ("double", "d"):
            self.hit(args, double=True)
        elif cmd in ("split", "p"):
            self.split()
        elif cmd in ("stand", "s"):
            self.finish()
        elif cmd in ("surrender", "r"):
            self.finish(surrender=True)
        elif cmd in ("seen", "+"):
            self.seen(args)
        elif cmd == "dealer":
            self.dealer(args)
        elif cmd in ("next", "n"):
            self.next_round()
        elif cmd in ("count", "c"):
            self.show_count()
        elif cmd == "shoe":
            self.show_shoe()
        elif cmd in ("undo", "u"):
            self.undo()
        elif cmd == "reset":
            self.reset()
        elif cmd == "rules":
            print(self.rules.describe() + f"   house edge ≈ {self.rules.house_edge():.2f}%")
        elif cmd in ("help", "?"):
            print(HELP)
        else:
            print(f"Unknown command {cmd!r}. Type 'help'.")


def _rules_from(ns: argparse.Namespace) -> Rules:
    return Rules(
        decks=ns.decks,
        hit_soft_17=not ns.s17,
        double_after_split=not ns.no_das,
        late_surrender=not ns.no_surrender,
        blackjack_pays=1.2 if ns.six_five else 1.5,
    )


def _one_shot(ns: argparse.Namespace, cards: List[str]) -> None:
    rules = _rules_from(ns)
    if "vs" not in cards:
        sys.exit("Usage: advise <your cards...> vs <dealer upcard>   e.g. advise 10 6 vs 10")
    i = cards.index("vs")
    mine, up = parse_cards(cards[:i]), parse_cards(cards[i + 1:])[0]
    shoe = Shoe(rules.decks)
    for c in mine + [up]:
        shoe.see(c)
    session = Session(rules, ns.spread, use_count=ns.tc is not None)
    session.shoe = shoe
    session.hands, session.up = [Hand(mine)], up
    if ns.tc is not None:
        # Pretend the shoe's running count gives the requested true count.
        shoe.running_count = round(ns.tc * shoe.decks_left)
    session.show_table()


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="bj", description="Blackjack strategy and Hi-Lo counting trainer")
    p.add_argument("--decks", type=int, default=6)
    p.add_argument("--s17", action="store_true", help="dealer stands on soft 17 (default: hits)")
    p.add_argument("--no-das", action="store_true", help="no double after split")
    p.add_argument("--no-surrender", action="store_true")
    p.add_argument("--six-five", action="store_true", help="blackjack pays 6:5")
    p.add_argument("--spread", type=int, default=8, help="max bet in units (default 8)")
    p.add_argument("--basic", action="store_true", help="basic strategy only, ignore count deviations")
    p.add_argument("--tc", type=float, help="advise mode: true count to assume")
    p.add_argument("command", nargs="*", help="optional: advise <cards> vs <upcard>")
    ns = p.parse_args(argv)

    if ns.command and ns.command[0] == "advise":
        try:
            _one_shot(ns, ns.command[1:])
        except ValueError as e:
            sys.exit(str(e))
        return
    Session(_rules_from(ns), ns.spread, use_count=not ns.basic).run()
