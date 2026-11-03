import unittest

from bj import Hand, Rules, Shoe, advise, decide, parse_card
from bj.counter import bet_units
from bj.odds import BUST, Odds
from bj.strategy import DOUBLE, HIT, SPLIT, STAND, SURRENDER, take_insurance

H17 = Rules()
S17 = Rules(hit_soft_17=False)
NO_SURRENDER = Rules(late_surrender=False)


def hand(*cards, split=False):
    return Hand([parse_card(str(c)) for c in cards], from_split=split)


def infinite_deck():
    return {v: (4 / 13 if v == 10 else 1 / 13) for v in range(1, 11)}


class TestCards(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_card("Kh"), 10)
        self.assertEqual(parse_card("a♠"), 1)
        self.assertEqual(parse_card("10"), 10)
        with self.assertRaises(ValueError):
            parse_card("Z")

    def test_totals(self):
        self.assertEqual((hand("A", 6).total, hand("A", 6).soft), (17, True))
        self.assertEqual((hand("A", 6, 10).total, hand("A", 6, 10).soft), (17, False))
        self.assertTrue(hand("A", "K").is_blackjack)
        self.assertFalse(hand("A", "K", split=True).is_blackjack)
        self.assertTrue(hand(10, 6, 9).busted)


class TestBasicStrategy(unittest.TestCase):
    def play(self, h, up, rules=H17):
        return decide(h, parse_card(str(up)), rules).action

    def test_hard(self):
        self.assertEqual(self.play(hand(10, 2), 2), HIT)
        self.assertEqual(self.play(hand(10, 2), 4), STAND)
        self.assertEqual(self.play(hand(6, 5), "A"), DOUBLE)
        self.assertEqual(self.play(hand(6, 5), "A", S17), HIT)
        self.assertEqual(self.play(hand(5, 4), 3), DOUBLE)
        self.assertEqual(self.play(hand(4, 3, 2), 3), HIT)  # can't double on 3 cards
        self.assertEqual(self.play(hand(10, 7), 10), STAND)

    def test_surrender(self):
        self.assertEqual(self.play(hand(10, 6), 10), SURRENDER)
        self.assertEqual(self.play(hand(10, 6), 10, NO_SURRENDER), HIT)
        self.assertEqual(self.play(hand(10, 7), "A"), SURRENDER)
        self.assertEqual(self.play(hand(10, 7), "A", S17), STAND)

    def test_soft(self):
        self.assertEqual(self.play(hand("A", 7), 3), DOUBLE)
        self.assertEqual(self.play(hand("A", 7), 9), HIT)
        self.assertEqual(self.play(hand("A", 2, 5), 3), STAND)  # soft 18, can't double
        self.assertEqual(self.play(hand("A", 8), 6), DOUBLE)
        self.assertEqual(self.play(hand("A", 8), 6, S17), STAND)

    def test_pairs(self):
        self.assertEqual(self.play(hand("A", "A"), 10), SPLIT)
        self.assertEqual(self.play(hand(8, 8), 10), SPLIT)
        self.assertEqual(self.play(hand(8, 8), "A"), SURRENDER)
        self.assertEqual(self.play(hand(8, 8), "A", NO_SURRENDER), SPLIT)
        self.assertEqual(self.play(hand(9, 9), 7), STAND)
        self.assertEqual(self.play(hand(10, "K"), 6), STAND)
        self.assertEqual(self.play(hand(5, 5), 9), DOUBLE)
        self.assertEqual(self.play(hand(4, 4), 5), SPLIT)


class TestDeviations(unittest.TestCase):
    def play(self, h, up, tc, rules=H17):
        return decide(h, parse_card(str(up)), rules, tc)

    def test_16_vs_10(self):
        self.assertEqual(self.play(hand(10, 3, 3), 10, 0.5).action, STAND)
        self.assertEqual(self.play(hand(10, 3, 3), 10, -1).action, HIT)
        self.assertTrue(self.play(hand(10, 3, 3), 10, 0.5).deviation)

    def test_12_vs_3(self):
        self.assertEqual(self.play(hand(10, 2), 3, 2).action, STAND)
        self.assertEqual(self.play(hand(10, 2), 3, 1).action, HIT)

    def test_tens_vs_6(self):
        self.assertEqual(self.play(hand(10, "Q"), 6, 4).action, SPLIT)
        self.assertEqual(self.play(hand(10, "Q"), 6, 3).action, STAND)

    def test_surrender_index(self):
        self.assertEqual(self.play(hand(10, 4), 10, 3).action, SURRENDER)
        self.assertEqual(self.play(hand(10, 4), 10, 2).action, HIT)
        # 15 vs 10: surrender at TC >= 0, otherwise hit (not stand)
        self.assertEqual(self.play(hand(10, 5), 10, -1).action, HIT)
        self.assertEqual(self.play(hand(10, 5), 10, 0).action, SURRENDER)

    def test_insurance(self):
        self.assertTrue(take_insurance(1, 3.0))
        self.assertFalse(take_insurance(1, 2.9))
        self.assertFalse(take_insurance(1, None))


class TestCounting(unittest.TestCase):
    def test_hi_lo(self):
        shoe = Shoe(6)
        for c in [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]:
            shoe.see(c)
        self.assertEqual(shoe.running_count, 3)
        self.assertEqual(shoe.cards_left, 312 - 10)

    def test_true_count(self):
        shoe = Shoe(2)
        for c in [2, 3, 4, 5, 6] * 5 + [2]:
            shoe.see(c)  # 26 low cards: RC +26 with 1.5 decks left
        self.assertAlmostEqual(shoe.true_count, 26 / 1.5)

    def test_empty_rank(self):
        shoe = Shoe(1)
        for _ in range(4):
            shoe.see(1)
        with self.assertRaises(ValueError):
            shoe.see(1)

    def test_bet_ramp(self):
        self.assertEqual(bet_units(-3), 1)
        self.assertEqual(bet_units(2.7), 2)
        self.assertEqual(bet_units(20, spread=8), 8)


class TestOdds(unittest.TestCase):
    def test_dealer_outcomes_sum_to_one(self):
        for up in range(1, 11):
            o = Odds(infinite_deck(), up, H17)
            self.assertAlmostEqual(sum(o.dealer.values()), 1.0)

    def test_known_dealer_bust_rates(self):
        # Infinite-deck reference values (H17): vs 6 ≈ 44%, vs 5 ≈ 42%, vs 10 (no BJ) ≈ 23%
        self.assertAlmostEqual(Odds(infinite_deck(), 6, H17).dealer[BUST], 0.442, delta=0.01)
        self.assertAlmostEqual(Odds(infinite_deck(), 5, H17).dealer[BUST], 0.418, delta=0.01)
        self.assertAlmostEqual(Odds(infinite_deck(), 10, H17).dealer[BUST], 0.230, delta=0.01)

    def test_known_evs(self):
        o = Odds(infinite_deck(), 10, H17)
        self.assertAlmostEqual(o.ev_stand(16), -0.540, delta=0.01)
        self.assertAlmostEqual(o.ev_hit(16, False), -0.535, delta=0.01)
        o = Odds(infinite_deck(), 6, H17)
        self.assertGreater(o.ev_double(11, False), o.ev_hit(11, False))

    def test_bust_if_hit(self):
        o = Odds(infinite_deck(), 10, H17)
        self.assertAlmostEqual(o.bust_if_hit(12), 4 / 13)
        self.assertEqual(o.bust_if_hit(11), 0)


class TestAdvisor(unittest.TestCase):
    def test_advise(self):
        a = advise(hand(10, 6), 10, H17, infinite_deck(), true_count=None)
        self.assertEqual(a.decision.action, SURRENDER)
        self.assertEqual(a.ev[SURRENDER], -0.5)
        self.assertIsNone(a.insurance)
        a = advise(hand(10, 6), 1, H17, infinite_deck(), true_count=4)
        self.assertTrue(a.insurance)


if __name__ == "__main__":
    unittest.main()
