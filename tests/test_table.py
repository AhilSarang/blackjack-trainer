import unittest
from types import SimpleNamespace as S

from bj.counter import Shoe
from bj.vision.table import Table

W, H = 1280, 720


def see(code, x, y, conf=0.9):
    return S(code=code, x=x, y=y, conf=conf)


class TestTable(unittest.TestCase):
    def setUp(self):
        self.t = Table(Shoe(6), confirm_frames=3, clear_seconds=2.0)
        self.now = 0.0

    def frames(self, sightings, n=1):
        events = []
        for _ in range(n):
            self.now += 0.05
            events += self.t.update(sightings, self.now, W, H)
        return events

    def test_card_counted_once_after_confirming(self):
        self.frames([see("KS", 600, 500)], 2)
        self.assertEqual(self.t.shoe.cards_seen, 0)
        events = self.frames([see("KS", 600, 500)], 10)
        self.assertEqual(events, ["You: KS"])
        self.assertEqual(self.t.shoe.cards_seen, 1)
        self.assertEqual(self.t.shoe.running_count, -1)

    def test_one_frame_misread_is_ignored(self):
        self.frames([see("5H", 600, 500)])
        self.frames([], 20)
        self.assertEqual(self.t.shoe.cards_seen, 0)
        self.assertEqual(self.t.cards, [])

    def test_zones_and_hands(self):
        table = [see("10D", 600, 150), see("6C", 550, 500), see("10H", 650, 510)]
        self.frames(table, 5)
        self.assertEqual(self.t.dealer_hand.cards, [10])
        self.assertEqual(sorted(self.t.player_hand.cards), [6, 10])

    def test_covered_card_stays_in_round(self):
        self.frames([see("9S", 600, 500)], 5)
        self.frames([], 10)        # a hand covers it for half a second
        self.frames([see("9S", 610, 505)], 5)
        self.assertEqual(self.t.shoe.cards_seen, 1)
        self.assertEqual(self.t.player_hand.cards, [9])

    def test_card_slid_into_place_not_double_counted(self):
        self.frames([see("QH", 300, 500)], 3)   # counted while the dealer slides it in
        self.frames([see("QH", 900, 600)], 5)
        self.assertEqual(self.t.shoe.cards_seen, 1)
        self.assertEqual(self.t.player_hand.cards, [10])

    def test_second_copy_of_a_card_counts_separately(self):
        # Multi-deck shoe: a 3S in your hand and another 3S dealt to the dealer.
        self.frames([see("3S", 600, 560), see("9H", 640, 520)], 8)
        self.frames([see("3S", 600, 560), see("9H", 640, 520), see("3S", 400, 120)], 8)
        self.assertEqual(self.t.shoe.cards_seen, 3)
        self.assertEqual(self.t.dealer_hand.cards, [3])

    def test_hit_cards_fanning_past_the_line_stay_yours(self):
        # Each hit overlaps the last card, stepping up toward the dealer (the line is y=324).
        table = [see("10D", 640, 120), see("9C", 600, 560)]
        self.frames(table, 8)
        for code, x, y in [("2S", 630, 480), ("AH", 660, 400), ("3D", 690, 320)]:
            table.append(see(code, x, y))
            self.frames(table, 8)
        self.assertEqual(self.t.dealer_hand.cards, [10])
        self.assertEqual(self.t.player_hand.cards, [9, 2, 1, 3])

    def test_moving_card_waits_before_joining_a_hand(self):
        for x in range(100, 700, 60):              # sliding across the table
            self.frames([see("6H", x, 500)])
        self.assertEqual(self.t.shoe.cards_seen, 1)   # already counted...
        self.assertEqual(self.t.player_hand.cards, [])  # ...but not placed yet
        self.frames([see("6H", 700, 500)], 6)
        self.assertEqual(self.t.player_hand.cards, [6])

    def test_sweep_does_not_change_sides(self):
        table = [see("KH", 500, 150), see("7C", 700, 150), see("9C", 600, 500), see("10S", 650, 520)]
        self.frames(table, 10)                   # dealer 17: the round is decided
        self.assertTrue(self.t.frozen)
        swept = [see("KH", 600, 600), see("7C", 600, 600), see("9C", 600, 200), see("10S", 600, 200)]
        self.frames(swept, 6)
        self.assertEqual(self.t.dealer_hand.cards, [10, 7])
        self.assertEqual(sorted(self.t.player_hand.cards), [9, 10])

    def test_decided_round_ends_quickly(self):
        self.frames([see("KH", 500, 150), see("7C", 700, 150), see("9C", 600, 500)], 10)
        events = self.frames([], 18)            # 0.9 s
        self.assertIn("Round over", events)

    def test_round_ends_when_table_clears(self):
        self.frames([see("2S", 600, 500), see("3S", 640, 500)], 5)
        events = self.frames([], 45)            # 2.25 s empty
        self.assertIn("Round over", events)
        self.assertEqual(self.t.player_hand.cards, [])
        self.assertEqual(self.t.shoe.running_count, 2)  # count survives the round

    def test_remove_misread(self):
        self.frames([see("AH", 600, 500)], 5)
        self.assertEqual(self.t.remove_last(), "AH")
        self.frames([see("AH", 600, 500)], 10)  # still on the table, but not re-counted
        self.assertEqual(self.t.shoe.cards_seen, 0)
        self.assertEqual(self.t.player_hand.cards, [])


if __name__ == "__main__":
    unittest.main()
