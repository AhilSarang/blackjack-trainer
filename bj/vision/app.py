"""Live blackjack trainer: point a camera at the table and it reads the cards.

    python -m bj.vision                 use camera 0 (press C to switch cameras)
    python -m bj.vision --camera 1
    python -m bj.vision --video game.mp4   or a photo: --video table.jpg

Put the dealer's cards in the top part of the picture and yours in the bottom.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..advisor import advise, count_status
from ..counter import Shoe
from ..rules import Rules
from ..strategy import DOUBLE, HIT, SPLIT, STAND, SURRENDER
from .detector import CardDetector
from .hud import draw_hud
from .table import Table

MODELS = Path(__file__).resolve().parents[2] / "models"
WINDOW = "Blackjack Trainer"

# BGR colours
WHITE, GREY, BLACK = (255, 255, 255), (170, 170, 170), (0, 0, 0)
GREEN, YELLOW, RED, BLUE, PURPLE = (90, 220, 90), (0, 205, 255), (70, 70, 255), (255, 170, 40), (230, 110, 220)
ACTION_COLOR = {HIT: YELLOW, STAND: GREEN, DOUBLE: BLUE, SPLIT: PURPLE, SURRENDER: RED}
FONT = cv2.FONT_HERSHEY_DUPLEX


def dealer_finished(dealer, rules: Rules) -> bool:
    t = dealer.total
    return t > 21 or t > 17 or (t == 17 and not (dealer.soft and rules.hit_soft_17))


def read_table(table: Table, rules: Rules, spread: int) -> Tuple[str, Tuple[int, int, int], List[str]]:
    """What to show: (headline, colour, detail lines). The last two lines are always
    the count and the next bet, so the bet advice is on screen the whole time."""
    you, dealer = table.player_hand, table.dealer_hand
    cs = count_status(table.shoe, rules, spread)
    if cs.bet_units >= 2:
        bet = f"Next bet: {cs.bet_units} units - BET HIGH"
    else:
        bet = "Next bet: 1 unit - bet minimum"
    footer = [f"Count {cs.running_count:+d}  (true {cs.true_count:+.1f})", bet]

    if not you.cards and not dealer.cards:
        return "Waiting for cards", WHITE, footer

    lines = [f"You: {you.total if you.cards else '-'}    Dealer: {dealer.total if dealer.cards else '-'}"]

    if len(dealer.cards) >= 2 or (you.busted and dealer.cards):
        # Dealer's turn (or you're already out): show the result once it's settled.
        if you.busted:
            return "BUST", RED, lines + footer
        if not dealer_finished(dealer, rules):
            return "Dealer drawing...", WHITE, lines + footer
        if you.is_blackjack and not dealer.is_blackjack:
            head, col = "BLACKJACK!", GREEN
        elif dealer.busted or you.total > dealer.total:
            head, col = "YOU WIN", GREEN
        elif you.total < dealer.total:
            head, col = "DEALER WINS", RED
        else:
            head, col = "PUSH", WHITE
        return head, col, lines + footer

    if len(you.cards) < 2 or not dealer.cards:
        return "Dealing...", WHITE, lines + footer
    if you.busted:
        return "BUST", RED, lines + footer
    if you.is_blackjack:
        return "BLACKJACK!", GREEN, lines + footer

    tc = table.shoe.true_count
    a = advise(you, dealer.cards[0], rules, table.shoe.probabilities(), tc)
    lines.append(f"Bust if you hit: {a.bust_if_hit:.0%}   Dealer busts: {a.dealer_bust:.0%}")
    if a.insurance is not None and len(you.cards) == 2:
        lines.append("Insurance: YES" if a.insurance else "Insurance: no")
    if a.decision.deviation:
        lines.append(f"(count play - basic says {a.decision.basic_action.lower()})")
    return a.decision.action, ACTION_COLOR[a.decision.action], lines + footer


# ---- drawing ------------------------------------------------------------------

def _text(img, text, org, scale, color, thick=1):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _shade(img, x1, y1, x2, y2, alpha=0.6):
    """Darken a rectangle so text on top of it is readable."""
    roi = img[max(0, y1):y2, max(0, x1):x2]
    roi[:] = (roi * (1 - alpha)).astype(np.uint8)


def draw(frame: np.ndarray, table: Table, headline: str, color, lines: List[str],
         flash: Optional[str], fps: float, hints: bool = True) -> np.ndarray:
    h, w = frame.shape[:2]
    s = w / 1280
    out = frame.copy()

    # Dealer / you divider.
    y = int(table.divider * h)
    cv2.line(out, (0, y), (w, y), WHITE, max(1, int(2 * s)), cv2.LINE_AA)
    _shade(out, int(w - 160 * s), y - int(40 * s), w, y + int(44 * s))
    _text(out, "DEALER", (int(w - 150 * s), y - int(12 * s)), 0.7 * s, WHITE)
    _text(out, "YOU", (int(w - 150 * s), y + int(32 * s)), 0.7 * s, WHITE)

    # Cards.
    for c in table.cards:
        cx, cy = int(c.x), int(c.y)
        if c.ignored:
            _text(out, f"{c.code} (ignored)", (cx - int(40 * s), cy), 0.6 * s, GREY)
            continue
        col = GREEN if c.counted else GREY
        label = c.code
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.9 * s, 2)
        pad = int(8 * s)
        cv2.rectangle(out, (cx - tw // 2 - pad, cy - th - pad), (cx + tw // 2 + pad, cy + pad), col, -1)
        cv2.putText(out, label, (cx - tw // 2, cy), FONT, 0.9 * s, BLACK, 2, cv2.LINE_AA)

    # Advice panel.
    panel_w, line_h = int(560 * s), int(34 * s)
    panel_h = int(100 * s) + line_h * len(lines)
    _shade(out, 0, 0, panel_w, panel_h)
    _text(out, headline, (int(20 * s), int(70 * s)), 1.8 * s, color, max(2, int(3 * s)))
    for i, line in enumerate(lines):
        col = GREEN if "BET HIGH" in line else WHITE
        _text(out, line, (int(22 * s), int(110 * s) + i * line_h), 0.75 * s, col, 2 if "BET HIGH" in line else 1)

    if not (flash or hints):
        return out
    _shade(out, 0, h - int(100 * s if flash else 44 * s), w, h)
    if flash:
        _text(out, flash, (int(20 * s), h - int(60 * s)), 0.9 * s, GREEN, 2)
    if hints:
        _text(out, f"Q quit  N new round  R new shoe  X remove last card  W/S move line  C camera   {fps:.0f} fps",
              (int(20 * s), h - int(20 * s)), 0.55 * s, GREY)
    return out


# ---- main loop ------------------------------------------------------------------

def open_source(args, camera: int):
    if args.video:
        if Path(args.video).suffix.lower() in (".jpg", ".jpeg", ".png", ".heic", ".webp"):
            img = cv2.imread(args.video)
            if img is None:
                sys.exit(f"Couldn't read {args.video}")
            return None, img
        return cv2.VideoCapture(args.video), None
    cap = cv2.VideoCapture(camera, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap, None


class DecisionLog:
    """Writes a line each time the advice or the result changes, for reviewing a session."""

    RECORD = set(ACTION_COLOR) | {"BUST", "BLACKJACK!", "YOU WIN", "DEALER WINS", "PUSH"}

    def __init__(self):
        self.lines: List[str] = []
        self._last = None

    def note(self, t: float, table: Table, headline: str, lines: List[str]) -> None:
        you, dealer = table.player_hand, table.dealer_hand
        key = (headline, tuple(you.cards), tuple(dealer.cards))
        if headline not in self.RECORD or key == self._last:
            return
        self._last = key
        stamp = f"{int(t // 60):02d}:{t % 60:04.1f}"
        hand = f"You {you.total} ({' '.join(c.code for c in table.counted_cards if c.zone == 'player')})"
        up = f"Dealer {dealer.total} ({' '.join(c.code for c in table.counted_cards if c.zone == 'dealer')})"
        count = next((l for l in lines if l.startswith("Count")), "")
        self.lines.append(f"{stamp}  Round {table.rounds_played + 1}:  {hand}  vs  {up}  ->  {headline}   {count}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="bj.vision", description="Live blackjack trainer")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", help="analyze a video file or photo instead of a camera")
    p.add_argument("--save", help="write a copy of the video with the overlay drawn on it (.mp4)")
    p.add_argument("--no-window", action="store_true", help="don't show a window (faster for --video)")
    p.add_argument("--decks", type=int, default=6)
    p.add_argument("--s17", action="store_true", help="dealer stands on soft 17")
    p.add_argument("--spread", type=int, default=8)
    p.add_argument("--conf", type=float, default=0.45, help="detection confidence (0-1)")
    p.add_argument("--confirm", type=int, default=5, help="frames a card must be seen before it counts")
    p.add_argument("--size", type=int, default=960, help="model input size; bigger reads small cards better but is slower")
    p.add_argument("--divider", type=float, default=0.45,
                   help="dealer/you line as a fraction of frame height (0.45 = a bit above the middle)")
    p.add_argument("--band", type=float, default=0.3,
                   help="height of the dealer's strip above the line that's read at full resolution")
    p.add_argument("--step", type=int, default=0,
                   help="with --video, read every Nth frame (default: about 8 frames per second of video)")
    p.add_argument("--fast", action="store_true",
                   help="with --video, use the quicker live-camera reading (less accurate on far cards)")
    args = p.parse_args(argv)

    model, classes = MODELS / "model.onnx", MODELS / "model.classes.json"
    if not model.exists() or not classes.exists():
        sys.exit("Card model not found. Run the setup first (see README: 'Live camera').")

    rules = Rules(decks=args.decks, hit_soft_17=not args.s17)
    detector = CardDetector(model, classes, conf=args.conf, size=args.size)
    table = Table(Shoe(rules.decks), divider=args.divider, confirm_frames=args.confirm)
    camera = args.camera
    cap, still = open_source(args, camera)
    if cap is not None and not cap.isOpened():
        if args.video:
            sys.exit(f"Couldn't open {args.video}")
        sys.exit(f"Couldn't open camera {camera}. Try --camera 1, and check that Terminal "
                 "has camera access (System Settings > Privacy & Security > Camera).")

    # With a video file, time comes from the video itself, so results don't depend
    # on how fast this computer processes frames.
    is_file = cap is not None and bool(args.video)
    src_fps = (cap.get(cv2.CAP_PROP_FPS) or 30) if is_file else 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
    accurate = bool(args.video) and not args.fast
    step = args.step or max(1, round(src_fps / 8))
    show = not args.no_window
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        print("Running. Click the video window, then press Q to quit.")
    writer = None
    log = DecisionLog()
    flash, flash_until, fps, last = None, 0.0, 0.0, time.time()
    frame_no, started = 0, time.time()

    while True:
        if still is not None:
            frame = still
        else:
            ok, frame = cap.read()
            if not ok:
                if args.video:
                    break
                time.sleep(0.05)
                continue
        frame_no += 1

        now = frame_no / src_fps if is_file else time.time() - started
        h, w = frame.shape[:2]
        events = []
        if not is_file or frame_no % step == 0:
            events = table.update(detector.detect_table(frame, table.divider, args.band, accurate), now, w, h)
        if events:
            flash, flash_until = "  ".join(events), now + 2.0
            if show:
                for e in events:
                    print(e)

        headline, color, lines = read_table(table, rules, args.spread)
        log.note(now, table, headline, lines)
        wall = time.time()
        fps = 0.9 * fps + 0.1 / max(wall - last, 1e-3)
        last = wall
        shown = draw_hud(frame, table, rules, args.spread, headline, color, log.lines,
                         flash if now < flash_until else None, fps, hints=show and not args.save)

        if args.save:
            if writer is None:
                writer = _open_writer(args.save, src_fps, w, h)
            writer.write(shown)
        if is_file and not show and total_frames and frame_no % max(1, total_frames // 20) == 0:
            print(f"  {frame_no * 100 // total_frames}%  ({frame_no}/{total_frames} frames)")

        if not show:
            if still is not None and frame_no >= args.confirm + 2:
                break
            continue
        cv2.imshow(WINDOW, shown)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("n"):
            table.end_round()
            flash, flash_until = "New round", now + 1.5
        elif key == ord("r"):
            table.shoe.reset()
            table.end_round()
            flash, flash_until = "New shoe - count reset to 0", now + 2.0
        elif key == ord("x"):
            code = table.remove_last()
            flash, flash_until = (f"Removed {code}" if code else "Nothing to remove"), now + 1.5
        elif key == ord("w"):
            table.divider = max(0.1, table.divider - 0.03)
        elif key == ord("s"):
            table.divider = min(0.9, table.divider + 0.03)
        elif key == ord("c") and not is_file and still is None:
            cap.release()
            camera = (camera + 1) % 5
            cap, _ = open_source(args, camera)
            if not cap.isOpened():
                camera = 0
                cap, _ = open_source(args, camera)
            flash, flash_until = f"Camera {camera}", now + 1.5

    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    _report(args, table, log)


def _open_writer(path: str, fps: float, w: int, h: int):
    for fourcc in ("avc1", "mp4v"):   # H.264 if this OpenCV build has it, else MPEG-4
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if writer.isOpened():
            return writer
    sys.exit(f"Couldn't create {path}")


def _report(args, table: Table, log: DecisionLog) -> None:
    cs = table.shoe
    rounds = table.rounds_played + (1 if table.counted_cards else 0)
    summary = [f"Rounds: {rounds}   Cards counted: {cs.cards_seen}   "
               f"Final count: {cs.running_count:+d} (true {cs.true_count:+.1f})", ""]
    text = "\n".join(summary + (log.lines or ["No decisions recorded."]))
    print("\n" + text)
    if args.save:
        notes = Path(args.save).with_suffix(".txt")
        notes.write_text(text + "\n")
        print(f"\nSaved video: {args.save}\nSaved notes: {notes}")


if __name__ == "__main__":
    main()
