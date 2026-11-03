# Blackjack engine

The strategy and counting brain for a camera-based blackjack trainer. Right now
you type cards in. Later the vision pipeline (YOLO card detection → table zones →
round state machine) will feed the same `advise()` call.

**Legal note:** counting in your head is legal, but using a device to count or
calculate odds inside a casino is illegal in most US jurisdictions (a felony in
Nevada). Use this for practice, analysis, and content only.

## Run

Needs Python 3.9 or later. There are no dependencies.

```bash
python3 -m bj                        # interactive trainer (6 decks, H17, DAS, late surrender)
python3 -m bj --s17 --decks 8        # other table rules (see --help)
python3 -m bj advise 10 6 vs 10      # one-off advice
python3 -m bj advise 12 vs 3 --tc 2  # ...with a true count applied
python3 -m unittest                  # tests
```

Example session:

```
> deal 10 3 vs 10
> hit 3
▶ You: 10 3 3  (hard 16)
  Dealer shows: 10
  → STAND   (count deviation; basic strategy says HIT)
    Bust if you hit: 62%   Dealer busts: 23%
  ─ Running +6   True +1.0   Decks left 5.9   Next bet 1 unit   Edge ≈ -0.1%
> dealer 7
```

Commands: `deal`, `hit`/`h`, `double`/`d`, `split`, `stand`/`s`,
`surrender`/`r`, `seen`/`+` (other players' cards), `dealer` (settles the round),
`next`, `count`, `shoe`, `undo`, `reset`, `help`.

## Live camera

Double-click **Live Blackjack.command** in the Blackjack Trainer folder. Or run:

```bash
.venv/bin/python -m bj.vision              # camera 0; press C to cycle cameras
.venv/bin/python -m bj.vision --camera 1
.venv/bin/python -m bj.vision --video clip.mp4   # or a photo
```

### Analyzing a recorded video

Double-click **Analyze Video.command**, pick a video, and it saves next to it:
- `<name> - analyzed.mp4`: the video with cards labeled and the advice drawn on it
- `<name> - analyzed.txt`: a timestamped list of every decision, the result, and the count

```bash
.venv/bin/python -m bj.vision --video hand.mov --save hand-analyzed.mp4 --no-window
```

It takes about twice the video's length on an M4 (it reads every 2nd frame at 960 px; `--step 1` reads every frame).
See `Examples/` for a sample.

Setup:
- Point the camera **straight down** at the table.
- The **dealer's cards go above the white line** and **yours go below it**. Press W/S to move the line.
- Keep the card corners visible. The model reads the corner (rank and suit).

How it decides:
- A card is counted after it's seen in 5 frames, so brief misreads are ignored.
- A card covered by your hand stays in the round.
- The round ends after the table has been empty for 2 seconds. The count carries over to the next round.
- Keys: **N** new round, **R** new shoe (reset the count), **X** undo a misread card, **Q** quit.

Model: [`sroot/lgd-cards-gen3`](https://huggingface.co/sroot/lgd-cards-gen3), YOLO11s, 52 classes, ONNX, run with
onnxruntime on CPU at 960 px input (about 120 ms per frame on an M4; at the model's
native 640 px, small or tilted corners got misread). It's AGPL-3.0, so the weights aren't bundled here — download `model.onnx` from the HuggingFace page into `models/`. See `models/MODEL_README.md`.

To set up the environment again on a fresh machine:

```bash
python3.12 -m venv .venv && .venv/bin/pip install opencv-python onnxruntime numpy
```

Not yet handled live: splits (two hands in the YOU area are summed as one), and other players' hands.

## What it does

| Piece | File |
|---|---|
| Card parsing, hand totals (soft/hard, pairs, blackjack) | `bj/cards.py` |
| Table rules and approximate house edge | `bj/rules.py` |
| Hi-Lo running/true count, shoe composition, bet ramp, edge | `bj/counter.py` |
| Basic strategy (4–8 decks, H17/S17) + Illustrious 18 + Fab 4 surrender | `bj/strategy.py` |
| Dealer outcome odds, bust %, EV of stand/hit/double from the live shoe | `bj/odds.py` |
| `advise()` / `count_status()`: the API the vision pipeline will call | `bj/advisor.py` |
| Interactive trainer (typed) | `bj/cli.py` |
| Card detector, round tracker, live overlay | `bj/vision/` |

## Known limits
- EVs treat the remaining shoe's card frequencies as fixed for the rest of the hand. That's accurate to a fraction of a percent until the very end of a shoe.
- Split EV isn't calculated yet; the split *decision* comes from the strategy tables.
- Doesn't enforce a limit on re-splits. Split aces get one card each.

## Next: vision
1. Fine-tune YOLO11 on playing-card corner indices (Roboflow has 52-class datasets).
2. Map the table with a homography: dealer zone, player spots, discard tray.
3. Track cards with ByteTrack; a state machine detects deal → actions → dealer → clear.
4. Each state change calls `shoe.see(card)` and `advise(...)`; draw the result as an overlay.
