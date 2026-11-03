---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
base_model: Ultralytics/YOLO11
tags:
- object-detection
- ultralytics
- yolo11
- onnx
- onnxruntime
- playing-cards
- card-detection
- casino
- live-game-defender
model-index:
- name: lgd-cards-gen3
  results:
  - task:
      type: object-detection
    dataset:
      type: lgd-poc-video-holdout
      name: LGD frozen PoC table-video holdout (225 frames, private)
    metrics:
    - type: recall
      value: 0.847
      name: Pip recall @50 (code-aware)
    - type: precision
      value: 0.771
      name: Precision proxy
---

# LGD Cards — Gen 3 (day-2 video) · YOLO11s playing-card detector, 52 classes ✅ CURRENT

The **current** generation of the **Live Game Defender (LGD)** playing-card detector, and the model
LGD serves in production. Gen 2 further fine-tuned on a **second day of real casino-table video**
(plus a human-QA correction pass), lifting frozen-holdout recall to **0.85**. Locates every playing
card in a frame and names it by **rank + suit** (52 classes); an empty frame yields no detection
(built-in "is there a card?" gate).

## Generations

| Gen | Repo | Trained on | Frozen real-video holdout recall | Status |
|---|---|---|---|---|
| 1 | [lgd-cards-gen1](https://huggingface.co/sroot/lgd-cards-gen1) | Roboflow `ow27d` v4 dataset | — (dataset-val only) | superseded |
| 2 | [lgd-cards-gen2](https://huggingface.co/sroot/lgd-cards-gen2) | + day-1 PoC table video | 0.68 | superseded |
| 3 | **lgd-cards-gen3** (this) | + day-2 PoC table video | **0.85** | ✅ **current — the production/served model** |
| 4 | [lgd-cards-gen4](https://huggingface.co/sroot/lgd-cards-gen4) | + day-3 deck-spread video | 0.786 ¹ | spread specialist — activated 2026-07-20, reverted same day (gameplay regressions); not production-approved |

¹ gen4's 0.786 is on an expanded 318-frame holdout where gen3 scores 0.750 — compare only within
one suite. Naming unified 2026-07-20: `genN` means the same model on HF, on disk and in every doc.

Chip detectors: [lgd-chips-gen1](https://huggingface.co/sroot/lgd-chips-gen1) · [lgd-chips-gen2](https://huggingface.co/sroot/lgd-chips-gen2).

## Classes (52)

```
10C 10D 10H 10S 2C 2D 2H 2S 3C 3D 3H 3S 4C 4D 4H 4S 5C 5D 5H 5S
6C 6D 6H 6S 7C 7D 7H 7S 8C 8D 8H 8S 9C 9D 9H 9S
AC AD AH AS JC JD JH JS KC KD KH KS QC QD QH QS
```

`C`=Clubs, `D`=Diamonds, `H`=Hearts, `S`=Spades. Full order is in `model.classes.json`.

## Files

- `model.onnx` — ONNX export (run with `onnxruntime`); this is the exact file LGD serves.
- `best.pt` — Ultralytics PyTorch weights (source for the ONNX export; use for further fine-tuning).
- `model.classes.json` — ordered class-name sidecar (index → card code).
- `data.yaml` — class list / dataset config.
- `train-args.yaml` — the exact Ultralytics training arguments for this run.
- `metrics.json` — frozen-holdout verdict vs. the previous generation.

## Training

- **Base:** Ultralytics `yolo11s.pt` (COCO-pretrained), `imgsz=640` (resumed once from epoch 15).
- **Data:** Roboflow `ow27d` base **+ day-1 PoC video + day-2 PoC video**, auto-labeled (detector
  proposes pip boxes; an LLM verifies/names over a closed 52-code vocabulary) and materialized as
  serve-matching tiles — **7,526 train / 1,103 val tiles**, with **14 human-QA label corrections**.
  Labeling cost **$8.13**.
- **Hardware:** NVIDIA RTX 3060 (12 GB).

## Metrics — frozen real-video holdout

Scored on a **frozen 225-frame holdout** of our own PoC recordings (never trained on), serve-style
auto-grid tiling, threshold 50, code-aware. Both generations scored in one run:

| Model | Recall | Precision proxy |
|---|---|---|
| **gen 3 (this)** | **0.847** | **0.771** |
| gen 2 | 0.745 | 0.759 |

Generation-over-generation on the frozen holdout: `stock 0.539 → gen1 0.680 → gen3 0.847`.

> ⚠️ **Not casino accuracy (rule of the project).** The holdout is our own proof-of-concept footage
> with LLM-verified (not fully human-verified) ground truth. These numbers show the detector
> *mechanism* and its improvement across generations — not a validated real-world full-deck accuracy
> claim on a live casino floor.

## Usage

```python
import json, onnxruntime as ort
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
names = json.load(open("model.classes.json"))   # index -> "AS", "10H", ...
# 640x640 letterboxed input. The detector boxes corner PIPS (~2 per card): cluster same-code pips
# within a few box-diagonals into ONE card-level detection before consuming results.
```

## License & provenance

**AGPL-3.0**, a fine-tune of **Ultralytics YOLO11** (`yolo11s.pt`, AGPL-3.0) — these weights inherit
AGPL-3.0 and **are not an original work of ours**. Networked deployment triggers AGPL §13 (offer the
Corresponding Source). `onnxruntime` (MIT) keeps the inference code AGPL-free; the weights stay AGPL.

Built for **Live Game Defender (LGD)** — an on-prem AI integrity monitor for live casino table
games. © 2026 TechTools s.r.o.
