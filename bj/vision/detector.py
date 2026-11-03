"""Playing-card detector: a YOLO11 ONNX model that boxes each card's corner index.

The model (sroot/lgd-cards-gen3 on Hugging Face) finds corner "pips" (rank + suit),
usually two per card. We merge same-code pips that are close together into one card.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Sighting:
    code: str      # e.g. "10H", "KS", "AD"
    x: float       # card centre, in frame pixels
    y: float
    conf: float
    boxes: list    # the pip boxes merged into this card, (x1, y1, x2, y2)


class CardDetector:
    def __init__(self, model_path: Path, classes_path: Path, conf: float = 0.45, size: int = 960):
        self.names: List[str] = json.loads(Path(classes_path).read_text())
        # On Apple Silicon the plain CPU provider beat CoreML for this model (~60 ms vs ~75 ms).
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # The model was trained at 640 on tiles of a table, so a full frame needs a bigger
        # input to keep card corners large enough. At 640, small or tilted corners got
        # misread (5H as 3H, KS as AS); 960 fixed that at ~120 ms/frame on an M4.
        self.size = inp.shape[2] if isinstance(inp.shape[2], int) else size
        self.conf = conf

    def detect(self, frame: np.ndarray) -> List[Sighting]:
        """Cards in the whole frame, read at `self.size` on the long side."""
        return _merge_pips(self._pips(frame, self.size))

    def detect_table(self, frame: np.ndarray, divider: float, band: float = 0.3,
                     accurate: bool = False) -> List[Sighting]:
        """Read the dealer's side and your side of the table separately.

        Each side is a strip of the frame, read turned 180 degrees: this model reads
        upside-down card corners poorly, and from the player's seat that's often the
        only corner showing. (It was likely trained on footage from the dealer's side.)

        accurate=True is for analyzing videos shot from the player's seat, where the
        dealer's cards are far away and small. On the BJ1 test video it read 53/54 of
        the player's cards (vs 30/54 shrinking the frame to 960 px) and 25/28 of the
        dealer's, at ~0.7 s per frame on an M4. The default is fast enough for the
        live camera, where an overhead iPhone sees every card up close.
        """
        h, w = frame.shape[:2]
        line = int(divider * h)
        d0, d1 = max(0, int((divider - band) * h)), min(h, line + int(0.05 * h))
        p0 = max(0, line - int(0.07 * h))
        if accurate:
            dealer = self._strip_pips(frame, d0, d1, scale=1.5, cap=3840)
            player = self._strip_pips(frame, p0, h, scale=1.0, cap=3840)
        else:
            dealer = self._strip_pips(frame, d0, d1, scale=1.0, cap=1920)
            player = self._strip_pips(frame, p0, h, longest=self.size)
        dealer = [p for p in dealer if p[2][1] + p[2][3] / 2 < line]
        player = [p for p in player if p[2][1] + p[2][3] / 2 >= line]
        return _merge_pips(dealer + player)

    def _strip_pips(self, frame: np.ndarray, y0: int, y1: int, scale: float = 1.0,
                    cap: int = 3840, longest: int = 0):
        """Pips in rows y0..y1 of the frame, read upside down, in frame coordinates."""
        strip = cv2.rotate(frame[y0:y1], cv2.ROTATE_180)
        sh, sw = strip.shape[:2]
        size = longest or min(cap, int(max(sh, sw) * scale))
        return [(c, conf, (sw - x - bw, sh - y - bh + y0, bw, bh))
                for c, conf, (x, y, bw, bh) in self._pips(strip, size)]

    def _pips(self, img: np.ndarray, size: int):
        """Run the model once. Input is letterboxed to a multiple of 32 on each side
        (the model takes any shape), so wide strips don't waste time on padding."""
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        ch, cw = -(-nh // 32) * 32, -(-nw // 32) * 32
        top, left = (ch - nh) // 2, (cw - nw) // 2
        canvas = np.full((ch, cw, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255

        out = self.session.run(None, {self.input_name: blob})[0][0].T   # (anchors, 4 + classes)
        scores = out[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= self.conf
        if not keep.any():
            return []
        xywh, cls, conf = out[keep, :4], cls[keep], conf[keep]

        # Undo the letterbox.
        xywh[:, 0] = (xywh[:, 0] - left) / scale
        xywh[:, 1] = (xywh[:, 1] - top) / scale
        xywh[:, 2:] /= scale
        boxes = [[float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh)] for cx, cy, bw, bh in xywh]
        idx = cv2.dnn.NMSBoxesBatched(boxes, conf.tolist(), cls.tolist(), self.conf, 0.5)
        return [(self.names[int(cls[i])], float(conf[i]), boxes[i]) for i in np.array(idx).flatten()]


def _merge_pips(pips) -> List[Sighting]:
    """Group same-code pip boxes that sit within a few box-diagonals into one card."""
    cards: List[Sighting] = []
    for code, conf, (x, y, bw, bh) in sorted(pips, key=lambda p: -p[1]):
        cx, cy, diag = x + bw / 2, y + bh / 2, (bw ** 2 + bh ** 2) ** 0.5
        box = (x, y, x + bw, y + bh)
        for card in cards:
            if card.code == code and ((card.x - cx) ** 2 + (card.y - cy) ** 2) ** 0.5 < 5 * diag:
                card.boxes.append(box)
                n = len(card.boxes)
                card.x += (cx - card.x) / n
                card.y += (cy - card.y) / n
                card.conf = max(card.conf, conf)
                break
        else:
            cards.append(Sighting(code, cx, cy, conf, [box]))
    return cards
