"""OCR with RapidOCR (PP-OCR models in ONNX, CPU) and reading-order line reconstruction.

Line reconstruction: estimate the page skew from the median angle of the text boxes' top
edges, rotate box centres by -skew, then group boxes whose vertical extents overlap by at
least `line_y_overlap` x the smaller box height. Within a line boxes are sorted by x; a wide
horizontal gap is written as a double space so columns (name ... qty ... price) stay visible.
"""
from __future__ import annotations

import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .config import load_config


@lru_cache(maxsize=1)
def get_engine():
    from rapidocr import RapidOCR

    n = int(load_config()["ocr"].get("threads", 4))
    return RapidOCR(params={
        "EngineConfig.onnxruntime.intra_op_num_threads": n,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })


def _load(img) -> tuple[Image.Image, float]:
    """Load (EXIF-rotated) and downscale so the long side <= max_side. Returns (RGB PIL image, scale).

    A PIL image (not a numpy array) is passed to RapidOCR: it treats arrays as BGR.
    """
    from PIL import ImageOps

    max_side = load_config()["ocr"]["max_side"]
    im = img if isinstance(img, Image.Image) else Image.open(img)
    im = (ImageOps.exif_transpose(im) or im).convert("RGB")
    s = 1.0
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im, s


def run_ocr(img) -> dict:
    """Return {'boxes': [{'quad','text','score'}...] in ORIGINAL image coordinates, 'lines', 't_ocr'}."""
    arr, s = _load(img)
    t0 = time.perf_counter()
    res = get_engine()(arr)
    t = time.perf_counter() - t0
    boxes = []
    if res.boxes is not None:
        for q, txt, sc in zip(res.boxes, res.txts, res.scores):
            quad = (np.asarray(q, dtype=float) / s).round(1).tolist()
            boxes.append({"quad": quad, "text": str(txt), "score": round(float(sc), 4)})
    lines = reconstruct_lines(boxes, load_config()["ocr"]["line_y_overlap"])
    return {"boxes": boxes, "lines": lines, "t_ocr": round(t, 3)}


def _skew(boxes: list[dict]) -> float:
    """Median angle (radians) of the top edges of boxes wider than they are tall."""
    angs = []
    for b in boxes:
        (x1, y1), (x2, y2) = b["quad"][0], b["quad"][1]
        w = math.hypot(x2 - x1, y2 - y1)
        h = math.hypot(b["quad"][3][0] - x1, b["quad"][3][1] - y1)
        if w > 1.5 * h and w > 0:
            angs.append(math.atan2(y2 - y1, x2 - x1))
    if not angs:
        return 0.0
    a = float(np.median(angs))
    return a if abs(a) < math.radians(20) else 0.0


def reconstruct_lines(boxes: list[dict], min_overlap: float = 0.5) -> list[str]:
    """Group OCR boxes into reading-order text lines."""
    if not boxes:
        return []
    a = _skew(boxes)
    ca, sa = math.cos(-a), math.sin(-a)
    items = []
    for b in boxes:
        pts = [(x * ca - y * sa, x * sa + y * ca) for x, y in b["quad"]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        items.append({"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys), "text": b["text"]})
    items.sort(key=lambda d: (d["y0"] + d["y1"]) / 2)
    lines: list[list[dict]] = []
    for it in items:
        h = it["y1"] - it["y0"]
        best, best_ov = None, 0.0
        for ln in lines[-4:]:  # only recent lines can overlap (sorted by y)
            ly0 = np.median([d["y0"] for d in ln])
            ly1 = np.median([d["y1"] for d in ln])
            ov = min(it["y1"], ly1) - max(it["y0"], ly0)
            denom = max(1e-6, min(h, ly1 - ly0))
            if ov / denom >= min_overlap and ov / denom > best_ov:
                best, best_ov = ln, ov / denom
        if best is None:
            lines.append([it])
        else:
            best.append(it)
    out = []
    for ln in lines:
        ln.sort(key=lambda d: d["x0"])
        char_w = np.median([(d["x1"] - d["x0"]) / max(1, len(d["text"])) for d in ln])
        parts = [ln[0]["text"]]
        for prev, cur in zip(ln, ln[1:]):
            gap = cur["x0"] - prev["x1"]
            parts.append("   " if gap > 3 * char_w else " ")
            parts.append(cur["text"])
        out.append("".join(parts).strip())
    return [s for s in out if s]


def ocr_text(lines: list[str]) -> str:
    return "\n".join(lines)


def draw_boxes(img_path: str | Path, boxes: list[dict]) -> Image.Image:
    from PIL import ImageDraw

    im = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(im)
    w = max(2, round(max(im.size) / 600))
    for b in boxes:
        pts = [tuple(p) for p in b["quad"]]
        d.polygon(pts, outline=(230, 60, 60), width=w)
    return im
