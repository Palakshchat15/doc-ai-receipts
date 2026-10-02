"""OCR every receipt of a split (resumable, one process).

Usage: .venv\\Scripts\\python scripts\\run_ocr.py --split dev|test
Output: data/ocr/<split>.jsonl  {id, boxes, lines, t_ocr}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, append_jsonl, gold_for, ocr_for, path  # noqa: E402
from src.ocr import run_ocr  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = path("ocr") / f"{a.split}.jsonl"
    done = set(ocr_for(a.split))
    rows = [r for r in gold_for(a.split) if r["id"] not in done]
    if a.limit:
        rows = rows[: a.limit]
    print(f"{len(done)} done, {len(rows)} to go", flush=True)
    for i, r in enumerate(rows, 1):
        res = run_ocr(ROOT / r["image"])
        append_jsonl(out, {"id": r["id"], **res})
        print(f"[{i}/{len(rows)}] {r['id']} {res['t_ocr']:.1f}s {len(res['lines'])} lines", flush=True)


if __name__ == "__main__":
    main()
