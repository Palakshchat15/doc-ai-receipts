"""Bucket the errors of one system into OCR / extraction / normalisation errors, with examples.

Usage: .venv\\Scripts\\python scripts\\error_analysis.py --split test [--pred results\\preds\\<file>.jsonl]
Default pred file: the main model (config llm.model) OCR+LLM run.
Output: results/error_analysis_<split>.json (buckets with counts and examples), results/errors_<split>.csv

Attribution for a wrong scalar field (subtotal / tax / service_charge / total):
  normalisation  the raw LLM string has exactly the gold digits but our parser turned it into another number
  ocr            the gold amount's digits appear nowhere in the OCR text (the LLM never saw it)
  missed         gold amount is in the OCR text, the model returned null
  wrong_value    gold amount is in the OCR text, the model returned a different number
  spurious       gold has no value (not labelled/printed), the model returned one
For items: gold items not matched are 'item_ocr' (price or name not in the OCR text) or
'item_missed' (both visible); an 'item_price_only' miss means the name matched a predicted item
but the line total differed; predicted items with no gold partner are 'item_extra'.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402

from src.config import gold_for, load_config, ocr_for, path, read_jsonl  # noqa: E402
from src.evaluation import EQ_TOL, match_items  # noqa: E402
from src.normalize import money_equal, norm_name, parse_amount  # noqa: E402
from src.schema import SCORED_FIELDS, Receipt  # noqa: E402

CORD_KEY = {"subtotal": ("sub_total", "subtotal_price"), "tax": ("sub_total", "tax_price"),
            "service_charge": ("sub_total", "service_price"), "total": ("total", "total_price")}

DESC = {
    "ocr": "Scalar wrong because the gold amount is not in the OCR text (misread digits, lost line).",
    "missed": "Scalar present in the OCR text but the model returned null.",
    "wrong_value": "Scalar present in the OCR text but the model picked another number (e.g. cash instead of total).",
    "spurious": "Model returned a value for a field the gold leaves empty.",
    "normalisation": "Model copied the right string but number parsing changed it.",
    "item_ocr": "Gold item missed; its price or name is not readable in the OCR text.",
    "item_missed": "Gold item missed although its name and price are in the OCR text.",
    "item_price_only": "Item found by name but with a different line total.",
    "item_extra": "Predicted item with no gold partner (modifier listed as item, split/merged lines, non-item line).",
}


def digits(s) -> str:
    return re.sub(r"\D", "", str(s or ""))


def gold_raw(cord: dict, field: str):
    a, b = CORD_KEY[field]
    blk = cord.get(a) or {}
    if isinstance(blk, list):
        blk = blk[0] if blk else {}
    v = blk.get(b)
    return v[0] if isinstance(v, list) and v else v


def ocr_numbers(lines: list[str]) -> set[str]:
    """Every digit string obtainable from an OCR number token (separators dropped, decimals .00 dropped)."""
    out = set()
    for ln in lines:
        for tok in re.findall(r"\d[\d.,\s]*\d|\d", ln):
            d = digits(tok)
            out.add(d)
            out.add(re.sub(r"0{2}$", "", d) if re.search(r"[.,]00$", tok) else d)
            for part in re.split(r"\s+", tok.strip()):
                out.add(digits(part))
    return out


def amount_in_ocr(value: float, raw, nums: set[str]) -> bool:
    cands = {digits(raw)} if raw is not None else set()
    if value is not None:
        cands.add(str(int(round(value))))
    cands.discard("")
    return any(c in nums for c in cands)


def name_in_ocr(name: str, lines: list[str]) -> bool:
    n = norm_name(name)
    if not n:
        return True
    return any(fuzz.partial_ratio(n, norm_name(ln)) >= 80 for ln in lines)


def analyse(split: str, pred_file: Path) -> dict:
    gold = {r["id"]: r for r in gold_for(split)}
    ocr = ocr_for(split)
    thr = load_config()["eval"]["item_name_threshold"]
    buckets = defaultdict(list)
    rows = []
    for p in read_jsonl(pred_file):
        if p.get("exception"):
            continue
        g = gold[p["id"]]
        gr = Receipt(**g["gold"])
        pr = Receipt(**p["pred"]) if p.get("pred") else Receipt()
        raw = json.loads(p["raw"]) if p.get("raw") else {}
        lines = ocr[p["id"]]["lines"] if p["id"] in ocr else []
        nums = ocr_numbers(lines)
        for f in SCORED_FIELDS:
            pv, gv = getattr(pr, f), getattr(gr, f)
            if money_equal(pv, gv, abs_tol=EQ_TOL):
                continue
            graw = gold_raw(g["cord"], f)
            praw = raw.get(f) if isinstance(raw, dict) else None
            if gv is None:
                b = "spurious"
            elif praw is not None and digits(praw) and digits(praw) == digits(graw) and \
                    not money_equal(parse_amount(praw), gv, abs_tol=EQ_TOL):
                b = "normalisation"
            elif not amount_in_ocr(gv, graw, nums):
                b = "ocr"
            elif pv is None:
                b = "missed"
            else:
                b = "wrong_value"
            ex = {"id": p["id"], "field": f, "gold": graw, "pred_raw": praw, "pred": pv}
            buckets[b].append(ex)
            rows.append({"bucket": b, **ex})
        pairs = match_items(pr.items, gr.items, thr, use_price=True)
        mp = {i for i, _, _ in pairs}
        mg = {j for _, j, _ in pairs}
        name_pairs = {j: i for i, j, _ in match_items(pr.items, gr.items, thr, use_price=False)}
        for j, gi in enumerate(gr.items):
            if j in mg:
                continue
            if j in name_pairs and name_pairs[j] not in mp:
                b = "item_price_only"
                pi = pr.items[name_pairs[j]]
                ex = {"id": p["id"], "field": "item", "gold": f"{gi.name} = {gi.line_total}",
                      "pred": f"{pi.name} = {pi.line_total}"}
                if not amount_in_ocr(gi.line_total, None, nums):
                    b = "item_ocr"
            elif not amount_in_ocr(gi.line_total, None, nums) or not name_in_ocr(gi.name, lines):
                b = "item_ocr"
                ex = {"id": p["id"], "field": "item", "gold": f"{gi.name} = {gi.line_total}", "pred": None}
            else:
                b = "item_missed"
                ex = {"id": p["id"], "field": "item", "gold": f"{gi.name} = {gi.line_total}", "pred": None}
            buckets[b].append(ex)
            rows.append({"bucket": b, **ex})
        used_by_price_only = {name_pairs[j] for j in range(len(gr.items)) if j in name_pairs and j not in mg}
        for i, pi in enumerate(pr.items):
            if i in mp or i in used_by_price_only:
                continue
            ex = {"id": p["id"], "field": "item", "gold": None, "pred": f"{pi.name} = {pi.line_total}"}
            buckets["item_extra"].append(ex)
            rows.append({"bucket": "item_extra", **ex})
    order = ["ocr", "missed", "wrong_value", "spurious", "normalisation", "item_ocr", "item_missed",
             "item_price_only", "item_extra"]
    out = {
        "pred_file": pred_file.name,
        "method": ("Each wrong field / unmatched item is attributed by checking whether the gold amount (digits) "
                   "appears in the OCR text and what the model returned; see scripts/error_analysis.py."),
        "buckets": [{"bucket": b, "count": len(buckets[b]), "description": DESC[b], "examples": buckets[b][:6]}
                    for b in order],
    }
    return out, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--pred", default=None)
    a = ap.parse_args()
    if a.pred:
        pf = Path(a.pred)
    else:
        m = load_config()["llm"]["model"].replace(":", "-")
        cands = sorted(path("preds").glob(f"{a.split}__ocr_llm__{m}__*.jsonl"))
        if not cands:
            sys.exit("no prediction file for the main model yet")
        pf = cands[-1]
    out, df = analyse(a.split, pf)
    (path("results") / f"error_analysis_{a.split}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                                                    encoding="utf-8")
    df.to_csv(path("results") / f"errors_{a.split}.csv", index=False)
    for b in out["buckets"]:
        print(f"{b['bucket']:16s} {b['count']:4d}  {b['description']}")


if __name__ == "__main__":
    main()
