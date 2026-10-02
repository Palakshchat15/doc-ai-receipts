"""System (a): OCR + keyword/regex heuristics. No learning, no LLM.

* Scalars: a line whose label matches a keyword pattern; the value is the last amount on that
  line, or the first amount on the next line if the label line has none.
  - total:    TOTAL / GRAND TOTAL / JUMLAH, excluding SUBTOTAL, TOTAL ITEM/QTY, TOTAL DISC, TOTAL TAX.
              When several lines qualify, GRAND TOTAL wins, else the first plain TOTAL line
              (later ones tend to be payment summaries).
  - subtotal: SUBTOTAL / SUB TOTAL / SUB-TOTAL / SUBTTL / NET AMT
  - tax:      TAX / PB1 / PPN / VAT / PAJAK      service: SERVICE / SVC / S.C / SC
  - discount: DISC / DISCOUNT / POTONGAN (not on item lines before the subtotal)
* Items (naive): lines above the first subtotal/total/tax line that contain letters and end
  with an amount; name = the text before the amounts, line_total = the last amount, quantity =
  a leading integer (optionally followed by x).
"""
from __future__ import annotations

import re

from .normalize import parse_amount
from .schema import Item, Receipt

# thousands groups may be separated by '.', ',', ', ' or a single space ("20 000", "121, 768")
# (space groups only when no other separator follows, so "1 120,000" stays qty 1 + 120,000)
AMOUNT = re.compile(r"-?\(?\d{1,3}(?:[.,]\s?\d{3}(?!\d))+(?:[.,]\d{1,2})?\)?"
                    r"|-?\d{1,3}(?:\s\d{3})+(?![\d.,])"
                    r"|-?\d+(?:[.,]\d{1,2})?")

RX = {
    "subtotal": re.compile(r"(?i)\b(sub\s*-?\s*t\w{0,3}l\b|net\s*amt|total\s*item\s*price)"),
    "tax": re.compile(r"(?i)\b(tax|pb\s*1|ppn|vat|pajak)\b"),
    "service_charge": re.compile(r"(?i)\b(service|svc|s\.?c\.?\b|serv\.?\s*charge)"),
    "discount": re.compile(r"(?i)\b(disc|discount|potongan)"),
    "grand": re.compile(r"(?i)\bgrand\s*total"),
    # also matches OCR-damaged '.OTAL' / '3TOTAL'
    "total": re.compile(r"(?i)((?<![a-z])t?otal\b|\bjumlah\b|\bamount\s*due\b)"),
    "not_total": re.compile(r"(?i)(sub\s*-?\s*total|total\s*(item|qty|quantity|disc|tax)|items?\s*:|qty)"),
    "stop": re.compile(r"(?i)\b(sub\s*-?\s*total|total|tax|pb\s*1|ppn|cash|tunai|change|kembali|net\s*amt)"),
}


def amounts(line: str) -> list[float]:
    out = []
    for m in AMOUNT.finditer(line):
        tok = m.group(0)
        digits = re.sub(r"\D", "", tok)
        if not digits:
            continue
        v = parse_amount(tok)
        if v is not None:
            out.append(abs(v))
    return out


def _value(lines: list[str], i: int) -> float | None:
    a = amounts(lines[i])
    if a:
        return a[-1]
    if i + 1 < len(lines):
        a = amounts(lines[i + 1])
        if a:
            return a[0]
    return None


def extract_regex(lines: list[str]) -> Receipt:
    vals: dict = {}
    total_cands, grand = [], None
    for i, ln in enumerate(lines):
        if RX["subtotal"].search(ln) and "subtotal" not in vals:
            vals["subtotal"] = _value(lines, i)
            continue
        if RX["grand"].search(ln) and grand is None:
            grand = _value(lines, i)
            continue
        if RX["total"].search(ln) and not RX["not_total"].search(ln):
            v = _value(lines, i)
            if v is not None:
                total_cands.append(v)
            continue
        for k in ("tax", "service_charge", "discount"):
            if k not in vals and RX[k].search(ln):
                v = _value(lines, i)
                if v is not None:
                    vals[k] = v
                break
    total = grand if grand is not None else (total_cands[0] if total_cands else None)
    # naive items
    items = []
    for ln in lines:
        if RX["stop"].search(ln):
            break
        a = amounts(ln)
        if not a or not re.search(r"[A-Za-z]{2,}", ln):
            continue
        m = re.match(r"^\s*(\d{1,2})\s*[xX]?\s+(.*)$", ln)
        qty, rest = (float(m.group(1)), m.group(2)) if m else (None, ln)
        name = AMOUNT.split(rest)[0].strip(" @:-x")
        if not re.search(r"[A-Za-z]{2,}", name):
            continue
        items.append(Item(name=name, quantity=qty, line_total=a[-1]))
    return Receipt(items=items, total=total, **{k: v for k, v in vals.items() if v is not None})
