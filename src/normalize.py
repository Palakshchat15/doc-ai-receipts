"""Number and text normalisation for receipt values.

CORD receipts are mostly Indonesian: '60.000' and '60,000' both mean sixty thousand rupiah,
while '70000.00' has a real decimal part. The rule used here:

* strip currency words/symbols (Rp, IDR, $, ...), brackets, spaces, trailing 'x' etc.;
* a separator ('.' or ',') followed by exactly 3 digits (and nothing else, or another group)
  is a thousands separator;
* the LAST separator followed by 1 or 2 digits at the end of the string is a decimal point;
* a leading '-' or surrounding '( )' makes the value negative only where the caller asks
  (discounts are stored as positive magnitudes, so callers normally take abs()).
"""
from __future__ import annotations

import re
import unicodedata

_CURRENCY_WORDS = re.compile(r"(?i)\b(rp|idr|usd|sgd|myr|eur|krw|won|php|thb|vnd)\b\.?")
_CURRENCY_SYMS = "$€£¥₩₫฿₱"
CURRENCY_MAP = {"rp": "IDR", "idr": "IDR", "$": "USD", "usd": "USD", "sgd": "SGD", "myr": "MYR",
                "rm": "MYR", "€": "EUR", "eur": "EUR", "£": "GBP", "¥": "JPY", "₩": "KRW",
                "krw": "KRW", "won": "KRW", "php": "PHP", "₱": "PHP", "thb": "THB", "฿": "THB",
                "vnd": "VND", "₫": "VND"}


def parse_amount(value) -> float | None:
    """Parse a money/quantity string into a float. Returns None if no number is present.

    >>> parse_amount('60.000'), parse_amount('24,000'), parse_amount('70000.00')
    (60000.0, 24000.0, 70000.0)
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        # CORD sometimes repeats a key; the first occurrence is the printed field.
        for v in value:
            p = parse_amount(v)
            if p is not None:
                return p
        return None
    s = unicodedata.normalize("NFKC", str(value)).strip()
    if not s:
        return None
    neg = False
    s = _CURRENCY_WORDS.sub(" ", s)
    for ch in _CURRENCY_SYMS:
        s = s.replace(ch, " ")
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        neg = True
    # keep digits, separators and a leading minus
    m = re.search(r"-?\s*\d[\d.,\s]*", s)
    if not m:
        return None
    tok = m.group(0)
    if tok.lstrip().startswith("-"):
        neg = True
    tok = re.sub(r"[\s-]", "", tok).strip(".,")
    if not tok:
        return None
    # Decide the decimal separator.
    dec_part = ""
    m2 = re.search(r"[.,](\d{1,2})$", tok)
    if m2:
        dec_part = m2.group(1)
        tok = tok[: m2.start()]
    int_part = re.sub(r"[.,]", "", tok)
    if not int_part and not dec_part:
        return None
    num = float(f"{int_part or 0}.{dec_part or 0}")
    return -num if neg else num


def parse_quantity(value) -> float | None:
    """Quantities like '2', '1X', '2.00', '(2', 'x3', '1.00xITEMs'."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, list):
        return parse_quantity(value[0]) if value else None
    s = str(value)
    m = re.search(r"\d+(?:[.,]\d+)?", s)
    if not m:
        return None
    t = m.group(0).replace(",", ".")
    # '1.000' as a quantity is 1 (three decimals printed), not one thousand
    try:
        return float(t)
    except ValueError:
        return None


def detect_currency(text: str | None) -> str | None:
    """Return an ISO code if a currency word/symbol appears in the text."""
    if not text:
        return None
    t = text.lower()
    for sym in ("€", "£", "¥", "₩", "₫", "฿", "₱", "$"):
        if sym in t:
            return CURRENCY_MAP[sym]
    m = re.search(r"\b(rp|idr|usd|sgd|myr|rm|eur|krw|php|thb|vnd)\b", t)
    if m:
        return CURRENCY_MAP[m.group(1)]
    return None


def norm_name(name: str | None) -> str:
    """Normalise an item name for matching: upper case, alphanumerics and single spaces."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", str(name)).upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return " ".join(s.split())


def money_equal(a: float | None, b: float | None, abs_tol: float = 1.0, rel_tol: float = 0.0) -> bool:
    """Equality for money values. None == None counts as equal (field correctly absent)."""
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))
