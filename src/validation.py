"""Business rules, confidence signal and review routing.

Each rule returns 'pass', 'fail' or 'skip' (not enough fields to check). Rules:

items_sum     sum(items.line_total) == subtotal (or subtotal + tax when item prices include tax
              and the subtotal is printed net). If there is no subtotal, the items must sum to
              the total, or to total - tax - service + discount (receipts that print no subtotal).
total_check   subtotal - discount + tax + service == total. Indonesian receipts often print a tax
              that is already included in the prices ("PB1 included"), so the tax-inclusive form
              subtotal - discount + service == total also passes (recorded as a note).
line_math     for every item with quantity, unit price and line total: qty x unit == line total.
              An item with quantity 1 and no unit price is not checkable (skip).
total_present the total was found.
items_present at least one item with a line total was found.

Money comparisons use abs_tol / rel_tol from config.yaml (default 1 unit or 0.5%).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .normalize import money_equal
from .schema import Receipt

PASS, FAIL, SKIP = "pass", "fail", "skip"


@dataclass
class RuleResult:
    name: str
    status: str
    detail: str = ""


@dataclass
class Validation:
    rules: list[RuleResult] = field(default_factory=list)
    confidence: float = 0.0
    schema_valid: bool = True
    decision: str = "review"          # auto_approve | review
    reasons: list[str] = field(default_factory=list)

    def status(self, name: str) -> str:
        for r in self.rules:
            if r.name == name:
                return r.status
        return SKIP

    def to_dict(self) -> dict:
        return {"rules": [r.__dict__ for r in self.rules], "confidence": round(self.confidence, 4),
                "schema_valid": self.schema_valid, "decision": self.decision, "reasons": self.reasons}


def _eq(a, b, abs_tol, rel_tol):
    return money_equal(a, b, abs_tol=abs_tol, rel_tol=rel_tol)


def check_items_sum(r: Receipt, abs_tol=1.0, rel_tol=0.005) -> RuleResult:
    lines = [i.line_total for i in r.items if i.line_total is not None]
    if not lines:
        return RuleResult("items_sum", SKIP, "no item line totals")
    s = sum(lines)
    if r.subtotal is not None:
        if _eq(s, r.subtotal, abs_tol, rel_tol):
            return RuleResult("items_sum", PASS, f"items {s:,.2f} == subtotal")
        # Item prices printed tax-inclusive while the subtotal is net of tax.
        if r.tax and _eq(s, r.subtotal + r.tax, abs_tol, rel_tol):
            return RuleResult("items_sum", PASS, "items include tax (== subtotal + tax)")
        return RuleResult("items_sum", FAIL, f"items {s:,.2f} vs subtotal {r.subtotal:,.2f}")
    if r.total is not None:
        disc, tax, svc = r.discount or 0, r.tax or 0, r.service_charge or 0
        targets = {r.total, r.total - tax - svc + disc, r.total - svc + disc}
        ok = any(_eq(s, t, abs_tol, rel_tol) for t in targets)
        return RuleResult("items_sum", PASS if ok else FAIL, f"items {s:,.2f} vs total {r.total:,.2f} (no subtotal)")
    return RuleResult("items_sum", SKIP, "no subtotal or total")


def check_total(r: Receipt, abs_tol=1.0, rel_tol=0.005) -> RuleResult:
    if r.subtotal is None or r.total is None:
        return RuleResult("total_check", SKIP, "needs subtotal and total")
    disc, tax, svc = r.discount or 0, r.tax or 0, r.service_charge or 0
    full = r.subtotal - disc + tax + svc
    if _eq(full, r.total, abs_tol, rel_tol):
        return RuleResult("total_check", PASS, f"{full:,.2f} == total")
    incl = r.subtotal - disc + svc
    if tax and _eq(incl, r.total, abs_tol, rel_tol):
        return RuleResult("total_check", PASS, "tax included in prices")
    return RuleResult("total_check", FAIL, f"subtotal-disc+tax+svc = {full:,.2f} vs total {r.total:,.2f}")


def check_line_math(r: Receipt, abs_tol=1.0, rel_tol=0.005) -> RuleResult:
    checked = bad = 0
    bad_names = []
    for it in r.items:
        if it.quantity is None or it.unit_price is None or it.line_total is None:
            continue
        checked += 1
        if not _eq(it.quantity * it.unit_price, it.line_total, abs_tol, rel_tol):
            bad += 1
            bad_names.append(it.name)
    if not checked:
        return RuleResult("line_math", SKIP, "no item has qty, unit price and line total")
    if bad:
        return RuleResult("line_math", FAIL, f"{bad}/{checked} items fail: {', '.join(bad_names[:3])}")
    return RuleResult("line_math", PASS, f"{checked} items checked")


def validate(r: Receipt | None, cfg: dict | None = None, schema_valid: bool = True,
             threshold: float | None = None) -> Validation:
    """Run all rules, compute confidence and the routing decision."""
    cfg = cfg or {}
    abs_tol = cfg.get("abs_tol", 1.0)
    rel_tol = cfg.get("rel_tol", 0.005)
    thr = cfg.get("approve_threshold", 1.0) if threshold is None else threshold
    v = Validation(schema_valid=schema_valid and r is not None)
    if r is None or not v.schema_valid:
        v.rules = [RuleResult("schema", FAIL, "output did not parse into the schema")]
        v.confidence = 0.0
        v.reasons = ["schema invalid"]
        return v
    has_total = r.total is not None
    has_items = any(i.line_total is not None for i in r.items)
    v.rules = [
        RuleResult("schema", PASS),
        RuleResult("total_present", PASS if has_total else FAIL),
        RuleResult("items_present", PASS if has_items else FAIL),
        check_items_sum(r, abs_tol, rel_tol),
        check_total(r, abs_tol, rel_tol),
        check_line_math(r, abs_tol, rel_tol),
    ]
    v.confidence = confidence(v)
    fails = [x.name for x in v.rules if x.status == FAIL]
    v.reasons = [f"{n} failed" for n in fails]
    v.decision = "auto_approve" if (v.confidence >= thr and not fails) else "review"
    if v.decision == "review" and not fails:
        v.reasons.append(f"confidence {v.confidence:.2f} below threshold {thr:.2f}")
    return v


# Penalties behind the confidence signal. Fails cost more than skips: a skip means the
# arithmetic could not be cross-checked, a fail means it was checked and is wrong.
PENALTY = {
    "total_present": {FAIL: 0.5},
    "items_present": {FAIL: 0.3},
    "items_sum": {FAIL: 0.4, SKIP: 0.15},
    "total_check": {FAIL: 0.4, SKIP: 0.05},   # skip is free when items_sum passed
    "line_math": {FAIL: 0.2},
}


def confidence(v: Validation) -> float:
    if not v.schema_valid:
        return 0.0
    c = 1.0
    items_ok = v.status("items_sum") == PASS
    for r in v.rules:
        if r.name == "total_check" and r.status == SKIP and items_ok:
            continue  # items already reconcile with the total; nothing left unchecked
        c -= PENALTY.get(r.name, {}).get(r.status, 0.0)
    return max(0.0, min(1.0, round(c, 4)))
