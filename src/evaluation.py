"""Scoring: field-level accuracy/F1, item-level F1, whole-receipt exact match, routing quality.

Definitions (all after normalisation by src/normalize.py):
* Scalar field (subtotal, tax, service_charge, total): correct if pred == gold within 0.5 unit,
  where "both absent" also counts as correct (accuracy over all receipts).
  F1 per field: TP = gold present and pred equal; FP = pred present and (gold absent or different);
  FN = gold present and (pred absent or different). A wrong value is both an FP and an FN.
* Items: a predicted item matches a gold item if name similarity >= threshold (rapidfuzz ratio on
  normalised names) AND line totals are equal (within 0.5). One-to-one greedy matching by
  descending similarity. Micro P/R/F1 over all items. "name-only" F1 drops the price condition
  (diagnostic). Item-count accuracy: len(pred.items) == len(gold.items).
* Whole-receipt exact match: all four scalar fields correct AND every gold item matched with
  no extra predicted items.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from .normalize import money_equal, norm_name
from .schema import SCORED_FIELDS, Receipt

EQ_TOL = 0.5


def name_sim(a: str, b: str) -> float:
    return fuzz.ratio(norm_name(a), norm_name(b)) / 100.0


def match_items(pred: list, gold: list, threshold: float = 0.6, use_price: bool = True):
    """Greedy one-to-one matching. Returns list of (pred_idx, gold_idx, sim)."""
    cands = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if use_price and not money_equal(p.line_total, g.line_total, abs_tol=EQ_TOL):
                continue
            s = name_sim(p.name, g.name)
            if s >= threshold:
                cands.append((s, i, j))
    cands.sort(reverse=True)
    used_p, used_g, pairs = set(), set(), []
    for s, i, j in cands:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((i, j, s))
    return pairs


def score_receipt(pred: Receipt | None, gold: Receipt, threshold: float = 0.6) -> dict:
    p = pred or Receipt()
    out: dict = {"schema_valid": pred is not None}
    all_ok = True
    for f in SCORED_FIELDS:
        pv, gv = getattr(p, f), getattr(gold, f)
        ok = money_equal(pv, gv, abs_tol=EQ_TOL)
        out[f"{f}_correct"] = ok
        out[f"{f}_tp"] = int(gv is not None and ok)
        out[f"{f}_fp"] = int(pv is not None and not ok)
        out[f"{f}_fn"] = int(gv is not None and not ok)
        all_ok &= ok
    pairs = match_items(p.items, gold.items, threshold, use_price=True)
    pairs_name = match_items(p.items, gold.items, threshold, use_price=False)
    out["n_pred_items"] = len(p.items)
    out["n_gold_items"] = len(gold.items)
    out["item_tp"] = len(pairs)
    out["item_fp"] = len(p.items) - len(pairs)
    out["item_fn"] = len(gold.items) - len(pairs)
    out["item_name_tp"] = len(pairs_name)
    out["item_count_correct"] = len(p.items) == len(gold.items)
    qn = qc = 0
    for i, j, _ in pairs:
        gq = gold.items[j].quantity
        if gq is None:
            continue
        qn += 1
        pq = p.items[i].quantity
        qc += int(pq is not None and abs(pq - gq) < 1e-6)
    out["qty_checked"] = qn
    out["qty_correct"] = qc
    items_ok = out["item_fp"] == 0 and out["item_fn"] == 0
    out["items_exact"] = items_ok
    out["exact_match"] = bool(all_ok and items_ok)
    return out


def _f1(tp, fp, fn):
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f1


def summarise(scores: list[dict]) -> dict:
    """Aggregate per-receipt scores into one metrics row."""
    n = len(scores)
    if not n:
        return {"n": 0}
    m: dict = {"n": n, "schema_valid": sum(s["schema_valid"] for s in scores) / n}
    for f in SCORED_FIELDS:
        m[f"{f}_acc"] = sum(s[f"{f}_correct"] for s in scores) / n
        tp = sum(s[f"{f}_tp"] for s in scores)
        fp = sum(s[f"{f}_fp"] for s in scores)
        fn = sum(s[f"{f}_fn"] for s in scores)
        m[f"{f}_f1"] = _f1(tp, fp, fn)[2]
    tp = sum(s["item_tp"] for s in scores)
    fp = sum(s["item_fp"] for s in scores)
    fn = sum(s["item_fn"] for s in scores)
    m["item_p"], m["item_r"], m["item_f1"] = _f1(tp, fp, fn)
    ntp = sum(s["item_name_tp"] for s in scores)
    npred = sum(s["n_pred_items"] for s in scores)
    ngold = sum(s["n_gold_items"] for s in scores)
    m["item_name_f1"] = _f1(ntp, npred - ntp, ngold - ntp)[2]
    m["item_count_acc"] = sum(s["item_count_correct"] for s in scores) / n
    qn = sum(s["qty_checked"] for s in scores)
    m["qty_acc_matched"] = sum(s["qty_correct"] for s in scores) / qn if qn else float("nan")
    m["exact_match"] = sum(s["exact_match"] for s in scores) / n
    return m


def routing_stats(scores: list[dict], confidences: list[float], decisions: list[str]) -> dict:
    """Does routing catch errors? Exact-match / total accuracy inside vs outside the approved set."""
    appr = [s for s, d in zip(scores, decisions) if d == "auto_approve"]
    rev = [s for s, d in zip(scores, decisions) if d != "auto_approve"]
    n = len(scores)

    def rate(xs, k):
        return sum(x[k] for x in xs) / len(xs) if xs else float("nan")

    return {
        "auto_approve_rate": len(appr) / n if n else float("nan"),
        "approved_exact": rate(appr, "exact_match"),
        "review_exact": rate(rev, "exact_match"),
        "approved_total_acc": rate(appr, "total_correct"),
        "review_total_acc": rate(rev, "total_correct"),
        "errors_caught": (sum(not x["exact_match"] for x in rev) / max(1, sum(not x["exact_match"] for x in scores))),
    }


def routing_curve(scores: list[dict], confidences: list[float]) -> list[dict]:
    """Approve everything with confidence >= t, for each distinct t."""
    rows = []
    for t in sorted(set(confidences) | {0.0}, reverse=True):
        appr = [s for s, c in zip(scores, confidences) if c >= t]
        rows.append({
            "threshold": t,
            "approve_rate": len(appr) / len(scores),
            "approved_exact": (sum(s["exact_match"] for s in appr) / len(appr)) if appr else float("nan"),
            "approved_total_acc": (sum(s["total_correct"] for s in appr) / len(appr)) if appr else float("nan"),
        })
    return rows
