"""Score every prediction file of a split against gold.

Usage: .venv\\Scripts\\python scripts\\score.py --split dev|test
Writes:
  results/metrics_<split>.csv        one row per system: quality, routing, latency, tokens
  results/per_receipt_<split>.csv    one row per (system, receipt)
  results/routing_curve_<split>.csv  approve-rate vs accuracy of the approved set, per threshold
  results/metrics_<split>_subset.csv if any system covers only a subset: every system on that subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import gold_for, load_config, ocr_for, path, read_jsonl  # noqa: E402
from src.evaluation import routing_curve, routing_stats, score_receipt, summarise  # noqa: E402
from src.schema import Receipt  # noqa: E402
from src.validation import validate  # noqa: E402

LABELS = {"regex": "(a) OCR + regex", "ocr_llm__qwen2.5-7b": "(b) OCR + qwen2.5:7b",
          "ocr_llm__qwen2.5-3b": "(c) OCR + qwen2.5:3b", "vision": "(d) vision LLM",
          "gold_ocr_llm": "gold OCR + LLM (upper bound)"}


def label(stem: str) -> str:
    parts = stem.split("__")
    sysname = parts[1]
    if sysname == "regex":
        return LABELS["regex"]
    model, prompt = parts[2], parts[3]
    key = f"{sysname}__{model}"
    base = LABELS.get(key) or (f"(d) vision {model.replace('-', ':', 1)}" if sysname == "vision"
                              else f"{LABELS.get(sysname, sysname)} {model.replace('-', ':', 1)}")
    return f"{base} [{prompt}]"


def pct(xs, q):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.percentile(xs, q)) if xs else float("nan")


def score_file(f: Path, gold: dict, thr: float, ids: set | None = None):
    preds = [r for r in read_jsonl(f) if not r.get("exception")]
    if ids is not None:
        preds = [p for p in preds if p["id"] in ids]
    per, confs, decs = [], [], []
    cfg = load_config()["validation"]
    for p in preds:
        pr = Receipt(**p["pred"]) if p.get("pred") else None
        s = score_receipt(pr, Receipt(**gold[p["id"]]),
                          load_config()["eval"]["item_name_threshold"])
        v = validate(pr, cfg, schema_valid=p.get("error") is None, threshold=thr)
        s.update(id=p["id"], confidence=v.confidence, decision=v.decision,
                 t_ocr=p.get("t_ocr"), t_llm=p.get("t_llm"),
                 prompt_tokens=p.get("prompt_tokens"), completion_tokens=p.get("completion_tokens"),
                 error=p.get("error"))
        per.append(s)
        confs.append(v.confidence)
        decs.append(v.decision)
    return preds, per, confs, decs


def metrics_row(name, preds, per, confs, decs, ocr_times):
    m = {"system": name, **summarise(per), **routing_stats(per, confs, decs)}
    t_ocr = [p.get("t_ocr") or 0.0 for p in preds]
    t_llm = [p.get("t_llm") or 0.0 for p in preds]
    tot = [a + b for a, b in zip(t_ocr, t_llm)]
    m.update(ocr_p50=pct(t_ocr, 50), ocr_p95=pct(t_ocr, 95), llm_p50=pct(t_llm, 50), llm_p95=pct(t_llm, 95),
             total_p50=pct(tot, 50), total_p95=pct(tot, 95),
             throughput_per_min=(60 / np.mean(tot)) if tot and np.mean(tot) > 0 else float("nan"),
             prompt_tok_mean=np.mean([p.get("prompt_tokens") or 0 for p in preds]),
             completion_tok_mean=np.mean([p.get("completion_tokens") or 0 for p in preds]),
             truncated=sum(1 for p in preds if p.get("done_reason") == "length"))
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()
    thr = load_config()["validation"]["approve_threshold"] if a.threshold is None else a.threshold
    gold = {r["id"]: r["gold"] for r in gold_for(a.split)}
    ocr_times = {k: v["t_ocr"] for k, v in ocr_for(a.split).items()}
    files = sorted(path("preds").glob(f"{a.split}__*.jsonl"))
    rows, per_rows, curve_rows = [], [], []
    coverage = {}
    for f in files:
        name = label(f.stem)
        preds, per, confs, decs = score_file(f, gold, thr)
        if not preds:
            continue
        coverage[f] = {p["id"] for p in preds}
        rows.append(metrics_row(name, preds, per, confs, decs, ocr_times))
        for s in per:
            per_rows.append({"system": name, **s})
        for c in routing_curve(per, confs):
            curve_rows.append({"system": name, **c})
    # "perfect extractor": gold itself through the rules (routing ceiling)
    gper = [score_receipt(Receipt(**g), Receipt(**g)) for g in gold.values()]
    gv = [validate(Receipt(**g), load_config()["validation"], threshold=thr) for g in gold.values()]
    rows.append({"system": "gold labels through the rules (ceiling)", "n": len(gold),
                 **routing_stats(gper, [v.confidence for v in gv], [v.decision for v in gv])})
    out = path("results")
    df = pd.DataFrame(rows)
    df.to_csv(out / f"metrics_{a.split}.csv", index=False)
    pd.DataFrame(per_rows).to_csv(out / f"per_receipt_{a.split}.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out / f"routing_curve_{a.split}.csv", index=False)
    cols = ["system", "n", "total_acc", "subtotal_acc", "tax_acc", "service_charge_acc", "item_f1",
            "item_count_acc", "exact_match", "auto_approve_rate", "approved_exact", "llm_p50", "llm_p95"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df[[c for c in cols if c in df]].round(3).to_string(index=False))
    # subset comparison
    full = len(gold)
    sub = [ids for ids in coverage.values() if len(ids) < full]
    if sub:
        ids = set.intersection(*sub)
        srows = []
        for f in files:
            if f not in coverage or not ids <= coverage[f]:
                continue
            preds, per, confs, decs = score_file(f, gold, thr, ids)
            srows.append(metrics_row(label(f.stem), preds, per, confs, decs, ocr_times))
        sdf = pd.DataFrame(srows)
        sdf.to_csv(out / f"metrics_{a.split}_subset.csv", index=False)
        print(f"\nSubset of {len(ids)} receipts covered by every partial run:")
        print(sdf[[c for c in cols if c in sdf]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
