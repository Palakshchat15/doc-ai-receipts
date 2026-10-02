"""Run one extraction system over a split; per-receipt results cached in JSONL (resumable).

Usage:
  .venv\\Scripts\\python scripts\\eval_extract.py --split dev|test --system regex
  .venv\\Scripts\\python scripts\\eval_extract.py --split test --system ocr_llm --model qwen2.5:7b
  .venv\\Scripts\\python scripts\\eval_extract.py --split test --system vision --model qwen2.5vl:3b --limit 30
  .venv\\Scripts\\python scripts\\eval_extract.py --split dev --system gold_ocr_llm --model qwen2.5:7b   (OCR upper bound)
  add --smoke to run at most 3 receipts WITHOUT taking the shared LLM lock (allowed while another
  project holds it; seconds each).

Systems:
  regex         (a) OCR lines -> keyword/regex heuristics
  ocr_llm       (b)/(c) OCR lines -> LLM (structured outputs)
  vision        (d) image -> vision LLM, no OCR
  gold_ocr_llm  CORD's own word annotations as "perfect OCR" -> LLM (upper bound for the OCR stage)

Output: results/preds/<split>__<system>[__<model>__<prompt>].jsonl
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import extract  # noqa: E402
from src.baseline import extract_regex  # noqa: E402
from src.config import ROOT, append_jsonl, gold_for, load_config, ocr_for, path, read_jsonl  # noqa: E402
from src.llm import check_ollama  # noqa: E402
from src.lock import held_by_other, llm_lock, lock_owner  # noqa: E402
from src.tracing import Tracer  # noqa: E402
from src.validation import validate  # noqa: E402


def out_path(split: str, system: str, model: str | None, prompt: str) -> Path:
    if system == "regex":
        return path("preds") / f"{split}__regex.jsonl"
    safe = (model or "").replace(":", "-").replace("/", "_")
    return path("preds") / f"{split}__{system}__{safe}__{prompt}.jsonl"


def run(a) -> None:
    cfg = load_config()
    rows = gold_for(a.split)
    if a.ids:
        want = set(a.ids.split(","))
        rows = [r for r in rows if r["id"] in want]
    ocr = ocr_for(a.split) if a.system in ("regex", "ocr_llm") else {}
    if ocr:
        missing = [r["id"] for r in rows if r["id"] not in ocr]
        if missing:
            print(f"WARNING: {len(missing)} receipts have no OCR yet (run scripts/run_ocr.py); skipped", flush=True)
            rows = [r for r in rows if r["id"] in ocr]
    out = out_path(a.split, a.system, a.model, extract.PROMPT_VERSION)
    done = {r["id"] for r in read_jsonl(out) if not r.get("exception")}
    todo = [r for r in rows if r["id"] not in done]
    if a.limit:
        todo = todo[: max(0, a.limit - len(done & {r['id'] for r in rows}))]
    if a.smoke:
        todo = todo[:3]
    print(f"{out.name}: {len(done)} cached, {len(todo)} to run", flush=True)
    tracer = Tracer(path("traces_db"))
    for k, r in enumerate(todo, 1):
        rid = r["id"]
        rec = {"id": rid, "system": a.system, "model": a.model, "prompt_version": extract.PROMPT_VERSION}
        try:
            t0 = time.perf_counter()
            if a.system == "regex":
                o = ocr[rid]
                pred = extract_regex(o["lines"])
                rec.update(pred=pred.model_dump(), error=None, t_ocr=o["t_ocr"], t_llm=0.0)
                receipt, err = pred, None
            elif a.system in ("ocr_llm", "gold_ocr_llm"):
                lines = ocr[rid]["lines"] if a.system == "ocr_llm" else r["gold_lines"]
                ex = extract.extract_from_lines(lines, a.model)
                receipt, err = ex["receipt"], ex["error"]
                rec.update(raw=ex["llm"]["content"], pred=receipt.model_dump() if receipt else None, error=err,
                           t_ocr=ocr[rid]["t_ocr"] if a.system == "ocr_llm" else 0.0, t_llm=ex["llm"]["wall_s"],
                           **{k2: ex["llm"][k2] for k2 in ("prompt_tokens", "completion_tokens", "prompt_eval_s",
                                                            "eval_s", "load_s", "done_reason")})
            else:
                ex = extract.extract_from_image(ROOT / r["image"], a.model)
                receipt, err = ex["receipt"], ex["error"]
                rec.update(raw=ex["llm"]["content"], pred=receipt.model_dump() if receipt else None, error=err,
                           t_ocr=0.0, t_llm=ex["llm"]["wall_s"],
                           **{k2: ex["llm"][k2] for k2 in ("prompt_tokens", "completion_tokens", "prompt_eval_s",
                                                            "eval_s", "load_s", "done_reason")})
            v = validate(receipt, cfg["validation"], schema_valid=err is None)
            rec["validation"] = v.to_dict()
            rec["t_wall"] = time.perf_counter() - t0
            append_jsonl(out, rec)
            if a.system != "regex":
                tracer.log(source="eval", receipt=rid, mode=a.system, model=a.model,
                           prompt_version=extract.PROMPT_VERSION, t_ocr=rec["t_ocr"], t_llm=rec["t_llm"],
                           t_total=rec["t_ocr"] + rec["t_llm"], prompt_tokens=rec.get("prompt_tokens"),
                           completion_tokens=rec.get("completion_tokens"), output=rec["pred"],
                           confidence=v.confidence, decision=v.decision, rules=v.to_dict()["rules"], error=err)
            print(f"[{k}/{len(todo)}] {rid} llm {rec['t_llm']:.1f}s conf {v.confidence:.2f} {v.decision}"
                  f"{' ERR ' + err if err else ''}", flush=True)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            rec.update(exception=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-800:])
            append_jsonl(out, rec)
            print(f"[{k}/{len(todo)}] {rid} EXCEPTION {e}", flush=True)
            if "not reachable" in str(e):
                raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--system", required=True, choices=["regex", "ocr_llm", "vision", "gold_ocr_llm"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--ids", default="")
    ap.add_argument("--limit", type=int, default=0, help="total receipts (cached + new) for this run file")
    ap.add_argument("--smoke", action="store_true", help="<=3 receipts, no lock (allowed while RAG holds it)")
    a = ap.parse_args()
    if a.system == "regex":
        run(a)
        return
    if not a.model:
        a.model = load_config()["llm"]["vision_model" if a.system == "vision" else "model"]
    check_ollama(a.model)
    if a.smoke:
        print(f"smoke run (no lock); lock status: {lock_owner()}", flush=True)
        run(a)
        return
    if held_by_other():
        sys.exit(f"LLM lock held by another project: {lock_owner()}. Not running.")
    with llm_lock(f"eval {a.split} {a.system} {a.model}"):
        run(a)


if __name__ == "__main__":
    main()
