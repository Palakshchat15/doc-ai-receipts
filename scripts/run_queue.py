"""Run the CPU-heavy jobs one after another, each under the shared lock (..\\.llm_lock).

Waits (polling) while another project holds the lock, takes the lock for ONE job, releases it,
and re-checks before the next job, so the RAG project (priority) can take the lock in between.
Every job is resumable, so the queue can be killed and restarted at any time.

Usage: .venv\\Scripts\\python scripts\\run_queue.py --jobs ocr_dev ocr_test regex dev_7b [--poll 180]
Jobs:
  ocr_dev, ocr_test         batch OCR (scripts/run_ocr.py)
  regex                     regex baseline on dev and test (seconds; no lock needed, run for convenience)
  dev_7b, dev_3b            OCR+LLM on dev (prompt tuning)
  test_7b, test_3b          OCR+LLM on test (reported numbers)
  gold_dev_7b, gold_test_7b gold-OCR upper bound
  vision_test_N             vision LLM on the first N test receipts (e.g. vision_test_30)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, load_config  # noqa: E402
from src.lock import held_by_other, llm_lock, lock_owner  # noqa: E402

PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def cmd_for(job: str) -> list[list[str]]:
    vm = load_config()["llm"]["vision_model"]
    e = [PY, "scripts/eval_extract.py"]
    if job == "ocr_dev":
        return [[PY, "scripts/run_ocr.py", "--split", "dev"]]
    if job == "ocr_test":
        return [[PY, "scripts/run_ocr.py", "--split", "test"]]
    if job == "regex":
        return [e + ["--split", s, "--system", "regex"] for s in ("dev", "test")]
    if job.startswith("vision_test_"):
        return [e + ["--split", "test", "--system", "vision", "--model", vm, "--limit", job.split("_")[-1]]]
    split, size = job.split("_")[-2], job.split("_")[-1]
    system = "gold_ocr_llm" if job.startswith("gold_") else "ocr_llm"
    return [e + ["--split", split, "--system", system, "--model", f"qwen2.5:{size}"]]


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def wait_free(poll: int, free_for: int) -> None:
    """Block until the lock has been observed free continuously for `free_for` seconds.

    RAG has priority: its own queue releases and re-takes the lock between jobs, and we must not
    grab it in those short gaps."""
    free_since = None
    while True:
        if held_by_other():
            free_since = None
            log(f"waiting: lock held by {lock_owner().splitlines()[0]}")
        else:
            free_since = free_since or time.time()
            if time.time() - free_since >= free_for:
                return
        time.sleep(min(poll, free_for) if free_since else poll)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--poll", type=int, default=180)
    ap.add_argument("--free-for", type=int, default=600, help="lock must be free this long before we start")
    a = ap.parse_args()
    for job in a.jobs:
        for c in cmd_for(job):
            while True:
                wait_free(a.poll, a.free_for)
                log(f"start {job}: {' '.join(c[1:])}")
                # eval_extract.py takes the lock itself for LLM jobs; OCR is locked here.
                if "run_ocr.py" in c[1]:
                    try:
                        with llm_lock(job):
                            rc = subprocess.call(c, cwd=ROOT)
                    except Exception as e:  # noqa: BLE001  (lock taken by someone else meanwhile)
                        log(f"lock race: {e}")
                        rc = -1
                else:
                    rc = subprocess.call(c, cwd=ROOT)
                if rc != 0 and held_by_other():
                    log("another project took the lock; waiting to retry")
                    continue
                break
            if rc != 0:
                log(f"{job} exited with {rc}; stopping the queue")
                sys.exit(rc)
        log(f"done {job}")


if __name__ == "__main__":
    main()
