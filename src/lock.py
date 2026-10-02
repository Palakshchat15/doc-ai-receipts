"""Shared LLM lock (AI_Engineer_Portfolio/.llm_lock) so two projects never run LLM batch jobs at once."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .config import ROOT

LOCK = ROOT.parent / ".llm_lock"
PROJECT = "doc_ai_receipts"


class LockHeld(RuntimeError):
    pass


def lock_owner() -> str | None:
    """Return the text of the lock if held, else None."""
    if not LOCK.exists():
        return None
    try:
        return LOCK.read_text(encoding="utf-8").strip() or "(empty lock file)"
    except OSError:
        return "(unreadable lock file)"


def held_by_other() -> bool:
    o = lock_owner()
    return o is not None and PROJECT not in o


@contextmanager
def llm_lock(job: str):
    """Create the lock for the duration of a job; always removed afterwards (also on failure)."""
    o = lock_owner()
    if o is not None and PROJECT not in o:
        raise LockHeld(f"LLM lock is held by another project: {o}")
    info = {"project": PROJECT, "job": job, "pid": os.getpid(),
            "started": datetime.now().isoformat(timespec="seconds")}
    # O_EXCL so a lock created by someone else between the check and now is not overwritten
    if o is None:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise LockHeld(f"LLM lock appeared meanwhile: {lock_owner()}") from e
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(info))
    else:
        LOCK.write_text(json.dumps(info), encoding="utf-8")
    try:
        yield info
    finally:
        try:
            if LOCK.exists() and PROJECT in LOCK.read_text(encoding="utf-8"):
                LOCK.unlink()
        except OSError:
            time.sleep(1)
            if LOCK.exists():
                LOCK.unlink()
