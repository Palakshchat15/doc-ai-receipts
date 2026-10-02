"""Load config.yaml and resolve paths relative to the project root."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    p = ROOT / load_config()["paths"][key]
    return p


def read_jsonl(p: Path) -> list[dict]:
    if not Path(p).exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(p: Path, row: dict) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_name(which: str) -> str:
    """'dev' -> CORD 'validation', 'test' -> 'test'."""
    d = load_config()["data"]
    return {"dev": d["dev_split"], "test": d["test_split"]}.get(which, which)


def gold_for(which: str) -> list[dict]:
    return read_jsonl(path("gold") / f"{which}.jsonl")


def ocr_for(which: str) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(path("ocr") / f"{which}.jsonl")}
