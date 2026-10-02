"""Download CORD v2 (validation + test) at a pinned revision, write images and gold JSONL.

Usage: .venv\\Scripts\\python scripts\\prepare_data.py
Outputs:
  data/images/<split>_<idx>.jpg
  data/gold/{dev,test}.jsonl   one row per receipt: id, split, gold (our schema), cord (raw gt_parse),
                               gold_lines (CORD's own word boxes grouped into lines: an OCR upper bound)
  data/MANIFEST.json
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import ROOT, load_config, path  # noqa: E402
from src.schema import cord_to_receipt  # noqa: E402


def gold_lines(gt: dict) -> list[str]:
    """CORD valid_line -> text lines in reading order (words grouped by CORD's own line ids)."""
    rows = {}
    for vl in gt.get("valid_line", []):
        for w in vl.get("words", []):
            q = w["quad"]
            rid = w.get("row_id", vl.get("group_id"))
            rows.setdefault(rid, []).append(((q["x1"] + q["x3"]) / 2, (q["y1"] + q["y3"]) / 2, w["text"]))
    lines = []
    for rid, ws in rows.items():
        ws.sort()
        lines.append((sum(w[1] for w in ws) / len(ws), " ".join(w[2] for w in ws)))
    lines.sort()
    return [t for _, t in lines]


def main() -> None:
    cfg = load_config()["data"]
    api = HfApi()
    files = [s.rfilename for s in api.dataset_info(cfg["repo"], revision=cfg["revision"]).siblings]
    raw = path("raw")
    img_dir = path("images")
    img_dir.mkdir(parents=True, exist_ok=True)
    path("gold").mkdir(parents=True, exist_ok=True)
    manifest = {"repo": cfg["repo"], "revision": cfg["revision"], "licence": cfg["licence"], "splits": {}}
    for which, split in (("dev", cfg["dev_split"]), ("test", cfg["test_split"])):
        pq = [f for f in files if f.startswith(f"data/{split}-")]
        rows = []
        for f in pq:
            local = hf_hub_download(cfg["repo"], f, repo_type="dataset", revision=cfg["revision"],
                                    local_dir=str(raw))
            df = pd.read_parquet(local)
            for i, r in df.iterrows():
                gt = json.loads(r["ground_truth"])
                idx = int(gt["meta"]["image_id"])
                rid = f"{split}_{idx:03d}"
                img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                out = img_dir / f"{rid}.jpg"
                if not out.exists():
                    img.save(out, quality=92)
                rows.append({
                    "id": rid, "split": which, "image": str(out.relative_to(ROOT)).replace("\\", "/"),
                    "width": img.width, "height": img.height,
                    "gold": cord_to_receipt(gt["gt_parse"]).model_dump(),
                    "cord": gt["gt_parse"],
                    "gold_lines": gold_lines(gt),
                })
        rows.sort(key=lambda x: x["id"])
        with open(path("gold") / f"{which}.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][which] = {"cord_split": split, "n": len(rows), "files": pq}
        print(which, split, len(rows))
    with open(ROOT / "data" / "MANIFEST.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()
