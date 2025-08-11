#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _PathForSys

# Avoid shadowing the Hugging Face 'datasets' package by a local './datasets' folder.
# Remove the repository root from sys.path so 'from datasets import load_dataset'
# resolves to the installed package.
try:
    _repo_root = _PathForSys(__file__).resolve().parents[2]
    _repo_root_str = str(_repo_root)
    if _repo_root_str in sys.path:
        sys.path = [p for p in sys.path if p != _repo_root_str]
except Exception:
    pass

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

try:
    from datasets import load_dataset  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "[ERR] Missing dependency: 'datasets'. Install with: python3 -m pip install 'datasets==2.20.0' 'pyarrow>=11,<20'"
    )

"""
We export GLUE dev.jsonl matching bench/glue_eval.TASKS expectations.

Expected target fields per task:
  - cola, sst2:                sentence
  - mrpc, wnli, stsb:          sentence1, sentence2
  - qqp:                        question1, question2
  - qnli:                       question, sentence
  - rte:                        premise, hypothesis
  - mnli:                       premise, hypothesis

However, Hugging Face source field names differ for some tasks (e.g., RTE uses
sentence1/sentence2). We define a mapping from HF source fields to our target
schema for each task.
"""

# HF source -> target field mapping
FIELDS_MAP: Dict[str, Dict[str, str]] = {
    "cola": {"sentence": "sentence"},
    "sst2": {"sentence": "sentence"},
    "mrpc": {"sentence1": "sentence1", "sentence2": "sentence2"},
    "qqp": {"question1": "question1", "question2": "question2"},
    "qnli": {"question": "question", "sentence": "sentence"},
    # RTE on HF has sentence1/sentence2, but our harness expects premise/hypothesis
    "rte": {"sentence1": "premise", "sentence2": "hypothesis"},
    "wnli": {"sentence1": "sentence1", "sentence2": "sentence2"},
    "mnli": {"premise": "premise", "hypothesis": "hypothesis"},
    "stsb": {"sentence1": "sentence1", "sentence2": "sentence2"},
}

DEFAULT_TASKS = ["sst2", "rte", "mnli", "qnli", "qqp", "cola", "mrpc", "stsb", "wnli"]


def export_glue_dev(root: Path, tasks: List[str], *, force: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        tdir = root / task
        tdir.mkdir(parents=True, exist_ok=True)
        outp = tdir / "dev.jsonl"
        if outp.exists():
            try:
                size = outp.stat().st_size
            except FileNotFoundError:
                size = 0
            if size > 0 and not force:
                print(f"[SKIP] {outp} exists (size={size})")
                continue
            msg = "[OVERWRITE]" if force else "[FIX] Empty file – regenerating"
            print(f"{msg} {outp}")
        print(f"[EXPORT] {task} → {outp}")
        ds = load_dataset("glue", task)
        splits = ["validation_matched", "validation_mismatched"] if task == "mnli" else ["validation"]
        n = 0
        with outp.open("w", encoding="utf-8") as f:
            for sp in splits:
                for ex in ds[sp]:
                    # Map HF fields -> target fields
                    rec: Dict[str, Any] = {}
                    for src, dst in FIELDS_MAP[task].items():
                        rec[dst] = ex[src]
                    if task == "stsb":
                        rec["score"] = float(ex["label"])  # regression target
                    else:
                        rec["label"] = int(ex["label"])   # classification label
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
        print(f"[OK] wrote {outp} (n={n})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export GLUE dev.jsonl files to expected schema for local offline eval")
    ap.add_argument("--root", required=True, type=str, help="GLUE output root directory")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--force", action="store_true", help="Overwrite existing dev.jsonl files")
    args = ap.parse_args()

    export_glue_dev(Path(args.root), args.tasks, force=bool(args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
