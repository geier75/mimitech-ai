#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
REPLACEMENT = "info@mimitechai.com"

# Directories to skip entirely (datasets, third-party, build artifacts)
SKIP_DIRS = {
    ".git",
    "datasets",
    "whisper.cpp",
    "node_modules",
    "venv",
    ".venv",
    "build",
    "dist",
    "vxor/benchmarks/results",
    "__pycache__",
}

# File extensions to consider as text
TEXT_EXTS = {
    ".md", ".markdown", ".mdx",
    ".txt", ".rst",
    ".yml", ".yaml",
    ".toml", ".ini", ".cfg",
    ".py", ".sh", ".bash", ".zsh",
    ".json", ".jsonl",
}

# File patterns to skip even if extension matches
SKIP_FILE_BASENAMES = {
    "AUTHORS",  # third-party authorship lists
}

# Path prefixes to skip (relative to repo root)
SKIP_PATH_PREFIXES = [
    "datasets/",               # do not touch benchmark datasets
    "whisper.cpp/",            # vendor code
    "vxor/benchmarks/results/" # generated outputs
]


def is_skip_path(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    for pref in SKIP_PATH_PREFIXES:
        if rel.startswith(pref):
            return True
    parts = set(rel.split("/"))
    if parts & SKIP_DIRS:
        return True
    if path.name in SKIP_FILE_BASENAMES:
        return True
    return False


def should_process_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if is_skip_path(path):
        return False
    # Only process text-like files
    return path.suffix.lower() in TEXT_EXTS


def align_contacts(apply: bool) -> int:
    changed = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # prune skip dirs for efficiency
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = Path(dirpath, fn)
            if not should_process_file(p):
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            # Skip typical machine-generated JSONL datasets by quick heuristic
            if p.suffix.lower() in {".jsonl"} and "datasets/" in p.as_posix():
                continue
            new_text, n = EMAIL_RE.subn(REPLACEMENT, text)
            if n > 0:
                if apply:
                    p.write_text(new_text, encoding="utf-8")
                changed += 1
    return changed


def main():
    ap = argparse.ArgumentParser(description="Align all contact emails to info@mimitechai.com with safe exclusions.")
    ap.add_argument("--apply", action="store_true", help="apply changes (otherwise dry-run)")
    args = ap.parse_args()
    cnt = align_contacts(apply=args.apply)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] files changed: {cnt}")

if __name__ == "__main__":
    main()
