from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# --------- Time ---------

def utc_now() -> str:
    # Use datetime.UTC if available (Py 3.11+); fallback to timezone.utc for Py 3.9/3.10
    tz = getattr(dt, "UTC", None) or dt.timezone.utc
    ts = dt.datetime.now(tz).isoformat(timespec="microseconds")
    if ts.endswith("+00:00"):
        ts = ts[:-6] + "Z"
    return ts

# --------- Hashing ---------

def sha256_of_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(root: Path) -> str:
    # Stable tree hash over files under root (relative path + file hash)
    items: list[tuple[str, str]] = []
    for p in sorted([p for p in root.rglob("*") if p.is_file()]):
        # Ignore common OS noise for stable hashing across macOS/Windows
        # - .DS_Store, Thumbs.db files
        # - any component starting with '._' (AppleDouble)
        # - .ipynb_checkpoints directories
        parts = p.parts
        skip = False
        for part in parts:
            if part in (".DS_Store", "Thumbs.db", ".ipynb_checkpoints"):
                skip = True
                break
            if part.startswith("._"):
                skip = True
                break
        if skip:
            continue
        rel = str(p.relative_to(root)).replace(os.sep, "/")
        items.append((rel, sha256_of_file(p)))
    h = hashlib.sha256()
    for rel, fh in items:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(fh.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def sha256_of_array(arr: np.ndarray) -> str:
    a64 = np.asarray(arr, dtype=np.float64)
    return hashlib.sha256(a64.tobytes(order="C")).hexdigest()

# --------- Policy / Allowlist ---------

def load_allowlist(p: Path) -> set[str]:
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(s.split()[0])
    return set(rows)

# --------- Audit ---------

@dataclasses.dataclass
class AuditContext:
    run_dir: Path
    jsonl_path: Path
    seed: int | None
    host: str
    versions: Dict[str, Any]


def append_jsonl(ctx: AuditContext, event: str, **fields: Any) -> None:
    rec = {
        "ts_utc": utc_now(),
        "event": event,
        "seed": ctx.seed,
        "host": ctx.host,
        "versions": ctx.versions,
    }
    rec.update(fields)
    with ctx.jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_audit_summary(ctx: AuditContext, summary: Dict[str, Any]) -> None:
    out = ctx.run_dir / "audit.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

# --------- Env / Build ID ---------

def set_hermetic_env(threads: int) -> None:
    os.environ.setdefault("TZ", "UTC")
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)


def pip_freeze_all() -> str:
    try:
        proc = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=False)
        return proc.stdout
    except Exception as ex:  # pragma: no cover
        return f"pip_freeze_error: {ex}"


def compute_build_id(script_text: str, sbom_text: str, docker_digest: str = "") -> str:
    return hashlib.sha256((sha256_of_text(script_text) + sha256_of_text(sbom_text) + docker_digest).encode("utf-8")).hexdigest()
