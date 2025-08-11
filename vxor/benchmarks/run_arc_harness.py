from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bench.tools.audit import sha256_dir, sha256_of_file, utc_now


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _hash_files_under(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in sorted([p for p in root.rglob("*") if p.is_file()]):
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
        out[rel] = sha256_of_file(p)
    return out


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, int(0.95 * (len(s) - 1)))
    return float(s[k])


def run(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ARC-AGI-1 wrapper: offline audited run with allowlists and artifacts")
    ap.add_argument("--train-root", required=True, type=str)
    ap.add_argument("--eval-root", required=True, type=str)
    ap.add_argument("--attempts", type=int, default=1, help="kept for compatibility; current harness supports single guess")
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-leakage", action="store_true")
    ap.add_argument("--results-root", type=str, default="vxor/benchmarks/results/arc_real")
    ap.add_argument("--allowlist-json", type=str, default="vxor/benchmarks/allowlists/arc_glue_allowlist.json")
    args = ap.parse_args(argv)

    train = Path(args.train_root)
    evald = Path(args.eval_root)
    if not train.exists() or not evald.exists():
        print("[ERR] ARC paths missing", file=sys.stderr)
        return 2

    # Results layout
    results_root = Path(args.results_root)
    _ensure_dir(results_root)
    ts = utc_now().replace(":", "").replace("-", "").replace(".", "").replace("Z", "Z")[:15]

    # Create ephemeral allowlist (directory-level) for harness enforcement
    allowlist_dir = results_root / ts
    _ensure_dir(allowlist_dir)
    allowlist_path = allowlist_dir / "arc.allow"
    with allowlist_path.open("w", encoding="utf-8") as f:
        f.write(sha256_dir(train) + "\n")
        f.write(sha256_dir(evald) + "\n")

    # Produce per-file allowlist JSON (append/merge into single file path)
    allowlists_root = Path(args.allowlist_json).parent
    _ensure_dir(allowlists_root)
    allow_json_path = Path(args.allowlist_json)
    allow_json: Dict[str, Any] = {}
    if allow_json_path.exists():
        try:
            allow_json = json.loads(allow_json_path.read_text(encoding="utf-8"))
        except Exception:
            allow_json = {}
    allow_json["arc_train_root"] = str(train)
    allow_json["arc_eval_root"] = str(evald)
    allow_json["arc_train_dir_hash"] = sha256_dir(train)
    allow_json["arc_eval_dir_hash"] = sha256_dir(evald)
    allow_json.setdefault("arc_train_files", {}).update(_hash_files_under(train))
    allow_json.setdefault("arc_eval_files", {}).update(_hash_files_under(evald))
    allow_json_path.write_text(json.dumps(allow_json, indent=2, sort_keys=True), encoding="utf-8")

    # Prepare environment for hermetic offline run
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("TZ", "UTC")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    if args.offline:
        env["VXOR_FORCE_OFFLINE"] = "1"

    # Delegate to harness; set output-dir to results_root so the run dir is created under arc_real/
    cmd = [
        sys.executable, "-m", "bench.arc_eval",
        "--arc-train", str(train),
        "--arc-test", str(evald),
        "--solver", "vxor.benchmarks.solvers.vxor_arc_solver:solver",
        "--allowlist", str(allowlist_path),
        "--output-dir", str(results_root),
        "--threads", "1",
    ]
    print("[RUN] ", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, text=True)
    if proc.returncode != 0:
        return proc.returncode

    # Find the harness-created run directory (latest subdir under results_root)
    subdirs = sorted([p for p in results_root.iterdir() if p.is_dir()])
    if not subdirs:
        print("[ERR] No run directory found under results_root", file=sys.stderr)
        return 2
    run_dir = subdirs[-1]

    # Load audit files
    audit_json = run_dir / "audit.json"
    audit_jsonl = run_dir / "audit.jsonl"
    summary: Dict[str, Any] = {}
    if audit_json.exists():
        summary = json.loads(audit_json.read_text(encoding="utf-8"))

    # Compute p95 latencies from JSONL
    times: List[float] = []
    if audit_jsonl.exists():
        with audit_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("event") == "ARC_TEST":
                    t = float(rec.get("seconds", 0.0))
                    times.append(t)
    p95 = _p95(times)

    # Enrich and write summary.json
    meta = {
        "timestamp_utc": utc_now(),
        "argv": sys.argv,
        "host": os.uname().nodename,
        "git_commit": _git_commit_safe(),
        "dataset": {
            "arc_train_dir": str(train),
            "arc_eval_dir": str(evald),
            "arc_train_dir_hash": sha256_dir(train),
            "arc_eval_dir_hash": sha256_dir(evald),
            "allowlist_json": str(allow_json_path),
        },
        "latency": {
            "p95_time_per_test_s": p95,
            "n_tests": len(times),
        },
        "note": "attempts parameter is recorded but current harness supports one prediction per test",
    }
    out_summary = run_dir / "summary.json"
    summary_out = {"harness": summary, "meta": meta}
    out_summary.write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    print(f"[OK] ARC summary written: {out_summary}")
    return 0


def _git_commit_safe() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(run())
