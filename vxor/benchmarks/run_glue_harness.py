from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import math
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


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/Infinity with None to keep JSON valid."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    return obj


def run(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLUE wrapper: offline audited run with allowlists and per-task artifacts")
    ap.add_argument("--root", required=True, type=str, help="GLUE root directory containing per-task subfolders")
    ap.add_argument("--tasks", nargs="+", default=[
        "cola","sst2","mrpc","stsb","qqp","mnli","qnli","rte","wnli"
    ])
    ap.add_argument("--split", type=str, default="dev", choices=["dev","test"]) 
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--results-root", type=str, default="vxor/benchmarks/results/glue_real")
    ap.add_argument("--allowlist-json", type=str, default="vxor/benchmarks/allowlists/arc_glue_allowlist.json")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print("[ERR] GLUE root missing", file=sys.stderr)
        return 2

    results_root = Path(args.results_root)
    _ensure_dir(results_root)
    ts = utc_now().replace(":", "").replace("-", "").replace(".", "").replace("Z", "Z")[:15]

    # Ephemeral directory-level allowlist for harness
    allowlist_dir = results_root / ts
    _ensure_dir(allowlist_dir)
    allowlist_path = allowlist_dir / "glue.allow"
    with allowlist_path.open("w", encoding="utf-8") as f:
        f.write(sha256_dir(root) + "\n")

    # Update per-file allowlist JSON (merge)
    allow_json_path = Path(args.allowlist_json)
    _ensure_dir(allow_json_path.parent)
    allow_json: Dict[str, Any] = {}
    if allow_json_path.exists():
        try:
            allow_json = json.loads(allow_json_path.read_text(encoding="utf-8"))
        except Exception:
            allow_json = {}
    allow_json["glue_root"] = str(root)
    allow_json["glue_root_dir_hash"] = sha256_dir(root)
    allow_json.setdefault("glue_files", {}).update(_hash_files_under(root))
    allow_json_path.write_text(json.dumps(allow_json, indent=2, sort_keys=True), encoding="utf-8")

    # Hermetic env
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("TZ", "UTC")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    if args.offline:
        env["VXOR_FORCE_OFFLINE"] = "1"

    tasks_csv = ",".join(args.tasks)
    cmd = [
        sys.executable, "-m", "bench.glue_eval",
        "--glue-root", str(root),
        "--tasks", tasks_csv,
        "--solver", "vxor.benchmarks.solvers.vxor_glue_solver:solver",
        "--allowlist", str(allowlist_path),
        "--split", args.split,
        "--batch-size", str(args.batch_size),
        "--threads", "1",
        "--output-dir", str(results_root),
    ]
    print("[RUN] ", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, text=True)
    if proc.returncode != 0:
        return proc.returncode

    # Discover run directory: filter only subdirs that contain an audit.json
    subdirs = sorted([p for p in results_root.iterdir() if p.is_dir()])
    candidates = [p for p in subdirs if (p / "audit.json").exists()]
    if not candidates:
        print("[ERR] No run directory with audit.json found under results_root", file=sys.stderr)
        return 2
    # Choose the most recently modified candidate
    run_dir = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    # Load audit summary
    harness_summary: Dict[str, Any] = {}
    audit_json = run_dir / "audit.json"
    audit_jsonl = run_dir / "audit.jsonl"
    if audit_json.exists():
        text = audit_json.read_text(encoding="utf-8")
        try:
            # First try strict JSON
            harness_summary = json.loads(text)
        except Exception:
            # Be robust to NaN/Infinity tokens emitted upstream
            try:
                harness_summary = json.loads(text, parse_constant=lambda x: None)
            except Exception:
                print("[WARN] Could not parse audit.json; metrics will be omitted", file=sys.stderr)
                harness_summary = {}

    # Compute p95 per-task from GLUE_BATCH entries
    per_task_times: Dict[str, List[float]] = {t: [] for t in args.tasks}
    if audit_jsonl.exists():
        with audit_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("event") == "GLUE_BATCH":
                    t = str(rec.get("task"))
                    sec = float(rec.get("seconds", 0.0))
                    if t in per_task_times:
                        per_task_times[t].append(sec)
    per_task_p95 = {t: _p95(vs) for t, vs in per_task_times.items()}

    # Write per-task artifacts
    host = os.uname().nodename
    git_commit = _git_commit_safe()
    for t in args.tasks:
        t_dir = results_root / t
        _ensure_dir(t_dir)
        out = t_dir / f"{ts}.json"
        metrics = harness_summary.get("results", {}).get(t, {})
        rec = {
            "timestamp_utc": utc_now(),
            "host": host,
            "git_commit": git_commit,
            "argv": sys.argv,
            "dataset": {
                "glue_root": str(root),
                "glue_root_dir_hash": sha256_dir(root),
            },
            "task": t,
            "split": args.split,
            "latency": {
                "p95_time_per_task_s": float(per_task_p95.get(t, 0.0)),
            },
            "harness_metrics": metrics,
        }
        out.write_text(json.dumps(_sanitize(rec), indent=2, allow_nan=False), encoding="utf-8")
        print(f"[OK] GLUE task artifact written: {out}")

    return 0


def _git_commit_safe() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(run())
