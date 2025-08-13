#!/usr/bin/env python3
import argparse, os, json, datetime as dt
from typing import Any, Dict, List, Tuple


def now_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def find_json_files(root: str) -> List[str]:
    matched: List[str] = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".json"):
                matched.append(os.path.join(r, fn))
    return sorted(matched)


def safe_load(p: str) -> Tuple[Dict[str, Any], str]:
    try:
        with open(p, "r") as f:
            return json.load(f), "ok"
    except Exception as e:
        return {}, f"error: {e}"


def summarize_glue(files: List[str]) -> Dict[str, Any]:
    accs: List[float] = []
    ns: List[int] = []
    for p in files:
        data, _ = safe_load(p)
        m = data.get("metrics") if isinstance(data, dict) else None
        if isinstance(m, dict):
            acc = m.get("accuracy_baseline")
            n = m.get("n") or m.get("size")
            if isinstance(acc, (float, int)):
                accs.append(float(acc))
                if isinstance(n, int):
                    ns.append(n)
    avg_acc = sum(accs) / len(accs) if accs else None
    total_n = sum(ns) if ns else None
    return {"files": len(files), "avg_accuracy_baseline": avg_acc, "total_n": total_n}


def summarize_arc(files: List[str]) -> Dict[str, Any]:
    # ARC smoke currently produces minimal JSON; report file count only.
    return {"files": len(files)}


def summarize_sympy(files: List[str]) -> Dict[str, Any]:
    # Try to summarize driver runs by elapsed_ms if present
    drivers = [p for p in files if os.path.basename(p).startswith("driver_")]
    elapsed: List[float] = []
    for p in drivers:
        data, _ = safe_load(p)
        ems = data.get("elapsed_ms") if isinstance(data, dict) else None
        if isinstance(ems, (int, float)):
            elapsed.append(float(ems))
    avg_elapsed_ms = sum(elapsed) / len(elapsed) if elapsed else None
    return {"files": len(files), "driver_files": len(drivers), "avg_driver_elapsed_ms": avg_elapsed_ms}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="vxor/benchmarks/results")
    ap.add_argument("--out-dir", default="vxor/benchmarks/results/summary")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    arc_dir = os.path.join(args.results_dir, "arc_real")
    glue_dir = os.path.join(args.results_dir, "glue_real")
    sym_dir = os.path.join(args.results_dir, "sympy_linear")

    arc_files = find_json_files(arc_dir) if os.path.isdir(arc_dir) else []
    glue_files = find_json_files(glue_dir) if os.path.isdir(glue_dir) else []
    sym_files = find_json_files(sym_dir) if os.path.isdir(sym_dir) else []

    summary = {
        "timestamp": now_stamp(),
        "counts": {
            "arc_real": len(arc_files),
            "glue_real": len(glue_files),
            "sympy_linear": len(sym_files),
        },
        "glue": summarize_glue(glue_files),
        "arc": summarize_arc(arc_files),
        "sympy_linear": summarize_sympy(sym_files),
    }

    out_path = os.path.join(args.out_dir, f"summary_{summary['timestamp']}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    # also write/update latest.json pointer
    latest = os.path.join(args.out_dir, "latest.json")
    try:
        with open(latest, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass
    print(f"Summary written: {out_path}\nLatest symlinked copy: {latest}")


if __name__ == "__main__":
    main()
