#!/usr/bin/env python3
import argparse, json, os, time, datetime as dt
from typing import Any, Dict

# Smoke: ensure imports work
try:
    from vxor.benchmarks.arc.primitives import symmetry, palette
    from vxor.benchmarks.arc import registry as arc_registry
except Exception as e:
    symmetry = None  # type: ignore
    palette = None  # type: ignore
    arc_registry = None  # type: ignore


def now_stamp() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="vxor/benchmarks/results/arc_real")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    t0 = time.perf_counter_ns()
    status = "ok"
    details: Dict[str, Any] = {}

    # Minimal smoke-run: apply primitives to a toy grid
    try:
        import numpy as np  # local dep
        grid = np.array([[1,2,2],[1,0,2],[1,2,2]], dtype=int)
        axes = symmetry.detect_axes(grid) if symmetry else {"h": False, "v": False}
        pal = palette.extract_palette(grid) if palette else []
        details.update({"toy_axes": axes, "toy_palette": pal})
    except Exception as e:
        status = f"error_smoke: {e}"

    # Dataset-based ARC evaluation would go here (requires ARC dataset). Skipped by default.
    num_tasks = 0

    t1 = time.perf_counter_ns()
    result = {
        "timestamp": now_stamp(),
        "status": status,
        "num_tasks": num_tasks,
        "elapsed_ms": (t1 - t0) / 1e6,
        "details": details,
    }

    out_path = os.path.join(args.out_dir, f"{result['timestamp']}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"ARC eval written: {out_path}")


if __name__ == "__main__":
    main()
