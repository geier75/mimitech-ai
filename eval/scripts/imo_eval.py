#!/usr/bin/env python3
import argparse, json, os, time, datetime as dt, subprocess, sys
from typing import Any, Dict


def now_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="vxor/benchmarks/results/sympy_linear")
    ap.add_argument("--sizes", nargs="*", default=["3x2", "10x8"])  # quick sanity
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--policy", choices=["auto","exact","numeric"], default="auto")
    ap.add_argument("--timeout-s", type=float, default=10.0)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    t0 = time.perf_counter_ns()

    cmd = [
        sys.executable,
        "vxor/benchmarks/sympy_lin_sys_bench.py",
        "--mode", "random",
        "--sizes", *args.sizes,
        "--repeats", str(args.repeats),
        "--loops", str(args.loops),
        "--warmup", str(args.warmup),
        "--policy", str(args.policy),
        "--timeout-s", str(args.timeout_s),
        "--output-dir", os.path.join("vxor","benchmarks","results","sympy_linear"),
    ]

    status = "ok"
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        status = f"error:{e.returncode}"
    except Exception as e:
        status = f"error:{type(e).__name__}:{e}"

    t1 = time.perf_counter_ns()
    result = {
        "timestamp": now_stamp(),
        "status": status,
        "elapsed_ms": (t1 - t0) / 1e6,
        "note": "Detail-JSONs werden vom sympy_lin_sys_bench.py direkt unter results/sympy_linear/ geschrieben.",
    }

    out_path = os.path.join(args.out_dir, f"driver_{result['timestamp']}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"IMO eval driver written: {out_path}")


if __name__ == "__main__":
    main()
