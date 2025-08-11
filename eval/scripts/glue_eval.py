#!/usr/bin/env python3
import argparse, json, os, time, datetime as dt
from typing import Any, Dict


def now_stamp() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="vxor/benchmarks/results/glue_real")
    ap.add_argument("--task", default="sst2")
    ap.add_argument("--limit", type=int, default=256)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    t0 = time.perf_counter_ns()
    status = "ok"
    metrics: Dict[str, Any] = {}

    try:
        from datasets import load_dataset  # type: ignore
        import evaluate  # type: ignore
        ds = load_dataset("glue", args.task, split="validation")
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        # trivial baseline: predict class 1 always if label column exists
        y_true = ds["label"] if "label" in ds.column_names else []
        y_pred = [1 for _ in y_true]
        metric = evaluate.load("accuracy")
        acc = metric.compute(references=y_true, predictions=y_pred)["accuracy"] if y_true else None
        metrics = {"task": args.task, "n": len(y_true), "accuracy_baseline": acc}
    except Exception as e:
        status = f"skipped: {e}"

    t1 = time.perf_counter_ns()
    result = {
        "timestamp": now_stamp(),
        "status": status,
        "metrics": metrics,
        "elapsed_ms": (t1 - t0) / 1e6,
    }

    out_path = os.path.join(args.out_dir, f"{result['timestamp']}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"GLUE eval written: {out_path}")


if __name__ == "__main__":
    main()
