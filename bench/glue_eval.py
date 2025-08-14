from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

from bench.tools.audit import (
    AuditContext,
    append_jsonl,
    compute_build_id,
    load_allowlist,
    pip_freeze_all,
    set_hermetic_env,
    sha256_dir,
    utc_now,
)

# ----------------------------- IO -------------------------------

@dataclass
class Example:
    guid: str
    fields: Dict[str, Any]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_task_split(dir_path: Path, split: str) -> List[Dict[str, Any]]:
    p = dir_path / f"{split}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"missing split file: {p}")
    return read_jsonl(p)


# ------------------------- Task registry ------------------------

TaskSpec = Dict[str, Any]

TASKS: Dict[str, TaskSpec] = {
    # binary classification
    "cola": {"type": "binary", "text_fields": ["sentence"], "label_field": "label", "metric": "matthews"},
    "sst2": {"type": "binary", "text_fields": ["sentence"], "label_field": "label", "metric": "accuracy"},
    "mrpc": {"type": "binary", "text_fields": ["sentence1", "sentence2"], "label_field": "label", "metric": "acc_f1"},
    "qqp": {"type": "binary", "text_fields": ["question1", "question2"], "label_field": "label", "metric": "acc_f1"},
    "qnli": {"type": "binary", "text_fields": ["question", "sentence"], "label_field": "label", "metric": "accuracy"},
    "rte": {"type": "binary", "text_fields": ["premise", "hypothesis"], "label_field": "label", "metric": "accuracy"},
    "wnli": {"type": "binary", "text_fields": ["sentence1", "sentence2"], "label_field": "label", "metric": "accuracy"},
    # multiclass
    "mnli": {"type": "multiclass", "num_labels": 3, "text_fields": ["premise", "hypothesis"], "label_field": "label", "metric": "accuracy"},
    # regression
    "stsb": {"type": "regression", "text_fields": ["sentence1", "sentence2"], "label_field": "score", "metric": "pearson_spearman"},
}


# ----------------------- Metric compute -------------------------

def compute_metrics(task: str, y_true: List[Any], y_pred: List[Any]) -> Dict[str, float]:
    spec = TASKS[task]
    # Guard against empty datasets to avoid NaN metrics and invalid JSON output
    if len(y_true) == 0:
        if spec["metric"] == "accuracy":
            return {"accuracy": 0.0}
        if spec["metric"] == "acc_f1":
            return {"accuracy": 0.0, "f1": 0.0}
        if spec["metric"] == "matthews":
            return {"matthews": 0.0}
        if spec["metric"] == "pearson_spearman":
            return {"pearson": 0.0, "spearman": 0.0}
    if spec["metric"] == "accuracy":
        return {"accuracy": float(accuracy_score(y_true, y_pred))}
    if spec["metric"] == "acc_f1":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
        }
    if spec["metric"] == "matthews":
        return {"matthews": float(matthews_corrcoef(y_true, y_pred))}
    if spec["metric"] == "pearson_spearman":
        y_true_arr = np.asarray(y_true, dtype=np.float64)
        y_pred_arr = np.asarray(y_pred, dtype=np.float64)
        pr = float(pearsonr(y_true_arr, y_pred_arr)[0]) if len(y_true_arr) > 1 else 0.0
        sr = float(spearmanr(y_true_arr, y_pred_arr)[0]) if len(y_true_arr) > 1 else 0.0
        return {"pearson": pr, "spearman": sr}
    raise ValueError(f"unknown metric for task {task}")


# ------------------------- Solver API ---------------------------

# solver_fn signature: (task: str, examples: List[Dict[str, Any]]) -> List[Any]
#   - returns predictions: for classification: ints; for stsb: floats


# --------------------------- Eval -------------------------------

def eval_glue_task(
    task: str,
    dir_path: Path,
    solver_fn: Callable[[str, List[Dict[str, Any]]], List[Any]],
    ctx: AuditContext,
    split: str,
    batch_size: int,
    dump_errors: bool = False,
    save_preds: bool = False,
    topn: int = 1,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    spec = TASKS[task]
    data = load_task_split(dir_path, split)
    y_true: List[Any] = []
    y_pred: List[Any] = []
    # For optional dumps
    pred_records: List[Dict[str, Any]] = []
    err_records: List[Dict[str, Any]] = []
    # Ensure output dirs exist lazily
    preds_dir = ctx.run_dir / "preds"
    errors_dir = ctx.run_dir / "errors"

    ex_idx = 0
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        t0 = time.perf_counter()
        preds = solver_fn(task, batch)
        dt_s = time.perf_counter() - t0
        if not isinstance(preds, (list, tuple)) or len(preds) != len(batch):
            append_jsonl(ctx, "ERROR", reason="solver_shape", task=task, i=i, n_batch=len(batch))
            raise SystemExit("solver returned wrong number of predictions")
        # collect labels
        for ex, p in zip(batch, preds):
            # expected ground truth fields
            y = ex.get(spec["label_field"])  # may be missing in test; harness expects available
            # Apply thresholds only for binary tasks if prediction is a float-like score
            pred_val: Any = p
            score_val: Optional[float] = None
            if spec["type"] == "binary" and not isinstance(p, (int, np.integer)):
                try:
                    score_val = float(p)
                    thr = (thresholds or {}).get(task, 0.5)
                    pred_val = int(score_val >= float(thr))
                except Exception:
                    # fallback to int cast if possible
                    try:
                        pred_val = int(p)
                    except Exception:
                        pred_val = p
                        score_val = None
            else:
                # Normalize multiclass predictions to int when possible
                if spec["type"] in ("binary", "multiclass"):
                    try:
                        pred_val = int(p)
                    except Exception:
                        pass

            y_true.append(y)
            y_pred.append(pred_val)

            if save_preds or dump_errors:
                # Derive a guid if missing
                guid = ex.get("guid") or ex.get("id") or ex.get("idx") or f"{task}:{split}:{ex_idx}"
                text_payload = {k: ex.get(k) for k in spec.get("text_fields", []) if k in ex}
                rec: Dict[str, Any] = {
                    "guid": guid,
                    "idx": ex_idx,
                    "task": task,
                    "split": split,
                    "y_true": y,
                    "y_pred": pred_val,
                }
                if score_val is not None:
                    rec["y_score"] = score_val
                if text_payload:
                    rec["text"] = text_payload
                pred_records.append(rec)
                # Misclassifications only for classification tasks
                if dump_errors and spec["type"] in ("binary", "multiclass") and (y is not None) and (pred_val != y):
                    err_records.append(rec)
            ex_idx += 1
        append_jsonl(ctx, "GLUE_BATCH", task=task, i=i, n=len(batch), seconds=dt_s)

    metrics = compute_metrics(task, y_true, y_pred)
    append_jsonl(ctx, "GLUE_TASK_DONE", task=task, n=len(data), **metrics)

    # Write optional artifacts
    if save_preds and pred_records:
        preds_dir.mkdir(parents=True, exist_ok=True)
        out_p = preds_dir / f"{task}.jsonl"
        with out_p.open("w", encoding="utf-8") as f:
            for rec in pred_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        append_jsonl(ctx, "GLUE_PRED_DUMP", task=task, path=str(out_p), n=len(pred_records))
        if topn and topn > 1:
            # We do not have top-N candidates from the solver; log unavailability once per task.
            append_jsonl(ctx, "GLUE_TOPN_UNAVAILABLE", task=task, requested=topn)
    if dump_errors and err_records:
        errors_dir.mkdir(parents=True, exist_ok=True)
        out_e = errors_dir / f"{task}.jsonl"
        with out_e.open("w", encoding="utf-8") as f:
            for rec in err_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        append_jsonl(ctx, "GLUE_ERROR_DUMP", task=task, path=str(out_e), n=len(err_records))
    return metrics


# ---------------------------- Main ------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="vXor GLUE Benchmark Harness (deterministisch, auditiert)")
    p.add_argument("--glue-root", type=str, required=True, help="Pfad zum GLUE-Root (Unterordner pro Task mit dev.jsonl/test.jsonl)")
    p.add_argument("--tasks", type=str, default="cola,sst2,mrpc,stsb,qqp,mnli,qnli,rte,wnli", help="Kommagetrennte Liste von GLUE-Tasks")
    p.add_argument("--solver", type=str, required=True, help="Dotted Callable: modul.pfad:funktion (task, batch) -> predictions")
    p.add_argument("--allowlist", type=str, required=True, help="Pfad zur Allowlist (SHA256) für Dataset-Integrität (Directory-Hash)")
    p.add_argument("--split", type=str, default="dev", choices=["dev", "test"], help="Datensplit")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output-dir", type=str, default="runs/glue")
    # New analysis/calibration options
    p.add_argument("--dump-errors", action="store_true", help="Schreibe fehllabelte Beispiele pro Task in run_dir/errors/{task}.jsonl")
    p.add_argument("--save-preds", action="store_true", help="Schreibe Vorhersagen pro Task in run_dir/preds/{task}.jsonl")
    p.add_argument("--topn", type=int, default=1, help="Anzahl Top-N Kandidaten (nur wenn Solver solche liefert; sonst Info-Log)")
    p.add_argument("--thresholds-json", type=str, default="", help="JSON-Datei: {task: schwelle} für binäre Tasks")

    args = p.parse_args(argv)

    if int(args.threads) != 1:
        print("[ZTM] threads != 1 disabled by policy for determinism", file=sys.stderr)
        return 2
    set_hermetic_env(args.threads)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_dir = outdir / (utc_now().replace(":", "").replace("-", "").replace(".", "").replace("Z", "Z")[:15])
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_jsonl = run_dir / "audit.jsonl"
    ctx = AuditContext(
        run_dir=run_dir,
        jsonl_path=audit_jsonl,
        seed=None,
        host=os.uname().nodename,
        versions={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    )

    script_text = Path(__file__).read_text(encoding="utf-8")
    sbom_txt = pip_freeze_all()
    build_id = compute_build_id(script_text, sbom_txt, os.environ.get("DOCKER_IMAGE_DIGEST", ""))
    append_jsonl(ctx, "RUN_START", build_id=build_id)

    glue_root = Path(args.glue_root)
    if not glue_root.exists():
        append_jsonl(ctx, "ERROR", reason="dataset_missing", root=str(glue_root))
        return 2

    allow = load_allowlist(Path(args.allowlist))
    h_root = sha256_dir(glue_root)
    if h_root not in allow:
        append_jsonl(ctx, "ERROR", reason="allowlist_miss", root=h_root)
        return 2
    append_jsonl(ctx, "DATA_VERIFY_OK", root=h_root)

    # load solver
    mod_name, func_name = args.solver.split(":", 1) if ":" in args.solver else args.solver.rsplit(".", 1)
    mod = __import__(mod_name, fromlist=[func_name])
    solver_fn = getattr(mod, func_name)

    # Load thresholds mapping if provided
    thresholds: Optional[Dict[str, float]] = None
    if args.thresholds_json:
        tpath = Path(args.thresholds_json)
        if tpath.exists():
            try:
                thresholds = json.loads(tpath.read_text(encoding="utf-8"))
            except Exception as ex:
                append_jsonl(ctx, "WARN", reason="thresholds_parse_error", path=str(tpath), error=str(ex))
        else:
            append_jsonl(ctx, "WARN", reason="thresholds_missing", path=str(tpath))

    results: Dict[str, Any] = {}
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        if task not in TASKS:
            append_jsonl(ctx, "ERROR", reason="unknown_task", task=task)
            return 2
        try:
            metrics = eval_glue_task(
                task,
                glue_root / task,
                solver_fn,
                ctx,
                args.split,
                args.batch_size,
                dump_errors=args.dump_errors,
                save_preds=args.save_preds,
                topn=int(args.topn),
                thresholds=thresholds,
            )
            results[task] = metrics
        except Exception as ex:
            append_jsonl(ctx, "ERROR", reason="uncaught_exception", task=task, error=str(ex))
            print(f"[ZTM][FATAL] {ex}", file=sys.stderr)
            return 2

    summary = {
        "timestamp_utc": utc_now(),
        "build_id": build_id,
        "dataset": {
            "root_hash": h_root,
            "split": args.split,
        },
        "results": results,
        "policy": {
            "threads": 1,
            "simulation": "disabled_by_policy",
        },
        "sbom": sbom_txt,
    }
    from bench.tools.audit import write_audit_summary
    write_audit_summary(ctx, summary)
    append_jsonl(ctx, "RUN_END", ok=True)
    print(f"[OK] GLUE-Audit geschrieben nach {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
