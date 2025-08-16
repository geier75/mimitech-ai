from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bench.tools.audit import (
    AuditContext,
    append_jsonl,
    compute_build_id,
    load_allowlist,
    pip_freeze_all,
    set_hermetic_env,
    sha256_dir,
    sha256_of_text,
    utc_now,
)

# ----------------------------- Types -----------------------------

@dataclass
class ArcTask:
    task_id: str
    train: List[Dict[str, Any]]
    test: List[Dict[str, Any]]


# ----------------------------- IO -------------------------------

def read_arc_task(path: Path) -> ArcTask:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return ArcTask(task_id=path.stem, train=obj.get("train", []), test=obj.get("test", []))


def load_arc_split(dir_path: Path) -> List[ArcTask]:
    # Ignore OS noise files that may appear alongside real JSON (e.g., AppleDouble '._*.json', '.DS_Store').
    def _is_noise(p: Path) -> bool:
        name = p.name
        if name.startswith("._"):
            return True
        if name in (".DS_Store", "Thumbs.db"):
            return True
        return False
    files = sorted([p for p in dir_path.glob("*.json") if p.is_file() and not _is_noise(p)])
    return [read_arc_task(p) for p in files]


# ------------------------- Helper -------------------------------

def to_np_grid(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError("ARC grid must be 2D")
    return arr


def exact_match(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def resolve_callable(dotted: str):
    mod_name, func_name = dotted.split(":", 1) if ":" in dotted else dotted.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, func_name)


# ------------------------- Eval -------------------------------

def eval_arc(
    solver_fn,
    tasks: List[ArcTask],
    categories: Optional[Dict[str, str]],
    ctx: AuditContext,
    max_seconds_per_task: float,
    attempts: int = 1,
    agg: str = "first",
) -> Dict[str, Any]:
    n_total = 0
    n_correct = 0
    per_cat: Dict[str, Tuple[int, int]] = {}  # cat -> (correct, total)

    for t in tasks:
        for i, te in enumerate(t.test):
            n_total += 1
            cat = categories.get(t.task_id, "uncategorized") if categories else "uncategorized"
            t0 = time.perf_counter()
            try:
                # Hint the solver about desired candidate count via env (best effort)
                if attempts and attempts > 1:
                    os.environ["VXOR_ARC_TOPN"] = str(int(attempts))
                pred_list = solver_fn({
                    "task_id": t.task_id,
                    "train": t.train,
                    "test": [te],
                })
                # Normalize predictions into list of candidates for this single test
                candidates: List[np.ndarray] = []
                if not isinstance(pred_list, (list, tuple)):
                    raise ValueError("solver_fn must return a list")
                if len(pred_list) == 0:
                    candidates = []
                elif len(pred_list) == 1:
                    first = pred_list[0]
                    # Allow nested candidates: [ [grid1, grid2, ...] ]
                    # Detect triple nesting to distinguish from a single 2D grid (list[list[int]]).
                    if (
                        isinstance(first, (list, tuple))
                        and len(first) > 0
                        and isinstance(first[0], (list, tuple))
                        and len(first[0]) > 0
                        and isinstance(first[0][0], (list, tuple))
                    ):
                        # list of candidate grids
                        candidates = [to_np_grid(x) for x in first]
                    else:
                        candidates = [to_np_grid(first)]
                else:
                    # Some solvers may return multiple candidates directly even when given one test
                    candidates = [to_np_grid(x) for x in pred_list]
                gold = to_np_grid(te["output"]) if "output" in te else None
            except Exception as ex:
                append_jsonl(ctx, "ERROR", reason="solver_exception", task=t.task_id, test_index=i, error=str(ex))
                raise
            dt_s = time.perf_counter() - t0
            # Aggregation policy
            ok = False
            if gold is not None:
                if agg in ("best_of_n", "any_correct"):
                    ok = any(exact_match(c, gold) for c in candidates)
                else:  # "first"
                    first_pred = candidates[0] if candidates else None
                    ok = first_pred is not None and exact_match(first_pred, gold)
            n_correct += int(ok)
            c_ok, c_tot = per_cat.get(cat, (0, 0))
            per_cat[cat] = (c_ok + int(ok), c_tot + 1)
            first_shape = list(candidates[0].shape) if candidates else None
            append_jsonl(
                ctx,
                "ARC_TEST",
                task=t.task_id,
                test_idx=i,
                ok=ok,
                seconds=dt_s,
                category=cat,
                pred_shape=first_shape,
                gold_shape=list(gold.shape) if gold is not None else None,
                candidate_count=len(candidates),
                agg=agg,
            )
    acc = n_correct / n_total if n_total else 0.0
    cat_metrics = {k: {"acc": (v[0] / v[1] if v[1] else 0.0), "n": v[1]} for k, v in per_cat.items()}
    return {"accuracy": acc, "categories": cat_metrics, "n_total": n_total, "n_correct": n_correct}


# ------------------------- Main -------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="vXor ARC-AGI Benchmark Harness (deterministisch, auditiert)")
    p.add_argument("--arc-train", type=str, required=True, help="Pfad zum ARC-Train-Verzeichnis (JSON Dateien)")
    p.add_argument("--arc-test", type=str, required=True, help="Pfad zum ARC-Test-Verzeichnis (JSON Dateien)")
    p.add_argument("--solver", type=str, required=True, help="Dotted Callable: modul.pfad:funktion (nimmt task-dict und gibt Liste von Vorhersage-Grids)")
    p.add_argument("--allowlist", type=str, required=True, help="Pfad zur Allowlist (SHA256) für Dataset-Integrität")
    p.add_argument("--categories-json", type=str, default="", help="Optional: Mapping task_id->Kategorie (JSON)")
    p.add_argument("--output-dir", type=str, default="runs/arc", help="Ausgabeverzeichnis für Audits")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int)
    p.add_argument("--max-seconds", type=float, default=120.0, help="Zeitlimit pro Testfall (sek). Hinweis: Harness erfasst Zeit, enforced aber kein Abbruch.")
    p.add_argument("--attempts", type=int, default=1, help="Anzahl Kandidaten (falls Solver solche liefert)")
    p.add_argument("--agg", type=str, default="first", choices=["first", "best_of_n", "any_correct"], help="Aggregationsregel für Mehrfachkandidaten")

    args = p.parse_args(argv)

    # Determinismus / Policy
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
        seed=args.seed,
        host=os.uname().nodename,
        versions={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    )

    # Build-ID / SBOM
    script_text = Path(__file__).read_text(encoding="utf-8")
    sbom_txt = pip_freeze_all()
    build_id = compute_build_id(script_text, sbom_txt, os.environ.get("DOCKER_IMAGE_DIGEST", ""))
    append_jsonl(ctx, "RUN_START", build_id=build_id)

    # Dataset Allowlist
    allow = load_allowlist(Path(args.allowlist))
    train_dir = Path(args.arc_train)
    test_dir = Path(args.arc_test)
    if not train_dir.exists() or not test_dir.exists():
        append_jsonl(ctx, "ERROR", reason="dataset_missing", train=str(train_dir), test=str(test_dir))
        print("[ZTM] ARC Pfade existieren nicht", file=sys.stderr)
        return 2
    h_train = sha256_dir(train_dir)
    h_test = sha256_dir(test_dir)
    if not ({h_train, h_test} <= allow):
        append_jsonl(ctx, "ERROR", reason="allowlist_miss", train=h_train, test=h_test)
        print("[ZTM] Dataset Hash nicht in Allowlist", file=sys.stderr)
        return 2
    append_jsonl(ctx, "DATA_VERIFY_OK", train=h_train, test=h_test)

    # Kategorien (optional)
    categories = None
    if args.categories_json:
        cat_path = Path(args.categories_json)
        if not cat_path.exists():
            append_jsonl(ctx, "ERROR", reason="categories_missing", path=str(cat_path))
            return 2
        categories = json.loads(cat_path.read_text(encoding="utf-8"))

    # Solver laden
    try:
        solver_fn = resolve_callable(args.solver)
    except Exception as ex:
        append_jsonl(ctx, "ERROR", reason="solver_load_error", error=str(ex))
        print(f"[ZTM] Solver konnte nicht geladen werden: {ex}", file=sys.stderr)
        return 2

    # Daten laden
    train_tasks = load_arc_split(train_dir)
    test_tasks = load_arc_split(test_dir)
    append_jsonl(ctx, "ARC_SPLITS", n_train=len(train_tasks), n_test=len(test_tasks))

    # Evaluation (nur Testsplit; Train steht Solver zur Verfügung in den Aufgaben)
    try:
        metrics = eval_arc(
            solver_fn,
            test_tasks,
            categories,
            ctx,
            args.max_seconds,
            attempts=int(args.attempts),
            agg=str(args.agg),
        )
    except Exception as ex:
        append_jsonl(ctx, "ERROR", reason="uncaught_exception", error=str(ex))
        print(f"[ZTM][FATAL] {ex}", file=sys.stderr)
        return 2

    # Zusammenfassung schreiben
    summary = {
        "timestamp_utc": utc_now(),
        "build_id": build_id,
        "dataset": {
            "train_hash": h_train,
            "test_hash": h_test,
        },
        "results": metrics,
        "policy": {
            "threads": 1,
            "simulation": "disabled_by_policy",
        },
        "sbom": sbom_txt,
    }
    from bench.tools.audit import write_audit_summary
    write_audit_summary(ctx, summary)
    append_jsonl(ctx, "RUN_END", ok=True)
    print(f"[OK] ARC-Audit geschrieben nach {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
