from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _best_threshold(records: List[Dict[str, Any]]) -> Tuple[float, float, int]:
    """Return (best_threshold, best_accuracy, n_used) for binary y_true and float y_score.
    Uses midpoints between sorted unique scores as candidate thresholds plus 0.0 and 1.0.
    """
    pairs: List[Tuple[float, int]] = []
    for r in records:
        y = r.get("y_true")
        s = r.get("y_score")
        if y in (0, 1):
            try:
                sf = float(s)
            except Exception:
                continue
            pairs.append((sf, int(y)))
    n = len(pairs)
    if n == 0:
        return 0.5, 0.0, 0

    pairs.sort(key=lambda t: t[0])
    scores = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]

    # Precompute prefix sums for positives
    pref_pos = [0]
    c = 0
    for y in labels:
        c += y
        pref_pos.append(c)

    def acc_at(thr: float) -> float:
        # pred = 1 if score >= thr else 0
        import bisect

        i = bisect.bisect_left(scores, thr)
        tp = pref_pos[-1] - pref_pos[i]  # scores >= thr and y=1
        fp = (len(scores) - i) - tp      # scores >= thr and y=0
        tn = i - pref_pos[i]             # scores < thr and y=0
        fn = pref_pos[i]                 # scores < thr and y=1
        return float((tp + tn) / n) if n else 0.0

    # Candidate thresholds: midpoints between distinct scores, plus ends
    candidates: List[float] = []
    if scores:
        candidates.append(scores[0] - 1e-9)
        for a, b in zip(scores, scores[1:]):
            if b > a:
                candidates.append((a + b) / 2.0)
        candidates.append(scores[-1] + 1e-9)
    else:
        candidates = [0.5]

    best_thr = 0.5
    best_acc = -1.0
    for thr in candidates:
        a = acc_at(thr)
        if a > best_acc:
            best_acc = a
            best_thr = thr
    return float(best_thr), float(best_acc), n


def run(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit per-task thresholds for binary GLUE tasks from preds dumps with y_score")
    ap.add_argument("--preds-dir", type=str, default="", help="Directory containing {task}.jsonl produced by --save-preds")
    ap.add_argument("--preds-file", action="append", default=[], help="Explicit preds JSONL file(s); can be given multiple times")
    ap.add_argument("--tasks", nargs="+", default=[
        "cola","sst2","mrpc","qqp","qnli","rte","wnli"
    ], help="Binary tasks to calibrate (must have y_score in dumps)")
    ap.add_argument("--out", type=str, default="thresholds.json", help="Output JSON mapping {task: threshold}")
    args = ap.parse_args(argv)

    files: Dict[str, Path] = {}
    if args.preds_dir:
        root = Path(args.preds_dir)
        for t in args.tasks:
            p = root / f"{t}.jsonl"
            if p.exists():
                files[t] = p
    for f in args.preds_file:
        p = Path(f)
        t = p.stem
        files[t] = p

    if not files:
        print("[calibrate] no prediction files found", flush=True)
        Path(args.out).write_text(json.dumps({}, indent=2), encoding="utf-8")
        return 0

    out: Dict[str, Any] = {}
    for task, path in sorted(files.items()):
        recs = list(_read_jsonl(path))
        best_thr, best_acc, n = _best_threshold(recs)
        out[task] = {
            "threshold": best_thr,
            "dev_accuracy": best_acc,
            "n_scored": n,
            "source": str(path),
        }
        print(f"[calibrate] {task}: thr={best_thr:.4f} acc={best_acc:.4f} n={n}")

    # Emit compact mapping for harness consumption: {task: threshold}
    mapping = {t: v["threshold"] for t, v in out.items() if v.get("n_scored", 0) > 0}
    Path(args.out).write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    # Also save full report alongside
    Path(args.out).with_suffix(".full.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[calibrate] thresholds written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
