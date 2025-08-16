from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set deterministic environment defaults BEFORE importing numpy/scipy to ensure BLAS picks them up
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TZ", "UTC")

import numpy as np
import sympy as sp

# ------------------------- Helpers / Determinism -------------------------

def set_hermetic_env(threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TZ", "UTC")
    try:
        import time as _t
        if hasattr(_t, 'tzset'):
            _t.tzset()  # type: ignore
    except Exception:
        pass
    # Fail on FP anomalies
    np.seterr(all="raise")


def sha256_of_array(arr: np.ndarray) -> str:
    # Hash raw bytes + shape + dtype for collision resistance
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def sha256_of_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def utc_now() -> str:
    ts = dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")
    # Normalize '+00:00' to 'Z' for canonicalization
    if ts.endswith('+00:00'):
        ts = ts[:-6] + 'Z'
    return ts


def ensure_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise SystemExit(f"[ZTM] Non-finite values in {name} (NaN/Inf detected)")


def spectral_norm(A: np.ndarray) -> float:
    # Robust spectral norm via SVD
    s = np.linalg.svd(A, compute_uv=False)
    return float(s.max())


def cond_2(A: np.ndarray) -> float:
    s = np.linalg.svd(A, compute_uv=False)
    smin = float(s.min())
    smax = float(s.max())
    if smin == 0.0:
        return float("inf")
    return smax / smin


def forward_residual_ok(A: np.ndarray, b: np.ndarray, x: np.ndarray, tol: float) -> Tuple[bool, float, float, float, float]:
    Ax = np.einsum('ij,j->i', A, x, optimize=False)
    r = Ax - b
    nr = float(np.linalg.norm(r))
    nb = float(np.linalg.norm(b))
    nA = spectral_norm(A)
    nx = float(np.linalg.norm(x))
    bound = tol * (nb + nA * nx)
    return nr <= bound, nr, nb, nA, nx


def backward_error(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    Ax = np.einsum('ij,j->i', A, x, optimize=False)
    r = Ax - b
    nr = float(np.linalg.norm(r))
    nA = spectral_norm(A)
    nx = float(np.linalg.norm(x))
    nb = float(np.linalg.norm(b))
    denom = nA * nx + nb
    return float("inf") if denom == 0 else nr / denom


def np_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(A, b)


def sympy_lu_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    MA = sp.Matrix(A)
    Mb = sp.Matrix(b.reshape((-1, 1)))
    x = MA.LUsolve(Mb)
    return np.array(x, dtype=np.float64).reshape((-1,))


def scipy_spsolve_optional(A: np.ndarray, b: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        import scipy.sparse as sps
        import scipy.sparse.linalg as spla
        xs = spla.spsolve(sps.csr_matrix(A), b)
        return np.array(xs, dtype=np.float64).reshape((-1,)), None
    except Exception as ex:
        return None, str(ex)

# ------------------------- Audit JSONL -------------------------

@dataclasses.dataclass
class AuditContext:
    jsonl_path: Path
    seed: Optional[int]
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

# ------------------------- Allowlist -------------------------

def load_allowlist(p: Path) -> set[str]:
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(s.split()[0])
    return set(rows)

# ------------------------- Main flow -------------------------

def run(args: argparse.Namespace) -> int:
    set_hermetic_env(args.threads)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_dir = outdir / dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    audit_jsonl = run_dir / "audit.jsonl"
    ctx = AuditContext(
        jsonl_path=audit_jsonl,
        seed=args.seed,
        host=platform.node(),
        versions={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sympy": sp.__version__,
        },
    )

    # Build-ID components
    script_text = Path(__file__).read_text(encoding="utf-8")
    script_checksum = sha256_of_text(script_text)
    try:
        sbom_proc = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=False)
        sbom_txt = sbom_proc.stdout
        sbom_hash = sha256_of_text(sbom_txt)
    except Exception as _ex:
        sbom_txt = f"pip_freeze_error: {_ex}"
        sbom_hash = sha256_of_text(sbom_txt)
    docker_digest = os.environ.get("DOCKER_IMAGE_DIGEST", "")
    build_id = hashlib.sha256((script_checksum + sbom_hash + docker_digest).encode("utf-8")).hexdigest()

    append_jsonl(ctx, "RUN_START", script_checksum=script_checksum, build_id=build_id)

    # Enforce deterministic threads policy (Hard Mode: only 1 allowed)
    if int(args.threads) != 1:
        append_jsonl(ctx, "ERROR", reason="threads_policy_violation", requested=int(args.threads))
        raise SystemExit("[ZTM] threads != 1 disabled by policy for determinism")

    # Policy: dataset required, no simulation
    npz_path = Path(args.real165_npz)
    if not npz_path.exists():
        append_jsonl(ctx, "ERROR", reason="dataset_missing", path=str(npz_path))
        raise SystemExit("[ZTM] No dataset provided (--real165-npz required and must exist). Simulation is disabled by policy.")

    data = np.load(npz_path)
    try:
        A = np.array(data["A"], dtype=np.float64)
        b = np.array(data["b"], dtype=np.float64).reshape((-1,))
    except Exception as ex:
        append_jsonl(ctx, "ERROR", reason="dataset_invalid", error=str(ex))
        raise SystemExit(f"[ZTM] Invalid dataset: {ex}")

    if A.shape[0] != A.shape[1] or A.shape[0] != b.shape[0]:
        append_jsonl(ctx, "ERROR", reason="shape_mismatch", A_shape=str(A.shape), b_shape=str(b.shape))
        raise SystemExit("[ZTM] Dataset must be square A and matching b")

    # Finite checks
    if not np.all(np.isfinite(A)):
        append_jsonl(ctx, "ERROR", reason="non_finite_A")
        raise SystemExit("[ZTM] Non-finite values detected in A")
    if not np.all(np.isfinite(b)):
        append_jsonl(ctx, "ERROR", reason="non_finite_b")
        raise SystemExit("[ZTM] Non-finite values detected in b")

    # Allowlist verification (required)
    hA = sha256_of_array(A)
    hb = sha256_of_array(b)
    if not args.allowlist:
        append_jsonl(ctx, "ERROR", reason="allowlist_required")
        raise SystemExit("[ZTM] Allowlist is required (--allowlist path). Without a matching hash, run is forbidden.")
    allow_path = Path(args.allowlist)
    if not allow_path.exists():
        append_jsonl(ctx, "ERROR", reason="allowlist_missing", path=str(allow_path))
        raise SystemExit("[ZTM] Allowlist file not found")
    allow = load_allowlist(allow_path)
    allow_ok = {hA, hb} <= allow
    if not allow_ok:
        append_jsonl(ctx, "ERROR", reason="allowlist_miss", A=hA, b=hb)
        raise SystemExit(f"[ZTM] Dataset hash not in allowlist: A={hA}, b={hb}")
    append_jsonl(ctx, "DATA_VERIFY_OK", A=hA, b=hb)

    # Cross-solver, gates on 10x10 and full size
    n = A.shape[0]
    if n < 10:
        append_jsonl(ctx, "ERROR", reason="too_small", n=n)
        raise SystemExit("[ZTM] Dataset must be at least 10x10 to run gates")

    def block_checks(Ax: np.ndarray, bx: np.ndarray, label: str) -> Dict[str, Any]:
        try:
            # Condition gate
            c = cond_2(Ax)
            append_jsonl(ctx, "QCHECK_COND", label=label, cond2=c, cap=args.cond_cap, pass_=(c <= args.cond_cap))
            if not (c <= args.cond_cap):
                raise SystemExit(f"[ZTM] Condition number too high for {label}: {c:.3e} > {args.cond_cap}")

            # Solves
            x_np = np_solve(Ax, bx)
            x_sym = sympy_lu_solve(Ax, bx)
            x_sp_opt, sps_err = scipy_spsolve_optional(Ax, bx)

            # Cross-solver gate
            cross_ok = np.allclose(x_np, x_sym, rtol=1e-10, atol=1e-13)
            if not cross_ok:
                append_jsonl(ctx, "ERROR", reason="cross_solver_mismatch", label=label)
                raise SystemExit(f"[ZTM] Cross-solver mismatch (NumPy vs SymPy) on {label}")
            if x_sp_opt is not None:
                cross_ok2 = np.allclose(x_np, x_sp_opt, rtol=1e-10, atol=1e-13)
                append_jsonl(ctx, "QCHECK_CROSS_SCIPY", label=label, pass_=bool(cross_ok2))
                if not cross_ok2:
                    raise SystemExit(f"[ZTM] Cross-solver mismatch (NumPy vs SciPy) on {label}")
            else:
                append_jsonl(ctx, "QCHECK_CROSS_SCIPY", label=label, pass_=False, error=sps_err)

            # Forward residual gate
            f_ok, nr, nb, nA, nx = forward_residual_ok(Ax, bx, x_np, args.tol_forward)
            append_jsonl(ctx, "QCHECK_FORWARD", label=label, pass_=bool(f_ok), nr=nr, nb=nb, nA=nA, nx=nx, tol=args.tol_forward)
            if not f_ok:
                raise SystemExit(f"[ZTM] Forward residual too high on {label}: {nr:.3e} > bound")

            # Backward error gate
            be = backward_error(Ax, bx, x_np)
            be_ok = be <= args.tol_backward
            append_jsonl(ctx, "QCHECK_BACKWARD", label=label, pass_=bool(be_ok), beta=be, tol=args.tol_backward)
            if not be_ok:
                raise SystemExit(f"[ZTM] Backward error too high on {label}: {be:.3e}")

            # Perturbation invariance
            eps = 1e-12
            A2 = Ax + eps * np.sign(Ax)
            x2 = np_solve(A2, bx)
            f_ok2, nr2, *_ = forward_residual_ok(A2, bx, x2, 10.0 * args.tol_forward)
            be2 = backward_error(A2, bx, x2)
            be_ok2 = be2 <= 10.0 * args.tol_backward
            if not (f_ok2 and be_ok2):
                append_jsonl(ctx, "ERROR", reason="perturbation_invariance_fail", label=label, beta2=be2, nr2=nr2)
                raise SystemExit(f"[ZTM] Perturbation invariance failed on {label}")
            append_jsonl(ctx, "METAMORPHIC_PERTURB_OK", label=label, beta2=be2, nr2=nr2)

            # Scaling invariance
            alpha = 3.0
            A3 = alpha * Ax
            b3 = alpha * bx
            x3 = np_solve(A3, b3)
            if not np.allclose(x_np, x3, rtol=1e-10, atol=1e-13):
                append_jsonl(ctx, "ERROR", reason="scaling_invariance_fail", label=label)
                raise SystemExit(f"[ZTM] Scaling invariance failed on {label}")
            append_jsonl(ctx, "METAMORPHIC_SCALE_OK", label=label, alpha=alpha)

            # Permutation invariance (deterministic permutation: reverse)
            nloc = Ax.shape[0]
            perm = np.arange(nloc)[::-1]
            # Use index-based permutation to avoid dense permutation matrix and BLAS matmul
            Ap = Ax[perm][:, perm]
            bp = bx[perm]
            xp = np_solve(Ap, bp)
            x_back = np.empty_like(xp)
            x_back[perm] = xp
            if not np.allclose(x_np, x_back, rtol=1e-10, atol=1e-13):
                append_jsonl(ctx, "ERROR", reason="permutation_invariance_fail", label=label)
                raise SystemExit(f"[ZTM] Permutation invariance failed on {label}")
            append_jsonl(ctx, "METAMORPHIC_PERMUTE_OK", label=label)

            return {
                "cond2": c,
                "x_hash": sha256_of_array(x_np),
            }
        except Exception as ex:
            append_jsonl(ctx, "ERROR", reason="numeric_exception", label=label, error=str(ex))
            raise

    # Ten-block from top-left 10x10 (no randomness)
    A10 = A[:10, :10]
    b10 = b[:10]
    ref_xh = None
    res10 = None
    for r in range(int(args.repeats)):
        rres = block_checks(A10, b10, label="10x10")
        append_jsonl(ctx, "BLOCK_DONE", label="10x10", repeat=r, x_hash=rres["x_hash"]) 
        if r == 0:
            ref_xh = rres["x_hash"]
            res10 = rres
        else:
            if rres["x_hash"] != ref_xh:
                append_jsonl(ctx, "ERROR", reason="repeat_inconsistency", label="10x10", repeat=r)
                raise SystemExit("[ZTM] Non-deterministic solution across repeats (10x10)")

    # Full block (e.g., 165x165)
    ref_xhN = None
    resN = None
    for r in range(int(args.repeats)):
        rres = block_checks(A, b, label=f"{n}x{n}")
        append_jsonl(ctx, "BLOCK_DONE", label=f"{n}x{n}", repeat=r, x_hash=rres["x_hash"]) 
        if r == 0:
            ref_xhN = rres["x_hash"]
            resN = rres
        else:
            if rres["x_hash"] != ref_xhN:
                append_jsonl(ctx, "ERROR", reason="repeat_inconsistency", label=f"{n}x{n}", repeat=r)
                raise SystemExit("[ZTM] Non-deterministic solution across repeats (full)")

    # sbom_txt already computed above

    audit_json = {
        "timestamp_utc": utc_now(),
        "host": platform.node(),
        "versions": ctx.versions,
        "build_id": build_id,
        "config": {
            "seed": args.seed,
            "threads": args.threads,
            "repeats": int(args.repeats),
            "tol_forward": args.tol_forward,
            "tol_backward": args.tol_backward,
            "cond_cap": args.cond_cap,
        },
        "dataset": {
            "path": str(npz_path),
            "A_hash": hA,
            "b_hash": hb,
            "shape": list(A.shape),
        },
        "results": {
            "ten": res10,
            "full": resN,
        },
        "policy": {
            "quantum": "disabled_by_policy",
            "simulation": "disabled_by_policy",
        },
        "sbom": sbom_txt,
    }
    (run_dir / "audit.json").write_text(json.dumps(audit_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "sbom.txt").write_text(sbom_txt, encoding="utf-8")

    append_jsonl(ctx, "RUN_END", ok=True)
    print(f"[OK] Audit written to {run_dir}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Q-LOGIK Hard Mode Linear Solve Verifier")
    p.add_argument("--real165-npz", required=True, help="Path to NPZ with keys A,b (attested dataset)")
    p.add_argument("--allowlist", default="", help="Path to file with allowed SHA-256 hashes (one per line)")
    p.add_argument("--threads", type=int, default=1, help="Fix BLAS/threads for determinism")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--repeats", type=int, default=1, help="Repeat deterministic blocks and enforce identical results")
    p.add_argument("--tol-forward", type=float, default=1e-10)
    p.add_argument("--tol-backward", type=float, default=1e-12)
    p.add_argument("--cond-cap", type=float, default=1e12)
    p.add_argument("--output-dir", default="runs/hard")
    args = p.parse_args()

    try:
        code = run(args)
        raise SystemExit(code)
    except SystemExit:
        raise
    except Exception as ex:
        # Fail-closed with audit entry if possible
        try:
            outdir = Path(args.output_dir)
            run_dir = outdir / dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
            audit_jsonl = run_dir / "audit.jsonl"
            ctx = AuditContext(audit_jsonl, args.seed, platform.node(), {"python": sys.version.split()[0], "numpy": np.__version__, "sympy": sp.__version__})
            append_jsonl(ctx, "ERROR", reason="uncaught_exception", error=str(ex))
        except Exception:
            pass
        print(f"[ZTM][FATAL] {ex}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
