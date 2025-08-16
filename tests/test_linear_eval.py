import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "quantum_linear_eval.py"


def make_npz(tmpdir: Path, n: int = 12, seed: int = 0, badly_conditioned: bool = False) -> tuple[Path, str, str]:
    rng = np.random.default_rng(seed)
    # Build orthonormal Q via QR
    M = rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(M)
    # Eigenvalues
    if badly_conditioned:
        vals = np.geomspace(1e-14, 1.0, num=n)
    else:
        vals = np.geomspace(1.0, 3.0, num=n)
    D = np.diag(vals)
    A = (Q.T @ D @ Q).astype(np.float64)
    # Ensure symmetry numerically
    A = 0.5 * (A + A.T)
    # Make sure it's strictly PD
    A += np.eye(n) * 1e-12
    x_true = rng.normal(size=(n,)).astype(np.float64)
    b = (A @ x_true).astype(np.float64)
    p = tmpdir / "data.npz"
    np.savez(p, A=A, b=b)
    from hashlib import sha256

    def h(arr: np.ndarray) -> str:
        s = sha256()
        s.update(str(arr.shape).encode())
        s.update(str(arr.dtype).encode())
        s.update(arr.tobytes(order="C"))
        return s.hexdigest()

    return p, h(A), h(b)


def write_allowlist(tmpdir: Path, hashes: list[str]) -> Path:
    p = tmpdir / "allow.txt"
    p.write_text("\n".join(hashes) + "\n", encoding="utf-8")
    return p


def run_script(npz: Path, allow: Path, **kwargs) -> tuple[int, Path, dict]:
    outdir = kwargs.pop("outdir", None) or (npz.parent / "runs")
    cmd = [sys.executable, str(SCRIPT), "--real165-npz", str(npz), "--allowlist", str(allow), "--threads", "1", "--output-dir", str(outdir)]
    for k, v in kwargs.items():
        flag = f"--{k.replace('_', '-')}"
        cmd += [flag, str(v)]
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    proc = subprocess.run(cmd, cwd=str(npz.parent), capture_output=True, text=True, env=env)
    # Find latest run dir
    runs = sorted((outdir).glob("*/audit.json"))
    audit_json = {}
    if runs:
        audit_json = json.loads(runs[-1].read_text(encoding="utf-8"))
    return proc.returncode, outdir, audit_json


def test_determinism(tmp_path: Path):
    npz, hA, hb = make_npz(tmp_path, n=12, seed=0)
    allow = write_allowlist(tmp_path, [hA, hb])
    code1, out1, a1 = run_script(npz, allow, tol_forward=1e-10, tol_backward=1e-12, cond_cap=1e12)
    code2, out2, a2 = run_script(npz, allow, tol_forward=1e-10, tol_backward=1e-12, cond_cap=1e12)
    assert code1 == 0 and code2 == 0
    assert a1["results"]["full"]["x_hash"] == a2["results"]["full"]["x_hash"]
    assert a1["results"]["ten"]["x_hash"] == a2["results"]["ten"]["x_hash"]


def test_cross_solver_and_gates_ok(tmp_path: Path):
    npz, hA, hb = make_npz(tmp_path, n=12, seed=1)
    allow = write_allowlist(tmp_path, [hA, hb])
    code, outdir, audit = run_script(npz, allow)
    assert code == 0
    assert "results" in audit and "full" in audit["results"]


def test_residual_gate_fail(tmp_path: Path):
    npz, hA, hb = make_npz(tmp_path, n=12, seed=2)
    allow = write_allowlist(tmp_path, [hA, hb])
    code, outdir, audit = run_script(npz, allow, tol_forward=1e-16)
    assert code != 0


def test_condition_gate_fail(tmp_path: Path):
    npz, hA, hb = make_npz(tmp_path, n=12, seed=3, badly_conditioned=True)
    allow = write_allowlist(tmp_path, [hA, hb])
    # Force stricter cap to guarantee failure regardless of numerical tweaks
    code, outdir, audit = run_script(npz, allow, cond_cap=1e6)
    assert code != 0


def test_allowlist_required(tmp_path: Path):
    npz, hA, hb = make_npz(tmp_path, n=12, seed=4)
    # No allowlist passed -> should fail
    cmd = [sys.executable, str(SCRIPT), "--real165-npz", str(npz)]
    proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
    assert proc.returncode != 0


def test_audit_schema(tmp_path: Path):
    npz, hA, hb = make_npz(tmp_path, n=12, seed=5)
    allow = write_allowlist(tmp_path, [hA, hb])
    code, outdir, audit = run_script(npz, allow)
    assert code == 0
    # audit.json fields
    for key in ["timestamp_utc", "host", "versions", "build_id", "dataset", "results", "sbom"]:
        assert key in audit
