from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

import bench.quantum_linear_eval as qle


def _make_spd_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    M = rng.standard_normal((n, n))
    A = M @ M.T + np.eye(n) * 1.0
    return A.astype(np.float64, copy=False)


def _prepare_dataset(tmp_path: Path, n: int = 12) -> Dict[str, Path]:
    rng = np.random.default_rng(12345)
    A = _make_spd_matrix(n, rng)
    b = rng.standard_normal(n).astype(np.float64)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / "toy.npz"
    np.savez(npz_path, A=A, b=b)
    # Allowlist using the same hashing as the verifier
    allow = tmp_path / "allow.txt"
    hA = qle.sha256_of_array(A)
    hb = qle.sha256_of_array(b)
    allow.write_text(f"{hA}\n{hb}\n", encoding="utf-8")
    return {"npz": npz_path, "allow": allow}


def _run_once(tmp_path: Path, npz: Path, allow: Path, repeats: int = 1, monkeypatch=None) -> Path:
    out_dir = tmp_path / "runs"
    args = Namespace(
        real165_npz=str(npz),
        allowlist=str(allow),
        threads=1,
        seed=0,
        repeats=repeats,
        tol_forward=1e-10,
        tol_backward=1e-12,
        cond_cap=1e12,
        output_dir=str(out_dir),
    )
    code = qle.run(args)
    assert code == 0
    # pick newest run dir
    run_dirs: List[Path] = sorted(out_dir.iterdir())
    assert run_dirs, "no run directory created"
    return run_dirs[-1]


def test_permutation_invariance_avoids_matmul(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _prepare_dataset(tmp_path, n=12)

    # Any attempt to call np.matmul within the verifier should fail this test.
    def _no_matmul(*args, **kwargs):  # pragma: no cover - guard path
        raise AssertionError("np.matmul must not be called in permutation path")

    monkeypatch.setattr(qle.np, "matmul", _no_matmul, raising=True)

    run_dir = _run_once(tmp_path, paths["npz"], paths["allow"], repeats=1, monkeypatch=monkeypatch)
    # Ensure permutation gate actually ran and passed
    audit_jsonl = (run_dir / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert any("METAMORPHIC_PERMUTE_OK" in ln for ln in audit_jsonl)


def test_einsum_used_for_residuals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _prepare_dataset(tmp_path, n=12)
    called = {"einsum": 0}
    # capture original to avoid recursion when monkeypatching
    orig_einsum = np.einsum

    def _einsum_spy(*args, **kwargs):
        called["einsum"] += 1
        return orig_einsum(*args, **kwargs)

    monkeypatch.setattr(qle.np, "einsum", _einsum_spy, raising=True)

    run_dir = _run_once(tmp_path, paths["npz"], paths["allow"], repeats=1, monkeypatch=monkeypatch)
    assert called["einsum"] > 0, "np.einsum should be used in residual computations"

    # Ensure forward/backward gates were executed
    events = [json.loads(ln) for ln in (run_dir / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(ev.get("event") == "QCHECK_FORWARD" and ev.get("pass_") for ev in events)
    assert any(ev.get("event") == "QCHECK_BACKWARD" and ev.get("pass_") for ev in events)


def test_repeats_determinism_xhash_identical(tmp_path: Path) -> None:
    paths = _prepare_dataset(tmp_path, n=10)  # 10x10; "ten" and "full" both 10x10
    run_dir = _run_once(tmp_path, paths["npz"], paths["allow"], repeats=3)

    # Collect x_hash per repeat and label from audit.jsonl
    hashes: Dict[str, List[str]] = {}
    for ln in (run_dir / "audit.jsonl").read_text(encoding="utf-8").splitlines():
        ev = json.loads(ln)
        if ev.get("event") == "BLOCK_DONE":
            lbl = ev.get("label")
            xh = ev.get("x_hash")
            if lbl and xh:
                hashes.setdefault(lbl, []).append(xh)
    assert hashes, "no BLOCK_DONE events found"
    for lbl, arr in hashes.items():
        assert len(arr) >= 2
        assert all(x == arr[0] for x in arr), f"x_hash mismatch for label {lbl}: {arr}"
