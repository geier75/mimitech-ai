# Linear Solve Verifier — Q‑LOGIK Hard Mode (v2)

Production‑grade verification tool for linear systems: deterministic, fail‑closed, auditable.

- Script: `bench/quantum_linear_eval.py`
- Policy: `bench/policies/linear_solve.ztm.md`
- Tests: `tests/test_linear_eval.py`

## Usage

Prereqs: Python 3.11, NumPy, SymPy (SciPy optional). Ensure dataset hashes are in the allowlist.

```
python bench/quantum_linear_eval.py \
  --real165-npz data/uk165_real.npz \
  --allowlist bench/policies/data.allow \
  --seed 1234 --repeats 3 --threads 1 \
  --tol-forward 1e-10 --tol-backward 1e-12 --cond-cap 1e12 \
  --output-dir runs/hard
```

- Determinism: threads fixed, `PYTHONHASHSEED=0`, `TZ=UTC`, float64, `np.seterr(all="raise")`.
- Gates enforced: condition (SVD), forward residual, backward error, cross‑solver (NumPy vs SymPy, SciPy if available), perturbation, scaling, permutation.
- Quantum path: disabled by policy.

## Outputs
- `runs/<ts>/audit.json` — results + environment + build_id + hashes.
- `runs/<ts>/audit.jsonl` — event log.
- `runs/<ts>/sbom.txt` — pip freeze snapshot.

## Exit Codes
- 0 on success; non‑zero on any violation. Errors are logged to JSONL.

## Numerical stability: einsum() and index-based permutation

To fix the previous "divide by zero encountered in matmul" issue and improve determinism across BLAS backends (notably on Apple Silicon), the verifier uses two changes in `bench/quantum_linear_eval.py`:

- Einsum residuals instead of matmul:

  ```python
  # bench/quantum_linear_eval.py
  def forward_residual_ok(A, b, x, tol):
      Ax = np.einsum('ij,j->i', A, x, optimize=False)
      r = Ax - b
      ...

  def backward_error(A, b, x):
      Ax = np.einsum('ij,j->i', A, x, optimize=False)
      r = Ax - b
      ...
  ```

  Using `np.einsum('ij,j->i', ...)` avoids dense `A @ x` matmul branches that previously led to numerical anomalies on some BLAS implementations, while remaining clear and deterministic.

- Index-based permutation (no permutation matrices):

  ```python
  # In block_checks()
  n = A.shape[0]
  perm = np.arange(n)[::-1]        # deterministic reverse permutation
  Ap = A[perm][:, perm]
  bp = b[perm]
  xp = np.linalg.solve(Ap, bp)
  x_back = np.empty_like(xp)
  x_back[perm] = xp                # invert permutation by indexing
  assert np.allclose(x, x_back, rtol=1e-10, atol=1e-13)
  ```

  This removes dense permutation matrices and avoids extra matmul, improving stability and performance.

### Determinism checks and CI

- The harness supports repeated runs with strict hash equality enforcement via `--repeats N`. On any mismatch, the run fails closed.
- Example CI invocation:

  ```bash
  python bench/quantum_linear_eval.py \
    --real165-npz data/uk165_real.npz \
    --allowlist bench/policies/data.allow \
    --threads 1 --repeats 5 \
    --tol-forward 1e-10 --tol-backward 1e-12 --cond-cap 1e12 \
    --output-dir runs/hard
  ```

- Pytest regression tests validate these behaviors:

  ```bash
  pytest -q tests/test_linear_eval_regression.py
  ```

  Coverage includes: einsum-based residuals, no matmul on permutation path, and identical `x_hash` across repeats (`BLOCK_DONE` events).

## Tests

Install pytest and deps, then run:
```
pytest -q tests/test_linear_eval.py
```

The tests validate determinism, gates, allowlist enforcement, and audit schema.
