# Q-LOGIK / ZTM Policy — Linear Solve Verifier (Hard Mode v2)

This policy governs the production-grade verification tool at `bench/quantum_linear_eval.py`.

## Principles
- Zero-Simulation: Only attested real datasets are allowed. No random/simulated A, b.
- Fail-Closed: Any violation aborts immediately (non-zero exit) and is logged to JSONL.
- Determinism: Hermetic runtime, fixed threads, float64, reproducible across machines.
- Proven Numerics: Multi-stage gates (forward/backward error, condition, metamorphic), cross-solver proofs, perturbation invariance.

## Required Inputs
- `--real165-npz PATH`: NPZ with keys `A`, `b` (float64-compatible). Square A, len(b)=n.
- `--allowlist PATH`: Text file of allowed SHA-256 hashes (one per line). Both `sha256(A)` and `sha256(b)` must be listed.

## Determinism & Hermeticity
- Environment pins before import: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `PYTHONHASHSEED=0`, `TZ=UTC`.
- CLI: `--threads N` (default 1) pins thread env vars.
- `np.seterr(all="raise")` with ZTM policy.
- All arithmetic in float64.

## Gates (defaults)
- Condition number (2-norm via SVD): `cond_2(A) ≤ 1e12` (override via `--cond-cap`).
- Forward residual: `‖A x − b‖ ≤ T_fwd (‖b‖ + ‖A‖ ‖x‖)` with `T_fwd = 1e-10` (`--tol-forward`).
- Backward error: `β = ‖r‖ / (‖A‖‖x‖ + ‖b‖) ≤ 1e-12` (`--tol-backward`).
- Cross-solver consistency (10×10 and full n×n): NumPy vs SymPy (and SciPy if available): `allclose(rtol=1e-10, atol=1e-13)`.
- Perturbation invariance: `A' = A + ε·sign(A)` with `ε=1e-12` must satisfy gates within `10×` tolerances.
- Metamorphic tests:
  - Scaling invariance: `(αA, αb)` yields same solution `x`.
  - Permutation invariance: Reverse-permute rows/cols and back-permute solution must match.

## Audit & Attestation
- JSONL events: `RUN_START`, `DATA_VERIFY_OK`, `QCHECK_*`, `METAMORPHIC_*`, `ERROR`, `RUN_END`.
- Hashes: SHA-256 of raw `A`, `b`, and computed solutions `x` per block.
- SBOM: `pip freeze --all` stored as `sbom.txt` and embedded in `audit.json`.
- Build ID: `build_id = SHA256(script_checksum + sbom_hash + DOCKER_IMAGE_DIGEST)` included in events and `audit.json`.

## Outputs
- `runs/<timestamp>/audit.json` — summary, environment matrix, dataset hashes, results, build_id.
- `runs/<timestamp>/audit.jsonl` — append-only event log.
- `runs/<timestamp>/sbom.txt` — dependency snapshot.

## Exit Code Policy
- Any gate/policy violation → non-zero exit. No silent fallbacks.

## CLI Reference
```
python bench/quantum_linear_eval.py \
  --real165-npz data/uk165_real.npz \
  --allowlist bench/policies/data.allow \
  --seed 1234 --repeats 3 --threads 1 \
  --tol-forward 1e-10 --tol-backward 1e-12 --cond-cap 1e12 \
  --output-dir runs/hard
```

## Notes
- SciPy is optional; any issues are logged and do not bypass cross-solver gates where applicable.
- Quantum/NISQ path is disabled by policy until hardware attestation and tomography validation exist (`status: disabled_by_policy`).
