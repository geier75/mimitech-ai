# vXor AGI Solver Adapters for ARC-AGI and GLUE

This document describes how to use the production vXor AGI inference adapters with the deterministic ARC/GLUE harnesses, including offline fallbacks, environment configuration, allowlist hashing, and example commands.

Both harnesses enforce ZTM policies (threads=1, dataset allowlists, audit logs). Network access is typically prohibited for fully hermetic runs. The adapters therefore default to an offline, deterministic fallback unless production endpoints are explicitly configured via environment variables. See policy notes below.

## Adapters

- ARC adapter: `vxor.benchmarks.solvers.vxor_arc_solver:solver`
- GLUE adapter: `vxor.benchmarks.solvers.vxor_glue_solver:solver`

These modules implement the solver signatures required by the harnesses:

- ARC: `solver(task_dict) -> List[pred_grid]`
- GLUE: `solver(task: str, batch: List[Dict[str, Any]]) -> List[pred]`
  - For classification tasks, `pred` are `int` labels
  - For STS-B, `pred` are `float` similarity scores

## Environment configuration

Set these variables to enable production inference. If unset (or `VXOR_FORCE_OFFLINE=1`), the adapters fall back to deterministic, offline baselines.

- `VXOR_API_BASE`: Base URL, e.g. `https://api.vxor.ai`
- `VXOR_API_KEY`: Bearer token for authentication
- `VXOR_API_TIMEOUT`: Request timeout in seconds (default: `10`)
- `VXOR_FORCE_OFFLINE`: If set to `1`, adapters never make network calls (default: `0`)

Current endpoints used (verify with vXor API docs):

- ARC: `POST ${VXOR_API_BASE}/api/arc/infer`
  - Request: `{ task_id, train: [{input, output}], test: [{input}] }`
  - Response: `{ predictions: [pred_grid, ...] }`
- GLUE: `POST ${VXOR_API_BASE}/api/glue/infer`
  - Request: `{ task: string, examples: [ {...}, ... ] }`
  - Response: `{ predictions: [int|float, ...] }`

Note: The exact endpoint paths/payloads may differ in your production. Adjust in `vxor/benchmarks/solvers/vxor_*_solver.py` if necessary.

## Dataset allowlists

Compute stable SHA-256 tree hashes and update allowlists before running:

```bash
# ARC
python -m bench.tools.hash_dir /path/to/arc/train > /tmp/arc-train.sha
python -m bench.tools.hash_dir /path/to/arc/test  > /tmp/arc-test.sha
cat /tmp/arc-*.sha >> bench/policies/arc.allow

# GLUE
python -m bench.tools.hash_dir /path/to/glue/root > /tmp/glue.sha
cat /tmp/glue.sha >> bench/policies/glue.allow
```

## Running ARC with vXor adapter

Offline fallback (policy-compliant, no network):

```bash
VXOR_FORCE_OFFLINE=1 \
python -m bench.arc_eval \
  --arc-train /path/to/arc/train \
  --arc-test /path/to/arc/test \
  --solver vxor.benchmarks.solvers.vxor_arc_solver:solver \
  --allowlist bench/policies/arc.allow \
  --output-dir runs/arc \
  --threads 1
```

With production inference (requires policy exception and network access):

```bash
VXOR_API_BASE="https://api.vxor.ai" \
VXOR_API_KEY="<SECRET>" \
python -m bench.arc_eval \
  --arc-train /path/to/arc/train \
  --arc-test /path/to/arc/test \
  --solver vxor.benchmarks.solvers.vxor_arc_solver:solver \
  --allowlist bench/policies/arc.allow \
  --output-dir runs/arc \
  --threads 1
```

## Running GLUE with vXor adapter

Offline fallback (policy-compliant, no network):

```bash
VXOR_FORCE_OFFLINE=1 \
python -m bench.glue_eval \
  --glue-root /path/to/glue \
  --tasks "cola,sst2,mrpc,stsb,qqp,mnli,qnli,rte,wnli" \
  --solver vxor.benchmarks.solvers.vxor_glue_solver:solver \
  --allowlist bench/policies/glue.allow \
  --split dev \
  --batch-size 32 \
  --threads 1 \
  --output-dir runs/glue
```

With production inference (requires policy exception and network access):

```bash
VXOR_API_BASE="https://api.vxor.ai" \
VXOR_API_KEY="<SECRET>" \
python -m bench.glue_eval \
  --glue-root /path/to/glue \
  --tasks "cola,sst2,mrpc,stsb,qqp,mnli,qnli,rte,wnli" \
  --solver vxor.benchmarks.solvers.vxor_glue_solver:solver \
  --allowlist bench/policies/glue.allow \
  --split dev \
  --batch-size 32 \
  --threads 1 \
  --output-dir runs/glue
```

## Auditing and outputs

Each run writes to `runs/<arc|glue>/<UTC>*/`:

- `audit.jsonl`: full event stream (`RUN_START`, `DATA_VERIFY_OK`, `ARC_TEST`/`GLUE_BATCH`, `RUN_END`)
- `audit.json`: build_id, SBOM, dataset hashes, metrics summary

Exit codes: `0` success; `2` fail-closed on policy/allowlist/solver errors.

## Policy notes

- Default behavior is offline fallback to honor hermetic ZTM policy.
- Setting `VXOR_API_BASE` and `VXOR_API_KEY` enables network calls from the solver. Ensure this is allowed in your environment and that secrets are provided via environment variables (not command-line or files committed to VCS).
- The adapters never log secrets. Any failures fall back deterministically and print a one-line diagnostic to stderr.
