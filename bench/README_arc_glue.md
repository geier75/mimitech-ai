# vXor ARC-AGI & GLUE Benchmark Harness

Deterministic, fail-closed, audit-logged benchmark runners:
- `bench/arc_eval.py` — ARC-AGI JSON tasks
- `bench/glue_eval.py` — GLUE tasks (JSONL splits)

Both enforce ZTM policies: threads=1, dataset allowlist, offline, audit logs.

## Quick Start

### 1) Compute dataset hashes and update allowlists

Use the helper to compute stable SHA-256 of directories:

```bash
python -m bench.tools.hash_dir /path/to/arc/train > /tmp/arc-train.sha
python -m bench.tools.hash_dir /path/to/arc/test  > /tmp/arc-test.sha
cat /tmp/arc-*.sha >> bench/policies/arc.allow

python -m bench.tools.hash_dir /path/to/glue/root > /tmp/glue.sha
cat /tmp/glue.sha >> bench/policies/glue.allow
```

### 2) Run ARC (with baseline solver)

```bash
python -m bench.arc_eval \
  --arc-train /path/to/arc/train \
  --arc-test /path/to/arc/test \
  --solver vxor.benchmarks.solvers.baselines:arc_shape_copy_solver \
  --allowlist bench/policies/arc.allow \
  --output-dir runs/arc \
  --threads 1
```

- Expects ARC JSON files under train/test directories.
- Optional: `--categories-json bench/arc_categories.example.json` for per-category metrics.

### 3) Run GLUE (with baseline solver)

```bash
python -m bench.glue_eval \
  --glue-root /path/to/glue \
  --tasks "cola,sst2,mrpc,stsb,qqp,mnli,qnli,rte,wnli" \
  --solver vxor.benchmarks.solvers.baselines:glue_majority_solver \
  --allowlist bench/policies/glue.allow \
  --split dev \
  --batch-size 32 \
  --threads 1 \
  --output-dir runs/glue
```

- Expects per-task subfolders containing `dev.jsonl` or `test.jsonl`.

## Scientist-Principle PR-A: Cajal & Scheibel

The PR-A patchset introduces deterministic micro-feature primitives for ARC and adapter/feature registries for ARC/GLUE.

- ARC Cajal primitives: `vxor.benchmarks.arc.primitives` (palette, symmetry, periodicity, connected components)
- ARC Scheibel registry: `vxor.benchmarks.arc.registry` with `@register_heuristic()`
- GLUE Cajal markers: `vxor.benchmarks.glue.features` (negation, modal, entailment cues)
- GLUE Scheibel adapter registry: `vxor.benchmarks.glue.adapter_registry` with `@register_adapter()`

See `bench/README_cajal_scheibel.md` for APIs and tests.

## Outputs

Each run creates `runs/<arc|glue>/<UTC>*/` with:
- `audit.jsonl` — stream of structured audit events
- `audit.json` — summary with build_id, SBOM, dataset hashes, and metrics

Exit codes: `0` success; `2` fail-closed for policy/allowlist/solver errors.

## Data Formats

### ARC

Each `*.json` file is an ARC task with `train` and `test`, e.g.:
```json
{
  "train": [{"input": [[1]], "output": [[1]]}],
  "test":  [{"input": [[1]]}]
}
```

### GLUE

Per-task directory e.g. `sst2/dev.jsonl` with one JSON per line:
```json
{"sentence": "a very good movie", "label": 1}
{"sentence": "bad", "label": 0}
```

STS-B (regression) example `stsb/dev.jsonl`:
```json
{"sentence1": "A", "sentence2": "A", "score": 5.0}
{"sentence1": "A", "sentence2": "B", "score": 2.0}
```

Task registry and metrics are defined in `bench/glue_eval.py::TASKS`.

## Policies

- See `bench/policies/arc.ztm.md` and `bench/policies/glue.ztm.md`.
- Threads must be `1`.
- No network or dynamic downloads at runtime.
- Dataset directories must hash to values present in corresponding `*.allow` files.

## Baseline Solvers

- ARC: `vxor.benchmarks.solvers.baselines:arc_shape_copy_solver`
- GLUE: `vxor.benchmarks.solvers.baselines:glue_majority_solver`

These are trivial, deterministic baselines to validate the harness and audits.

## vXor Production Adapters

Adapters that connect the harnesses to vXor AGI production inference are provided:

- ARC: `vxor.benchmarks.solvers.vxor_arc_solver:solver`
- GLUE: `vxor.benchmarks.solvers.vxor_glue_solver:solver`

Usage, environment variables, and example commands are documented in `bench/README_vxor_adapters.md`.
