# GLUE Benchmark ZTM Policy

Zero-Trust Mode (ZTM) constraints for `bench/glue_eval.py`:

- Determinism: threads must be set to 1. Any other value fails closed.
- Offline: no network access or dynamic downloads at evaluation time.
- Dataset integrity: the GLUE root directory must hash to a value present in `bench/policies/glue.allow` using a stable tree SHA-256.
- Audit: the harness writes `audit.jsonl` and `audit.json` with build id, SBOM, dataset hashes, and per-task metrics.
- Fail-closed: any exception, solver load error, or policy violation terminates non-zero and is logged.
- Simulation/quantum: disabled by policy.

Operational guidance is in `bench/README_arc_glue.md`.
