# ARC-AGI Benchmark ZTM Policy

This document specifies the Zero-Trust Mode (ZTM) constraints for the ARC-AGI benchmark harness `bench/arc_eval.py`.

- Determinism: threads must be set to 1. Any other value fails closed.
- Offline: no network access, no external downloads during evaluation.
- Dataset integrity: both `--arc-train` and `--arc-test` directories must hash to values present in `bench/policies/arc.allow` using a stable tree SHA-256.
- Audit: the harness writes a structured `audit.jsonl` stream and a final `audit.json` summary containing build id, SBOM, dataset hashes, and results.
- Fail-closed: any exception, solver load error, or policy violation terminates with a non-zero exit code and is logged in the audit.
- Simulation/quantum paths: disabled by policy.

See `bench/README_arc_glue.md` for usage and operational guidance.
