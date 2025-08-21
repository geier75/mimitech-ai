# MISO Schema Changelog

This document tracks all changes to the MISO benchmark schema files.

## v1.0.0 - 2024-08-18

### Added
- Initial schema definitions for benchmark results and reports
- Reproducibility block with git commit, platform, and seed tracking
- Compute mode enforcement (full, light, stub)
- Minimum sample count validation requirements
- Cross-reference validation between summary and results

### Schema Files
- `bench_result.schema.json` - Individual benchmark test result validation
- `benchmark_report.schema.json` - Comprehensive benchmark report validation

### Requirements
- `schema_version` field is required in all reports
- `reproducibility` block is required with git_commit, python_version, platform, seed, compute_mode
- `compute_mode` must be one of: "full", "light", "stub"
- All timestamp fields must be in ISO 8601 format
- Accuracy values are stored as percentages (0-100)
- Sample counts must be non-negative integers

### Quality Gates
- Schema validation enforced in CI/CD pipeline
- Cross-checks ensure consistency between summary and detailed results
- Plausibility monitoring for duration, throughput, and accuracy metrics
- Minimum sample count enforcement per benchmark type

### Migration Notes
- This is the initial version - no migration required
- All new implementations must use this schema version
- Backward compatibility will be maintained for minor version increments

---

## Version Format

Schema versions follow semantic versioning:
- **Major** (v2.0.0): Breaking changes requiring migration
- **Minor** (v1.1.0): Backward-compatible additions
- **Patch** (v1.0.1): Bug fixes and clarifications

For breaking changes, a migration guide must be provided in `SCHEMA_MIGRATION_GUIDE.md`.

## Release Gate Requirements

1. **Minor Changes**: Update changelog
2. **Major Changes**: Update changelog + migration guide + version bump
3. **All Changes**: Schema validation tests must pass
4. **Breaking Changes**: Explicit approval from architecture team required

## Schema Validation

All schema files are validated for:
- JSON Schema Draft 7 compliance
- Consistent version identifiers in `$id` fields
- Required field completeness
- Semantic versioning in version strings
- Cross-reference consistency
