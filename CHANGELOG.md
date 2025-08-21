# CHANGELOG - VXOR.AI Benchmark System

## [1.0.0] - 2025-08-17

### Added
- **Unified BenchResult Schema**: Canonical result structure with 6 core fields
- **Backward Compatibility**: 11 legacy properties for seamless migration
- **Schema Validation**: Hard gates preventing invalid benchmark results
- **Reproducible Benchmarks**: Monotonic timing and deterministic seeds
- **CI/CD Integration**: Automated pipeline with hard gates for data integrity
- **Comprehensive Documentation**: Migration guide, ADR, and field mapping tables

### Schema Contract
- `BenchResult` with mandatory fields: name, accuracy, samples_processed, dataset_paths, started_at, finished_at
- Status values: "PASS", "PARTIAL", "ERROR" with clear validation rules
- Accuracy stored as [0.0, 1.0], displayed as [0.0, 100.0]%
- Hard gates: samples_processed > 0, 0.0 ≤ accuracy ≤ 1.0, finished_at > started_at

### Data Integrity
- Minimum sample requirements: MMLU ≥14k, ARC ≥1k, HellaSwag/WinoGrande/PIQA ≥800
- Authentic dataset validation with MIT compliance markers
- JSON manifest generation with canonical field names
- Comprehensive test suite with 46k+ samples processed

### Performance & Reliability
- Monotonic clock timing (time.monotonic()) for accurate duration measurement
- PYTHONHASHSEED=0 for reproducible results
- Deterministic random seeds (seed=42) across all benchmarks
- 12/12 benchmark success rate in test suite

### Breaking Changes
- Legacy field access requires migration to canonical names or properties
- BenchmarkResult.success (bool) → BenchResult.status (str) 
- duration_ms → duration_s (available as computed property)
- Validation failures now raise ValueError instead of silent acceptance

### Migration Path
- All legacy properties available as computed fields
- Automatic conversion utilities provided
- Detailed migration guide in /MIGRATION_GUIDE_BENCHMARK_SCHEMA.md
- No immediate code changes required for consumers

## [0.x.x] - Previous Versions
Legacy benchmark implementations with inconsistent schemas (deprecated)
