# ADR: Unified Result Schema for VXOR.AI Benchmarks

## Status
**ACCEPTED** - Version 1.0 (2025-08-17)
**UPDATED** - Schema Validation Implementation (2025-08-17)

## Context
Multiple result classes exist across the VXOR.AI codebase with inconsistent field names, data types, and semantics. This leads to AttributeErrors and maintenance overhead.

## Decision
**BenchResult** wird als kanonisches Schema etabliert mit folgender Struktur:

### Pflichtfelder (Contract Requirements)
- `name: str` - Benchmark name (non-empty)
- `accuracy: float` - Accuracy in range [0.0, 1.0] 
- `samples_processed: int` - Must be > 0
- `dataset_paths: List[str]` - Non-empty list of dataset paths
- `started_at: float` - Timestamp (time.time())
- `finished_at: float` - Timestamp > started_at

### Erlaubte Status-Werte
- `"PASS"` - samples_processed > 0 AND 0.0 ≤ accuracy ≤ 1.0
- `"PARTIAL"` - samples_processed > 0 BUT accuracy outside valid range
- `"ERROR"` - samples_processed = 0 OR other validation failures

### Accuracy-Format
- **Interne Speicherung**: [0.0, 1.0] (float)
- **Report-Ausgabe**: [0.0, 100.0] % (rounded to 1 decimal)
- **JSON-Export**: [0.0, 1.0] (preserved precision)

### Core Fields (Required)
```python
@dataclass
class BenchResult:
    # Identity & Status
    name: str                    # Canonical benchmark name
    status: str                  # PASS | PARTIAL | ERROR
    
    # Performance Metrics  
    accuracy: float              # 0.0 to 1.0 (percentage / 100)
    samples_processed: int       # Number of samples processed (> 0)
    duration_s: float           # Execution time in seconds
    
    # Metadata
    dataset_paths: List[str]     # Paths to authentic datasets
    started_at: float           # Unix timestamp (start)
    finished_at: float          # Unix timestamp (end)
    
    # Optional Extensions
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
```

### Computed Properties (Backward Compatibility)
```python
@property
def execution_time(self) -> float:           # Alias for duration_s
def benchmark_name(self) -> str:             # Alias for name  
def samples(self) -> int:                    # Alias for samples_processed
def accuracy_score(self) -> float:           # Alias for accuracy
def authentic_data_source(self) -> str:      # Constant "MIT_AUTHENTIC_DATASETS"
```

## Field Contracts

### Status Field
- **Type**: `str` 
- **Values**: `"PASS"` | `"PARTIAL"` | `"ERROR"`
- **Logic**: 
  - `PASS`: samples_processed > 0 AND 0.0 ≤ accuracy ≤ 1.0
  - `PARTIAL`: samples_processed > 0 AND accuracy outside valid range  
  - `ERROR`: samples_processed ≤ 0

### Accuracy Field
- **Type**: `float`
- **Range**: `0.0` to `1.0` (NOT percentage)
- **Validation**: Values > 1.0 indicate data corruption

### Duration Field
- **Type**: `float` 
- **Unit**: Seconds (NOT milliseconds)
- **Precision**: 2 decimal places for display

### Samples Field
- **Type**: `int`
- **Constraint**: Must be > 0 for valid results
- **Hard Gate**: Zero samples = automatic ERROR status

## Rationale
1. **BenchResult** already in use and working in test suite
2. Minimal breaking changes through alias properties
3. Clear validation rules prevent false positives
4. Consistent time units across all benchmarks
5. MIT-compliant authentic data tracking

## Schema Validation Implementation

### Schema Storage
- **Location**: `schemas/` (Top-Level Directory)
- **Files**: 
  - `schemas/bench_result.schema.json` - Individual test result validation
  - `schemas/benchmark_report.schema.json` - Comprehensive report validation
- **Versioning**: Semantic versioning in `$id` field (e.g., `v1.0.0`)

### Git Integration
- Schema files are **versioniert** (Exception in `.gitignore`: `!schemas/*.json`)
- Reports bleiben weiterhin ignoriert (`*.json` rule applies to other locations)
- Schema changes require explicit commits and PR reviews

### Validation Strategy
- **Report Generation**: Validate against `benchmark_report.schema.json` before write
- **CI Pipeline**: Post-run validation of generated reports in `tests/reports/`
- **Test Coverage**: Both positive (valid) and negative (invalid) test cases

### Schema Versioning
- **schema_version**: Required field in all reports (`v1.0.0` pattern)
- **Migration**: New schema versions as separate files, old versions preserved
- **Compatibility**: Migration guides in `MIGRATION_GUIDE_BENCHMARK_SCHEMA.md`

### Data Integrity Gates
- **Accuracy**: Range [0-100] for reports, [0.0-1.0] internally
- **Status Values**: `passed`, `failed`, `skipped`, `error` only
- **Consistency**: `total_tests` = `passed` + `failed` + `skipped` + `errors`
- **Samples**: All `samples_processed` values must be ≥ 0

## Consequences
- Legacy `BenchmarkResult` deprecated but supported via conversion
- All producers must return canonical `BenchResult` objects
- Aggregation functions operate on homogeneous result lists
- JSON serialization standardized on canonical field names
- **Schema violations block CI** - ensures data quality at pipeline level
- Schema evolution tracked through semantic versioning
