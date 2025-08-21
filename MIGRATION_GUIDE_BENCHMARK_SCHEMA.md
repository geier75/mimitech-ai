# VXOR.AI Benchmark Result Schema Migration Guide

## Übersicht

Dieses Dokument beschreibt die Migration von Legacy-Benchmark-Result-Schemas zum neuen kanonischen `BenchResult` Schema in VXOR.AI.

## Motivation

- **Einheitlichkeit**: Verschiedene Result-Klassen mit inkonsistenten Feldnamen
- **Wartbarkeit**: Schwierige Aggregation und Report-Generierung
- **Typsicherheit**: Fehlende Validierung und harte Schranken
- **Rückwärtskompatibilität**: Legacy-Code sollte ohne Änderungen funktionieren

## Canonical Schema: BenchResult

### Core Fields (Canonical)

```python
@dataclass
class BenchResult:
    name: str                    # Benchmark name
    accuracy: float              # Accuracy in [0.0, 1.0]
    samples_processed: int       # Number of samples processed
    dataset_paths: List[str]     # Paths to authentic datasets
    started_at: float           # Start timestamp (time.time())
    finished_at: float          # End timestamp (time.time())
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
```

### Computed Properties (Backward Compatibility)

```python
# Legacy aliases - automatically computed
@property
def samples(self) -> int:                    # → samples_processed
def accuracy_score(self) -> float:          # → accuracy  
def benchmark_name(self) -> str:            # → name
def test_name(self) -> str:                 # → name
def duration_s(self) -> float:              # finished_at - started_at
def execution_time(self) -> float:          # → duration_s
def duration_ms(self) -> float:             # duration_s * 1000
def status(self) -> str:                    # "PASS"/"PARTIAL"/"ERROR"
def success(self) -> bool:                  # status == "PASS"
def authentic_data_source(self) -> str:     # "MIT_AUTHENTIC_DATASETS"
```

## Migration Steps

### 1. Legacy BenchmarkResult → BenchResult

**Vorher (Legacy):**
```python
from comprehensive_benchmark_suite import BenchmarkResult

result = BenchmarkResult(
    test_name="MMLU",
    category="reasoning", 
    duration_ms=1500.0,
    success=True,
    error_message="",
    metadata={"samples": 1000, "accuracy": 0.85}
)
```

**Nachher (Canonical):**
```python
from tests.vxor_master_benchmark_suite import BenchResult

result = BenchResult(
    name="MMLU",
    accuracy=0.85,
    samples_processed=1000,
    dataset_paths=["/data/authentic/mmlu"],
    started_at=time.time(),
    finished_at=time.time() + 1.5
)
```

### 2. Conversion Function

```python
def convert_legacy_to_canonical(legacy_result: BenchmarkResult) -> BenchResult:
    """Convert legacy BenchmarkResult to canonical BenchResult"""
    
    # Extract data from metadata if present
    accuracy = legacy_result.metadata.get("accuracy", 0.0)
    samples = legacy_result.metadata.get("samples", 0)
    
    # Calculate timing
    duration_s = legacy_result.duration_ms / 1000.0
    now = time.time()
    
    return BenchResult(
        name=legacy_result.test_name,
        accuracy=float(accuracy),
        samples_processed=int(samples),
        dataset_paths=["/data/authentic"],  # Update with real paths
        started_at=now - duration_s,
        finished_at=now,
        metadata=legacy_result.metadata,
        error_message=legacy_result.error_message if not legacy_result.success else None
    )
```

### 3. Report Generation Migration

**Vorher:**
```python
def generate_report(results: List[BenchmarkResult]):
    return {
        "total_time": sum(r.duration_ms for r in results) / 1000,
        "success_rate": len([r for r in results if r.success]) / len(results),
        "benchmarks": [
            {
                "name": r.test_name,
                "status": "PASS" if r.success else "FAIL",
                "time": r.duration_ms
            } for r in results
        ]
    }
```

**Nachher:**
```python
def generate_report(results: List[BenchResult]):
    return {
        "total_time": sum(r.duration_s for r in results),
        "success_rate": len([r for r in results if r.status == "PASS"]) / len(results),
        "benchmarks": [
            {
                "name": r.name,
                "status": r.status,
                "accuracy": r.accuracy,
                "samples_processed": r.samples_processed,
                "duration_s": r.duration_s
            } for r in results
        ]
    }
```

### 4. Manifest Serialization Migration

**Vorher:**
```json
{
  "test_name": "MMLU",
  "duration_ms": 1500,
  "success": true,
  "samples": 1000
}
```

**Nachher:**
```json
{
  "benchmark_name": "MMLU",
  "execution_time": 1.5,
  "status": "PASS",
  "accuracy": 0.8500,
  "samples_processed": 1000,
  "dataset_paths": ["/data/authentic/mmlu"]
}
```

## Validation und Hard Gates

### Neue Validierung

```python
result = BenchResult(...)

# Automatische Validierung bei Zugriff auf status
assert result.status in ["PASS", "PARTIAL", "ERROR"]

# Explizite Validierung
try:
    result.validate()  # Raises ValueError if invalid
except ValueError as e:
    print(f"Invalid result: {e}")
```

### Hard Gates

- **samples_processed > 0**: Keine Simulation erlaubt
- **0.0 ≤ accuracy ≤ 1.0**: Gültige Accuracy-Werte
- **finished_at > started_at**: Positive Ausführungszeit
- **status ∈ ["PASS", "PARTIAL", "ERROR"]**: Gültige Status-Werte

## Breaking Changes

### ❌ Entfernte Fields

- `duration_ms` → Verwende `duration_s` oder `duration_ms` property
- `success` (bool) → Verwende `status` string oder `success` property
- `test_name` → Verwende `name`
- `category` → Verwende `metadata["category"]` falls nötig

### ✅ Rückwärtskompatible Properties

Alle Legacy-Feldnamen sind als Properties verfügbar:

```python
# Funktioniert weiterhin
result.samples              # → result.samples_processed
result.test_name            # → result.name
result.duration_ms          # → result.duration_s * 1000
result.success              # → result.status == "PASS"
```

## CI/CD Integration

### GitHub Actions Update

```yaml
- name: Run Benchmark Suite
  run: |
    python3 tests/vxor_master_benchmark_suite.py --preset short
    
- name: Validate Results
  run: |
    python3 -c "
    from tests.vxor_master_benchmark_suite import BenchResult
    # Results are automatically validated on creation
    "
```

## Beispiel: Complete Migration

```python
# Legacy producer
def run_legacy_benchmark():
    return BenchmarkResult(
        test_name="MMLU",
        category="reasoning",
        duration_ms=1500.0,
        success=True,
        error_message="",
        metadata={"samples": 1000, "accuracy": 0.85}
    )

# Canonical producer
def run_canonical_benchmark():
    t0 = time.time()
    # ... actual benchmark logic ...
    t1 = time.time()
    
    return BenchResult(
        name="MMLU",
        accuracy=0.85,
        samples_processed=1000,
        dataset_paths=["/data/authentic/mmlu"],
        started_at=t0,
        finished_at=t1
    )

# Migration wrapper
def migrate_legacy_producer():
    legacy_result = run_legacy_benchmark()
    return convert_legacy_to_canonical(legacy_result)
```

## Troubleshooting

### Häufige Fehler

1. **KeyError: 'samples'** → Verwende `'samples_processed'`
2. **AttributeError: 'BenchResult' object has no attribute 'duration_ms'** → Verwende `.duration_ms` property
3. **ValueError: samples_processed must be > 0** → Überprüfe Benchmark-Logik
4. **TypeError: unsupported operand type(s)** → Überprüfe Typen (int/float)

### Debugging

```python
# Debug result schema
print(f"Name: {result.name}")
print(f"Status: {result.status}")
print(f"Samples: {result.samples_processed}")
print(f"Accuracy: {result.accuracy:.2%}")
print(f"Duration: {result.duration_s:.2f}s")

# Validate result
try:
    result.validate()
    print("✅ Result is valid")
except ValueError as e:
    print(f"❌ Invalid result: {e}")
```

## Zusammenfassung

- **Canonical Schema**: `BenchResult` mit 6 core fields
- **Backward Compatibility**: Alle Legacy-Properties verfügbar
- **Validation**: Automatische Validierung und Hard Gates
- **Migration**: Schrittweise Migration möglich
- **CI/CD**: Unterstützung für authentische Daten und harte Schranken

## Schema Versioning Strategy (v1.0.0)

### Version Format
- **Pattern**: `v{major}.{minor}.{patch}` (z.B. `v1.0.0`)
- **Storage**: In `schemas/` directory als separate Dateien
- **Field**: Jeder Report enthält `"schema_version": "v1.0.0"`

### Version Evolution
```
v1.0.0 - Initial schema with basic validation
v1.1.0 - New optional fields (backward compatible)
v2.0.0 - Breaking changes (new migration required)
```

### Schema Files Structure
```
schemas/
├── bench_result.schema.json          # Individual result validation
├── benchmark_report.schema.json      # Complete report validation  
├── v1.1.0/                          # Future versions
│   ├── bench_result.schema.json
│   └── benchmark_report.schema.json
└── README.md                        # Schema documentation
```

### Migration Strategy
1. **Backward Compatibility**: Old schema versions remain available
2. **Gradual Migration**: Reports can specify their schema version
3. **Validation**: Validator automatically selects correct schema version
4. **CI Integration**: All schema versions tested in CI pipeline

**Next Steps:**
1. Migriere Benchmark-Producer zu `BenchResult`
2. Update Report-Generation auf canonical fields
3. Test alle Änderungen mit authentischen Daten
4. Update CI/CD Pipeline für neue Schema-Validierung
5. **Implement schema versioning in reports**
6. **Add CI schema validation for all generated reports**
