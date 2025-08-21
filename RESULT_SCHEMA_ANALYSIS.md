# VXOR.AI Result Schema Inventur

## Gefundene Result-Typen

### 1. BenchResult (vxor_master_benchmark_suite.py) - AKTUELL
```python
@dataclass
class BenchResult:
    # Core Felder
    name: str                    # Benchmark-Name
    accuracy: float              # 0.0 to 1.0
    samples: int                 # Must be > 0
    dataset_paths: List[str]     # Pfade zu Datasets
    started_at: float            # Start-Timestamp
    finished_at: float           # End-Timestamp
    
    # Computed Properties (Aliases)
    @property
    def samples_processed(self) -> int        # Alias für samples
    def accuracy_score(self) -> float         # Alias für accuracy  
    def status(self) -> str                   # PASS/PARTIAL/ERROR
    def execution_time(self) -> float         # finished_at - started_at
    def benchmark_name(self) -> str           # Alias für name
    def authentic_data_source(self) -> str    # "MIT_AUTHENTIC_DATASETS"
```

### 2. BenchmarkResult (comprehensive_benchmark_suite.py) - LEGACY
```python
@dataclass
class BenchmarkResult:
    test_name: str                           # Test-Name
    category: str                            # Kategorie
    duration_ms: float                       # Ausführungszeit in ms
    throughput_ops_per_sec: float           # Durchsatz
    memory_usage_mb: float                   # Speicherverbrauch
    cpu_usage_percent: float                # CPU-Auslastung
    gpu_usage_percent: float                # GPU-Auslastung
    energy_consumption_watts: float         # Energieverbrauch
    success: bool                           # Erfolg/Fehler
    error_message: str                      # Fehlermeldung
    metadata: Dict[str, Any]                # Zusätzliche Metadaten
    timestamp: str                          # Zeitstempel als String
```

## Feldnamen-Synonyme & Varianten

### Namen-Feld
- `name` (BenchResult) ← KANONISCH
- `test_name` (BenchmarkResult)
- `benchmark_name` (Property)

### Status/Ergebnis-Feld
- `status` (computed) ← KANONISCH (PASS/PARTIAL/ERROR)
- `success` (BenchmarkResult) - boolean
- `error_message` (BenchmarkResult)

### Sample-Count-Feld
- `samples` (BenchResult) ← KANONISCH
- `samples_processed` (Property alias)
- Kein Äquivalent in BenchmarkResult

### Accuracy-Feld
- `accuracy` (BenchResult) ← KANONISCH (0.0-1.0)
- `accuracy_score` (Property alias)
- Kein Äquivalent in BenchmarkResult

### Zeitffelder
- `started_at`, `finished_at` (BenchResult) ← KANONISCH (float timestamps)
- `execution_time` (computed property) ← KANONISCH 
- `duration_ms` (BenchmarkResult) - Millisekunden
- `timestamp` (BenchmarkResult) - String

### Metadaten
- `dataset_paths` (BenchResult) ← KANONISCH
- `authentic_data_source` (computed)
- `metadata` (BenchmarkResult)
- `category` (BenchmarkResult)

### Performance-Metriken (nur BenchmarkResult)
- `throughput_ops_per_sec`
- `memory_usage_mb` 
- `cpu_usage_percent`
- `gpu_usage_percent`
- `energy_consumption_watts`

## Erkannte Probleme

1. **Inkonsistente Zeiteinheiten**: float seconds vs milliseconds vs string
2. **Verschiedene Status-Systeme**: PASS/PARTIAL/ERROR vs boolean success
3. **Fehlende Standardfelder**: BenchmarkResult hat keine accuracy/samples
4. **Doppelte Konzepte**: name vs test_name, execution_time vs duration_ms
5. **Unterschiedliche Granularität**: BenchResult für AI-Benchmarks, BenchmarkResult für Performance

## Empfohlenes Canonical Schema

**BenchResult** als Basis verwenden mit Erweiterungen für Legacy-Kompatibilität.
