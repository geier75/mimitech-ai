# VXOR.AI Result Schema - Field Mapping Table

## Legacy → Canonical Mapping

### Field Name Mappings
| Legacy Field | Canonical Field | Conversion Rule |
|-------------|----------------|-----------------|
| `test_name` | `name` | Direct copy |
| `benchmark_name` | `name` | Direct copy |
| `success` | `status` | `True→"PASS"`, `False→"ERROR"` |
| `error_message` | `status` + `error_message` | `""→"PASS"`, `!""→"ERROR"` |
| `samples` | `samples_processed` | Direct copy |
| `duration_ms` | `duration_s` | `value / 1000.0` |
| `timestamp` | `started_at` | Parse string to float |
| `accuracy` | `accuracy` | Ensure 0.0-1.0 range |
| `accuracy_score` | `accuracy` | Direct copy |

### Status Field Conversion Rules
```python
def convert_status(legacy_result) -> str:
    # From boolean success field
    if hasattr(legacy_result, 'success'):
        if legacy_result.success and legacy_result.samples_processed > 0:
            return "PASS"
        elif legacy_result.samples_processed > 0:
            return "PARTIAL" 
        else:
            return "ERROR"
    
    # From error_message field
    if hasattr(legacy_result, 'error_message'):
        if not legacy_result.error_message and legacy_result.samples_processed > 0:
            return "PASS"
        else:
            return "ERROR"
    
    # Default logic
    if legacy_result.samples_processed > 0 and 0.0 <= legacy_result.accuracy <= 1.0:
        return "PASS"
    elif legacy_result.samples_processed > 0:
        return "PARTIAL"
    else:
        return "ERROR"
```

### Time Field Conversion
```python
def convert_time_fields(legacy_result) -> tuple[float, float]:
    # BenchmarkResult → BenchResult
    if hasattr(legacy_result, 'duration_ms'):
        duration_s = legacy_result.duration_ms / 1000.0
        # Approximate timestamps if missing
        now = time.time()
        finished_at = now
        started_at = now - duration_s
        return started_at, finished_at
    
    # String timestamp → float
    if hasattr(legacy_result, 'timestamp') and isinstance(legacy_result.timestamp, str):
        try:
            finished_at = time.mktime(time.strptime(legacy_result.timestamp, '%Y-%m-%d %H:%M:%S'))
            started_at = finished_at - legacy_result.duration_s
            return started_at, finished_at
        except ValueError:
            # Fallback to current time
            now = time.time()
            return now, now
    
    # Default: current time
    now = time.time()
    return now, now
```

### Accuracy Scaling Rules
```python
def normalize_accuracy(value: float) -> float:
    """Ensure accuracy is in 0.0-1.0 range"""
    if value > 1.0:
        # Assume percentage (0-100) → normalize to 0.0-1.0
        return min(value / 100.0, 1.0)
    return max(0.0, min(value, 1.0))
```

## Backward Compatibility Aliases

### Required Properties in BenchResult
```python
@property
def test_name(self) -> str:
    """Legacy alias for name"""
    return self.name

@property  
def benchmark_name(self) -> str:
    """Legacy alias for name"""
    return self.name

@property
def samples(self) -> int:
    """Legacy alias for samples_processed"""
    return self.samples_processed

@property
def accuracy_score(self) -> float:
    """Legacy alias for accuracy"""
    return self.accuracy

@property
def execution_time(self) -> float:
    """Legacy alias for duration_s"""
    return self.duration_s

@property
def duration_ms(self) -> float:
    """Legacy compatibility - seconds to milliseconds"""
    return self.duration_s * 1000.0

@property
def success(self) -> bool:
    """Legacy compatibility - status to boolean"""
    return self.status == "PASS"

@property
def authentic_data_source(self) -> str:
    """MIT compliance marker"""
    return "MIT_AUTHENTIC_DATASETS"
```

## Conversion Function Template
```python
def convert_legacy_result(legacy_result) -> BenchResult:
    """Convert any legacy result type to canonical BenchResult"""
    
    # Extract name
    name = getattr(legacy_result, 'name', 
                  getattr(legacy_result, 'test_name',
                         getattr(legacy_result, 'benchmark_name', 'unknown')))
    
    # Convert status
    status = convert_status(legacy_result)
    
    # Extract metrics
    accuracy = normalize_accuracy(getattr(legacy_result, 'accuracy', 0.0))
    samples_processed = getattr(legacy_result, 'samples_processed',
                               getattr(legacy_result, 'samples', 0))
    
    # Convert time fields
    started_at, finished_at = convert_time_fields(legacy_result)
    duration_s = finished_at - started_at
    
    # Extract dataset paths
    dataset_paths = getattr(legacy_result, 'dataset_paths', [])
    
    # Optional fields
    error_message = getattr(legacy_result, 'error_message', None)
    metadata = getattr(legacy_result, 'metadata', {})
    
    return BenchResult(
        name=name,
        status=status,
        accuracy=accuracy,
        samples_processed=samples_processed,
        duration_s=duration_s,
        dataset_paths=dataset_paths,
        started_at=started_at,
        finished_at=finished_at,
        metadata=metadata,
        error_message=error_message
    )
```

## Validation Rules

### Hard Gates (Must Pass)
1. `samples_processed > 0` for non-ERROR status
2. `0.0 <= accuracy <= 1.0` for PASS status  
3. `duration_s >= 0.0` always
4. `finished_at >= started_at` always
5. `status in ["PASS", "PARTIAL", "ERROR"]`

### Soft Warnings (Log but Allow)
1. `accuracy == 0.0` → potential issue
2. `duration_s == 0.0` → timing not measured
3. `dataset_paths == []` → authenticity concern
4. `error_message` present with `status == "PASS"`
