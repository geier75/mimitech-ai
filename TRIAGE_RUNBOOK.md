# MISO Triage Runbook - Phase 14

**Systematic troubleshooting guide for MISO quality gate failures**

## 🚨 Emergency Response Matrix

| Failure Type | Severity | Response Time | Escalation |
|-------------|----------|---------------|------------|
| Schema Validation | Critical | Immediate | Block merge |
| Dataset Integrity | Critical | Immediate | Block merge |
| Security/Supply Chain | Critical | Immediate | Security team |
| Drift Detection | Warning | 1 hour | Review required |
| Performance Regression | Warning | 2 hours | Performance review |
| Documentation | Low | Next sprint | Documentation team |

## 🔍 Triage Decision Tree

```
Build Failure? → Start Here
├── Red CI Status?
│   ├── Schema Gate Failed? → Go to [Schema Issues](#schema-issues)
│   ├── Dataset Gate Failed? → Go to [Dataset Issues](#dataset-issues) 
│   ├── Benchmark Failed? → Go to [Benchmark Issues](#benchmark-issues)
│   ├── Drift Alert? → Go to [Drift Issues](#drift-issues)
│   ├── Security Alert? → Go to [Security Issues](#security-issues)
│   └── Other? → Go to [General Troubleshooting](#general-troubleshooting)
├── Performance Issues? → Go to [Performance Issues](#performance-issues)
├── Documentation Issues? → Go to [Documentation Issues](#documentation-issues)
└── Process Issues? → Go to [Process Issues](#process-issues)
```

---

## 🔧 Issue Resolution Guides

### Schema Issues

**Symptoms:**
- `❌ Schema validation failed`
- `Invalid JSON schema` errors
- `Required field missing` errors
- `Value out of range` errors

**Diagnosis Steps:**
1. **Identify the failing component:**
   ```bash
   # Check schema validation logs
   grep "schema.*fail\|validation.*error" logs/*.log
   
   # Validate specific file
   python -c "
   from miso.validation.schema_validator import SchemaValidator
   validator = SchemaValidator()
   validator.validate_benchmark_report_file('benchmark_report.json')
   "
   ```

2. **Check schema version compatibility:**
   ```bash
   # Verify schema version in report
   jq '.schema_version' benchmark_report.json
   
   # Check available schemas
   ls schemas/
   ```

3. **Validate field requirements:**
   ```bash
   # Check for required fields
   python scripts/validate_report_fields.py benchmark_report.json
   ```

**Common Fixes:**
- **Missing fields**: Add required fields (name, accuracy, samples_processed, duration_s, status, schema_version)
- **Wrong types**: Ensure accuracy is float 0-1, duration_s > 0, samples_processed > 0
- **Invalid status**: Must be "PASS", "PARTIAL", or "ERROR"
- **Schema version**: Update to current version or add migration

**Resolution Time:** < 30 minutes

---

### Dataset Issues

**Symptoms:**
- `❌ Dataset validation failed`
- `Minimum sample count not met`
- `Checksum mismatch detected`
- `Dataset not found`

**Diagnosis Steps:**
1. **Check dataset integrity:**
   ```bash
   # Verify checksums
   python scripts/validate_datasets.py
   
   # Check sample counts
   python -c "
   from miso.datasets.dataset_integrity import DatasetValidator
   validator = DatasetValidator()
   validator.validate_sample_counts()
   "
   ```

2. **Verify dataset permissions:**
   ```bash
   # Check read access
   ls -la datasets/*/
   
   # Verify CI permissions
   python scripts/setup_ci_permissions.py --verify-only
   ```

3. **Check manifest files:**
   ```bash
   # Verify all manifests exist
   find datasets/ -name "manifest.sha256" -exec wc -l {} +
   ```

**Common Fixes:**
- **Checksum mismatch**: Regenerate manifest with `python scripts/generate_checksums.py`
- **Low sample count**: Download full dataset or adjust minimum thresholds
- **Missing manifest**: Create with `scripts/create_dataset_manifest.py`
- **Permission denied**: Fix with `python scripts/setup_ci_permissions.py`

**Resolution Time:** < 1 hour

---

### Benchmark Issues

**Symptoms:**
- `❌ Benchmark execution failed`
- `Reproducibility check failed`
- `Cross-check mismatch`
- `Timeout or performance issues`

**Diagnosis Steps:**
1. **Check reproducibility:**
   ```bash
   # Verify seeds and environment
   python -c "
   import os
   print('PYTHONHASHSEED:', os.getenv('PYTHONHASHSEED'))
   print('OMP_NUM_THREADS:', os.getenv('OMP_NUM_THREADS'))
   print('MKL_NUM_THREADS:', os.getenv('MKL_NUM_THREADS'))
   "
   ```

2. **Validate cross-checks:**
   ```bash
   # Check structured logging
   python -c "
   from miso.logging.structured_logger import BenchmarkLogger
   logger = BenchmarkLogger('test')
   # Review JSONL logs for prediction/sample mismatches
   "
   ```

3. **Check compute mode:**
   ```bash
   # Verify compute mode declaration
   jq '.metrics.compute_mode' benchmark_report.json
   ```

**Common Fixes:**
- **Non-deterministic results**: Set seeds consistently, fix environment variables
- **Cross-check failure**: Ensure predictions_count == samples_processed
- **Missing compute mode**: Add to metrics.compute_mode field
- **Performance timeout**: Optimize implementation or increase timeout

**Resolution Time:** < 2 hours

---

### Drift Issues

**Symptoms:**
- `⚠️ Performance drift detected`
- `❌ Critical drift beyond tolerance`
- `Accuracy regression detected`
- `Baseline comparison failed`

**Diagnosis Steps:**
1. **Check drift report:**
   ```bash
   # Generate drift analysis
   python scripts/detect_drift.py -r benchmark_report.json
   
   # Review drift details
   python -c "
   from miso.baseline.drift_detector import DriftDetector
   detector = DriftDetector()
   # Analyze specific metrics
   "
   ```

2. **Compare with baseline:**
   ```bash
   # Check current baseline
   ls -la baseline/current/
   
   # Review baseline metrics
   cat baseline/current/baseline_metrics.json
   ```

3. **Validate tolerance windows:**
   ```bash
   # Check plausibility settings
   python -c "
   from miso.monitoring.plausibility_monitor import PlausibilityMonitor
   monitor = PlausibilityMonitor()
   monitor.get_benchmark_windows('mmlu')
   "
   ```

**Resolution Actions:**
- **Acceptable drift**: Document justification and update tolerance
- **Performance regression**: Investigate code changes, optimize implementation
- **Data drift**: Verify dataset integrity, check for upstream changes
- **New baseline needed**: Create new golden baseline after validation

**Resolution Time:** < 4 hours

---

### Security Issues

**Symptoms:**
- `❌ SBOM generation failed`
- `❌ Artifact signing failed`
- `❌ Provenance verification failed`
- `⚠️ Dependency vulnerability`

**Diagnosis Steps:**
1. **Check SBOM generation:**
   ```bash
   # Generate SBOM manually
   python scripts/generate_sbom.py --verify
   ```

2. **Verify artifact signatures:**
   ```bash
   # Check signature verification
   python scripts/sign_artifacts.py verify *.json
   ```

3. **Review dependencies:**
   ```bash
   # Check for known vulnerabilities
   pip audit
   
   # Verify requirements.txt
   pip check
   ```

**Resolution Actions:**
- **SBOM failure**: Update dependency information, fix pip environment
- **Signing failure**: Check signing keys, verify cosign/GPG setup
- **Vulnerability**: Update vulnerable dependency, assess impact
- **Provenance failure**: Verify build environment, check git information

**Escalation:** Security team immediately for critical vulnerabilities

**Resolution Time:** < 1 hour (non-critical), Immediate (critical)

---

### Performance Issues

**Symptoms:**
- Slow benchmark execution
- Memory usage spikes
- CI timeouts
- Resource exhaustion

**Diagnosis Steps:**
1. **Profile performance:**
   ```bash
   # Check system resources
   top -p $(pgrep python)
   
   # Profile memory usage
   python -m memory_profiler benchmark_script.py
   ```

2. **Review timing logs:**
   ```bash
   # Analyze duration patterns
   jq '.results[].duration_s' benchmark_report.json | sort -n
   ```

**Resolution Actions:**
- **Memory leaks**: Profile and fix memory usage
- **Slow execution**: Optimize algorithms, use appropriate compute mode
- **Resource limits**: Increase CI resource allocation
- **Timeout**: Optimize critical path or increase timeout limits

**Resolution Time:** < 4 hours

---

### Documentation Issues

**Symptoms:**
- Outdated documentation
- Missing migration guides
- Broken links
- Unclear procedures

**Resolution Actions:**
- Update relevant documentation files
- Add missing examples and guides
- Verify all links and references
- Review with documentation team

**Resolution Time:** < 1 day

---

### Process Issues

**Symptoms:**
- PR template incomplete
- Missing approvals
- Policy violations
- Workflow failures

**Resolution Actions:**
- Complete PR checklist items
- Request appropriate reviews
- Address policy compliance
- Fix workflow configuration

**Resolution Time:** < 2 hours

---

## 🚀 Quick Reference Commands

### Immediate Diagnostics
```bash
# Overall health check
make test-short

# Schema validation
python scripts/validate_all_schemas.py

# Dataset integrity
python scripts/validate_datasets.py

# Drift detection
python scripts/detect_drift.py -r benchmark_report.json --quiet

# CI permissions
python scripts/setup_ci_permissions.py --verify-only

# SBOM verification  
python scripts/generate_sbom.py --verify

# Full system check
make test-all
```

### Log Analysis
```bash
# Find recent errors
grep -i "error\|fail\|critical" logs/*.log | tail -20

# Check structured logs
jq . logs/*.jsonl | head -10

# Schema validation logs
grep "schema" logs/*.log

# Performance metrics
grep "duration_s\|throughput" logs/*.log
```

### Recovery Actions
```bash
# Reset to clean state
git clean -fdx
pip install -r requirements.txt

# Regenerate all manifests
python scripts/generate_all_manifests.py

# Create new baseline (if approved)
python scripts/create_golden_baseline.py -r benchmark_report.json

# Fix permissions
python scripts/setup_ci_permissions.py

# Regenerate compliance artifacts
python scripts/generate_sbom.py
python scripts/sign_artifacts.py sign *.json
```

---

## 📞 Escalation Contacts

| Issue Type | Contact | Response SLA |
|-----------|---------|-------------|
| Critical Security | @security-team | 15 minutes |
| Schema Breaking Changes | @schema-team | 1 hour |
| Dataset Issues | @data-team | 2 hours |
| CI/CD Pipeline | @devops-team | 1 hour |
| Performance Critical | @performance-team | 4 hours |
| General Issues | @maintainers | 8 hours |

---

## 📚 Additional Resources

- **Schema Documentation**: `schemas/README.md`
- **Dataset Policies**: `DATA_POLICIES.md`  
- **Migration Guides**: `SCHEMA_MIGRATION_GUIDE.md`
- **Security Policies**: `miso/security/README.md`
- **Performance Guidelines**: `docs/PERFORMANCE.md`
- **API Reference**: `docs/API.md`

---

**Last Updated**: 2024-03-25  
**Version**: 1.0  
**Owner**: MISO Reliability Team
