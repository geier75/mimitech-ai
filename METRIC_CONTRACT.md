# MISO Training Metric Contract
**Version**: 1.0  
**Status**: ACTIVE  
**Baseline Reference**: UNTRAINED_BASELINE  

## 🎯 Training Promotion Criteria

### Global Promotion Matrix
| Criterion | Threshold | Method | Gate |
|-----------|-----------|---------|------|
| **Average Accuracy Δ** | ≥ +3.0 pp | Bootstrap 95% CI | CI lower bound > 0 |
| **Per-Benchmark Δ** | ≥ +2.0 pp OR no regress < -1.0 pp | McNemar (classification) | Per-benchmark gates |
| **Contamination-Suite** | ≥ +1.0 pp AND no deterioration | Bootstrap CI | Must pass all |
| **SWE-Bench** | ≥ +5 additional solved tickets | Test-suite pass | All tests green |
| **Mandatory Gates** | Schema + Repro + Dedupe + Safety + Supply-Chain | Validation | All must be GREEN |

---

## 📊 Per-Benchmark Target Metrics

### **Mathematical Reasoning**
| Benchmark | Baseline | Target | Metric | Statistical Test |
|-----------|----------|--------|---------|------------------|
| **GSM8K** | 55-63% | ≥ 66% (+3 pp min) | Exact Match | Bootstrap CI |
| **ARC** | 58% | ≥ 61% (+3 pp min) | Accuracy | Bootstrap CI |

### **Language Understanding**  
| Benchmark | Baseline | Target | Metric | Statistical Test |
|-----------|----------|--------|---------|------------------|
| **MMLU** | 57-69% | ≥ 72% (+3 pp min) | Accuracy | Bootstrap CI |
| **HellaSwag** | 72% | ≥ 75% (+3 pp min) | Accuracy | Bootstrap CI |
| **WinoGrande** | 68% | ≥ 71% (+3 pp min) | Accuracy | Bootstrap CI |
| **PIQA** | 74% | ≥ 77% (+3 pp min) | Accuracy | Bootstrap CI |

### **Code Generation & Understanding**
| Benchmark | Baseline | Target | Metric | Statistical Test |
|-----------|----------|--------|---------|------------------|
| **HumanEval** | 48% | ≥ 55% (+7 pp min) | Pass@1 | Bootstrap CI |
| **MBPP** | 42% | ≥ 50% (+8 pp min) | Pass@1 | Bootstrap CI |
| **CodexGLUE** | 52% | ≥ 57% (+5 pp min) | Task-Averaged F1 | Bootstrap CI |
| **SWE-Bench Lite** | 18% (54/300) | ≥ 21% (63/300) | Resolved Issues | Exact test count |

---

## 🔬 Statistical Methodology

### **Bootstrap Confidence Intervals**
```python
# Sample size requirements
n_bootstrap = 10000
confidence_level = 0.95
min_sample_size = 100  # Per benchmark
```

**Implementation**: `scipy.stats.bootstrap()` with stratified resampling
**Reporting**: Mean Δ ± 95% CI, p-value < 0.05 required

### **McNemar Test (Classification)**
```python
# For paired predictions on same samples
from scipy.stats import mcnemar
# Null: P(baseline_correct, trained_wrong) == P(baseline_wrong, trained_correct)
```

**Use Cases**: MMLU, ARC, HellaSwag, WinoGrande, PIQA  
**Threshold**: p < 0.05 for improvement significance

### **SWE-Bench Exact Counting**
```python
# Resolved ticket counting with test validation
baseline_resolved = 54  # tickets with all tests passing
target_resolved = 63   # +9 additional resolved tickets
```

**Requirements**: 
- All upstream tests must pass
- Patch must be minimal and correct  
- No regression in previously solved tickets

---

## 🧪 Contamination-Resistant Evaluation Suite

### **Core Contamination Tests**
| Test Suite | Purpose | Sample Size | Pass Criteria |
|------------|---------|-------------|---------------|
| **Paraphrased MMLU** | Knowledge robustness | 1000 | No degradation vs original |
| **Adversarial GSM8K** | Reasoning robustness | 500 | ≥ +1pp improvement |
| **Code Variation Suite** | Code understanding | 300 | ≥ baseline performance |
| **Live Eval Proxy** | Fresh problem sets | 200 | Maintained performance |

### **Contamination Detection Protocol**
1. **N-gram Overlap**: < 10% 8-gram overlap with training data
2. **Semantic Similarity**: < 0.85 sentence-BERT cosine similarity  
3. **Manual Review**: 10% sample human verification
4. **Temporal Isolation**: Post-training-cutoff data only

---

## 📋 Promotion Gate Checklist

### **T0: Baseline Reference** ✓
- [ ] UNTRAINED_BASELINE frozen with commit hash
- [ ] Golden baseline artifacts stored: `baseline/untrained/`
- [ ] All evaluation seeds and presets documented
- [ ] Reproducibility block validated

### **T1: Metric Contract** ✓  
- [ ] This document reviewed and approved
- [ ] Statistical methods validated
- [ ] Contamination suite defined
- [ ] Promotion thresholds agreed

### **T2: Data Readiness**
- [ ] Train/Val/Test splits isolated
- [ ] Dedupe report: < 5% overlap with eval sets
- [ ] DATA_CARD.md completed with provenance
- [ ] License compliance verified

### **T3-T4: Training Pipeline**
- [ ] Mini-overfit successful (sanity check)  
- [ ] Reproducibility validated (same seed → same curves)
- [ ] Full training completed without divergence
- [ ] Checkpoints hashed and verified

### **T5: A/B Evaluation**
- [ ] Identical evaluation conditions vs T0
- [ ] Statistical significance achieved (Bootstrap CI > 0)
- [ ] Per-benchmark requirements met
- [ ] No critical regressions detected

### **T6: Code Tasks Validation**
- [ ] SWE-Bench: +5 additional resolved tickets
- [ ] All patches pass upstream tests
- [ ] Minimal diff policy enforced
- [ ] No regression in previously solved tickets

### **T7: Contamination Resistance**
- [ ] Contamination suite results ≥ baseline
- [ ] N-gram overlap < 10% verified
- [ ] Semantic similarity < 0.85 verified
- [ ] Manual review completed

### **T8: Safety & Guard-Rails**
- [ ] Red-team checks passed
- [ ] Jailbreak-smoke tests green  
- [ ] PII/TOS policy compliance verified
- [ ] Safety report filed

### **T9: Drift Detection**
- [ ] No negative drift in core benchmarks
- [ ] Item-level diff analysis completed
- [ ] Error profile analysis acceptable
- [ ] Drift report generated

### **T10: Supply Chain Security**
- [ ] Training SBOM generated and verified
- [ ] Checkpoint provenance documented
- [ ] All artifacts signed and verified
- [ ] SLSA build provenance complete

### **T11: Promotion Decision**
- [ ] All above gates GREEN
- [ ] Promotion protocol documented
- [ ] Go/No-Go decision recorded
- [ ] Stakeholder approvals obtained

### **T12: Rollback Readiness**
- [ ] SemVer registry updated
- [ ] Rollback procedure tested
- [ ] Checkpoint compatibility verified  
- [ ] Rollback documentation complete

---

## 🚨 Failure Response Protocol

### **Statistical Significance Failure**
- **Condition**: Bootstrap CI includes 0 OR McNemar p ≥ 0.05
- **Action**: Return to training (T3-T4) with adjusted hyperparameters
- **Documentation**: Record negative result and adjustment rationale

### **Per-Benchmark Regression**  
- **Condition**: Any benchmark drops > -1.0 pp
- **Action**: Investigate data contamination, evaluate training stability
- **Gate**: Must resolve before promotion consideration

### **Contamination Suite Failure**
- **Condition**: Performance drop in contamination-resistant tests
- **Action**: Review training data for leaks, adjust deduplication
- **Escalation**: Security team notification for potential data issues

### **Safety/Security Failure**
- **Condition**: Red-team findings OR supply chain verification failure  
- **Action**: IMMEDIATE HALT of promotion process
- **Escalation**: Security team + management notification

---

## 📚 Reference Documentation

- **Baseline Artifacts**: `baseline/untrained/`
- **Training Logs**: `training/logs/`  
- **Evaluation Reports**: `evaluation/reports/`
- **Statistical Scripts**: `scripts/statistical_analysis/`
- **Contamination Detection**: `scripts/contamination_detection/`

---

**Approval Required From:**
- [ ] Technical Lead (Statistical methodology)
- [ ] Data Team (Contamination & compliance)  
- [ ] Security Team (Safety & supply chain)
- [ ] Product Team (Success criteria)

**Last Updated**: 2024-03-25  
**Next Review**: Before first training run (T3)  
**Owner**: MISO Training Team
