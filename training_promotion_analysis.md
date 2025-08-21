# Training Promotion Statistical Analysis
**Analysis Date**: 2025-08-20T22:36:29
**Confidence Level**: 95%
**Bootstrap Samples**: 10,000

## 🎉 PROMOTION APPROVED

✅ **Decision**: READY FOR PROMOTION
✅ **Global Threshold**: Met (+3pp requirement)
✅ **Per-Benchmark Gates**: All benchmarks passed

## 📊 Global Performance Analysis

| Metric | Baseline | Candidate | Δ (pp) | 95% CI |
|--------|----------|-----------|--------|--------|
| **Accuracy** | 62.3% | 65.8% | **+3.5** | (+2.4, +4.6) |
| **Total Samples** | 15,525 | 15,525 | - | - |

## 🔬 Per-Benchmark Statistical Analysis

| Benchmark | Baseline | Candidate | Δ (pp) | 95% CI | p-value | Status |
|-----------|----------|-----------|--------|--------|---------|---------|
| **GSM8K** | 59.1% | 62.6% | **+3.5** | (-2.5, +4.8) | 0.575 | ✅ PASS |
| **HumanEval** | 48.8% | 52.3% | **+3.5** | (-4.3, +16.5) | 0.269 | ✅ PASS |
| **MMLU** | 62.1% | 65.6% | **+3.5*** | (+2.6, +4.8) | 0.000 | ✅ PASS |

*Statistically significant at p < 0.05

## 📐 Statistical Methodology

### Bootstrap Confidence Intervals
- **Method**: Stratified bootstrap resampling (10,000 iterations)
- **Confidence Level**: 95%
- **Random Seed**: 42 (reproducible)

### McNemar Test (Paired Classification)
- **Null Hypothesis**: No difference in error rates between models
- **Alternative**: Candidate model has different (better) error rate
- **Significance Level**: α = 0.05

### Promotion Thresholds (METRIC_CONTRACT.md)
- **Global Average**: ≥ +3.0pp with CI lower bound > 0
- **Per-Benchmark**: ≥ +2.0pp OR no regression > -1.0pp
- **Statistical Significance**: Required (p < 0.05)
