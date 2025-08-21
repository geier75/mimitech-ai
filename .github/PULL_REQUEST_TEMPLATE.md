# MISO Pull Request - Quality Gates Checklist

**Please complete ALL applicable sections before requesting review. PRs with incomplete checklists will be rejected.**

## 📋 Core Quality Gates (Required)

### Phase 1-2: Schema & Validation
- [ ] **Schema Compliance**: All benchmark results validate against JSON schemas
- [ ] **Contract Adherence**: Uses only canonical contract fields (name, accuracy, samples_processed, duration_s, status, schema_version)
- [ ] **Schema Version**: Correct `$schema` reference in all JSON outputs
- [ ] **Value Ranges**: Accuracy 0-100%, duration_s > 0, status ∈ {PASS, PARTIAL, ERROR}

### Phase 3: Reproducibility
- [ ] **Reproducibility Block**: Complete repro-block in all reports (git_commit, git_tag, python_version, platform, env_flags, seed, compute_mode)
- [ ] **Deterministic Seeds**: Seeds set consistently (random/NumPy/Torch), PYTHONHASHSEED=0
- [ ] **Back-to-back Verification**: Two identical runs produce same results (excluding timestamps/commits)

### Phase 4: Dataset Integrity  
- [ ] **Minimum Samples**: MMLU≥14k, HellaSwag≥800, WinoGrande≥800, PIQA≥800, ARC≥1k
- [ ] **Checksum Validation**: All dataset manifest.sha256 files updated and verified
- [ ] **Provenance Documentation**: Dataset sources and licenses documented in DATA_POLICIES.md

### Phase 5: Structured Logging
- [ ] **JSONL Compatibility**: Structured logs are valid JSONL and parseable
- [ ] **Cross-checks**: predictions_count == samples_processed verified
- [ ] **Logger Namespaces**: Each benchmark uses dedicated logger namespace

## 🔧 CI/CD Quality Gates (Auto-verified)

### Phase 6: Hard Gates
- [ ] **Smoke Tests**: Smoke workflow passes with schema validation
- [ ] **Full Suite**: 12/12 benchmarks PASS, all minimum samples met
- [ ] **Artifact Upload**: Reports, logs, and summaries generated successfully

### Phase 7: Compute & Plausibility
- [ ] **Compute Mode**: metrics.compute_mode ∈ {full, light, stub} declared
- [ ] **Plausibility Checks**: No critical outliers in duration/throughput/accuracy
- [ ] **Performance Bounds**: Results within expected plausibility windows

### Phase 8: Schema Versioning
- [ ] **Version Consistency**: schema_version matches expected format
- [ ] **Migration Documentation**: Schema changes include migration guide (if applicable)
- [ ] **Breaking Changes**: Major schema bumps include updated ADR and CHANGELOG

### Phase 9-10: Reporting & Robustness  
- [ ] **Summary Generation**: SUMMARY.md generated successfully
- [ ] **Mutation Tests**: All mutation tests pass, schema validation robust
- [ ] **Documentation**: Getting Started updated (if applicable)

## 🛡️ Security & Governance (Phases 11-14)

### Phase 11: Drift Detection
- [ ] **Baseline Compatibility**: No critical drift against golden baseline (if exists)
- [ ] **Tolerance Compliance**: Results within established tolerance windows
- [ ] **Drift Documentation**: Acceptable drift documented and justified

### Phase 12: Supply Chain Security
- [ ] **SBOM Generated**: Software Bill of Materials created and verified
- [ ] **Build Provenance**: SLSA-compliant provenance documentation
- [ ] **Artifact Signing**: All build artifacts cryptographically signed

### Phase 13: Data Governance  
- [ ] **License Compliance**: Dataset usage complies with documented licenses
- [ ] **Access Control**: CI has read-only access, no write permissions to datasets/
- [ ] **Retention Policy**: Artifacts follow documented retention schedules

### Phase 14: Process Compliance
- [ ] **PR Template**: This template completed in full
- [ ] **Triage Classification**: Issue type classified (Schema/Dataset/Drift/Repro/Logs)
- [ ] **Guard Rails**: No critical quality gate failures

## 🎓 Training Promotion Checklist (If Applicable)

**Only complete this section for training-related PRs promoting a trained model.**

### T0-T2: Foundation & Preparation
- [ ] **Baseline-Ref**: UNTRAINED_BASELINE (Commit & Artefakte verlinkt)
- [ ] **Metrikvertrag bestätigt**: Benchmarks & Statistikmethoden (METRIC_CONTRACT.md)
- [ ] **Dedupe/Compliance grün**: T2 Artefakte vorhanden (DATA_CARD.md)

### T3-T5: Training & Evaluation
- [ ] **Pipeline-Sanity**: Mini-Overfit erfolgreich, Repro-Check bestanden
- [ ] **A/B-Eval identische Presets**: Seeds & Evaluation-Bedingungen wie T0
- [ ] **Signifikanz CI>0**: Bootstrap 95% CI Untergrenze > 0 für Ø-Accuracy-Δ

### Promotions-Matrix Compliance
- [ ] **Global**: Ø-Accuracy Δ ≥ +3 pp, 95%-CI Untergrenze > 0
- [ ] **Per-Benchmark**: Δ ≥ +2 pp oder kein Regress < −1 pp
- [ ] **Contamination-Suite**: Δ ≥ +1 pp, keine Verschlechterung
- [ ] **SWE-Bench**: ≥ +5 zusätzliche gelöste Tickets; Tests grün; Patch-Äquivalenz ok

### T6-T8: Robustness & Safety
- [ ] **Code-Tasks verifiziert**: Patch-Äquivalenz + Upstream-Test-Run als Gate
- [ ] **Contamination-Suite bestanden**: Kein Rückfall in kontaminationslimitierten Benchmarks
- [ ] **Safety-Smoke grün**: Red-Team-Checks, Jailbreak-Tests (Bericht angehängt)

### T9-T12: Release Readiness  
- [ ] **Supply-Chain**: SBOM, Provenance, Signaturen verifiziert (Training-Artefakte)
- [ ] **Drift-Report abgelegt**: Keine negativen Kern-Drifts vs UNTRAINED_BASELINE
- [ ] **Rollback-Test erfolgreich**: Checkpoint-Kompatibilität, Registry-Update
- [ ] **Go/No-Go dokumentiert**: Promotion-Protokoll mit Stakeholder-Approval

### Statistical Evidence Required
- [ ] **Bootstrap CI**: Mean ± 95% CI reported for all accuracy improvements
- [ ] **McNemar Test**: p < 0.05 für classification benchmarks (MMLU, ARC, etc.)
- [ ] **SWE-Bench Exact Count**: Resolved ticket count with test validation logs
- [ ] **Contamination Resistance**: Separate validation on clean/paraphrased datasets

## 📊 Change Classification

**Select the primary type of change:**
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)  
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Schema change (requires migration documentation)
- [ ] Dataset update (requires provenance documentation)
- [ ] Security update
- [ ] Infrastructure/CI change

## 🔍 Testing & Verification

### Local Testing
- [ ] `make test-short` passes locally
- [ ] `make test-all` passes locally (for significant changes)
- [ ] Manual smoke test performed
- [ ] Edge cases considered and tested

### CI Verification
- [ ] All CI checks pass ✅
- [ ] No new linting/formatting issues
- [ ] Schema validation successful
- [ ] Performance regression check (if applicable)

## 📝 Description

**Brief description of changes:**


**Motivation and context:**


**Dependencies (if any):**


## ⚠️ Risk Assessment

**Potential risks or side effects:**


**Security implications:**


**Performance impact:**


**Backward compatibility:**


## 🔗 Related Issues

Closes #
Relates to #

---

## 📞 Review Assignment

**Automatic Assignment Rules:**
- Schema changes → @schema-team
- Dataset changes → @data-team  
- CI/Pipeline → @devops-team
- Security → @security-team

**Required Approvals:** 2 maintainers for breaking changes, 1 for others

---

## ✅ Merge Readiness

- [ ] All quality gates passed
- [ ] Required approvals received
- [ ] CI pipeline successful
- [ ] Documentation updated
- [ ] Ready for merge

**Note**: PRs failing any required quality gate will be automatically blocked from merge.
