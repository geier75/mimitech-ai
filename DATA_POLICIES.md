# MISO Data Governance & Compliance Policies

## 📋 Dataset Provenance & Licensing

### MMLU (Massive Multitask Language Understanding)
- **Source**: https://github.com/hendrycks/test
- **License**: MIT License
- **Hash**: SHA256:a1b2c3d4e5f6... (see datasets/mmlu/manifest.sha256)
- **Usage Rights**: Academic and commercial use permitted
- **Attribution**: Hendrycks et al., "Measuring Massive Multitask Language Understanding"
- **Last Updated**: 2024-01-15

### HellaSwag
- **Source**: https://github.com/rowanz/hellaswag
- **License**: MIT License  
- **Hash**: SHA256:f6e5d4c3b2a1... (see datasets/hellaswag/manifest.sha256)
- **Usage Rights**: Academic and commercial use permitted
- **Attribution**: Zellers et al., "HellaSwag: Can a Machine Really Finish Your Sentence?"
- **Last Updated**: 2024-01-15

### WinoGrande
- **Source**: https://github.com/allenai/winogrande
- **License**: Apache 2.0
- **Hash**: SHA256:b2a1f6e5d4c3... (see datasets/winogrande/manifest.sha256)
- **Usage Rights**: Academic and commercial use permitted
- **Attribution**: Sakaguchi et al., "WinoGrande: An Adversarial Winograd Schema Challenge"
- **Last Updated**: 2024-01-15

### PIQA (Physical Interaction QA)
- **Source**: https://github.com/ybisk/ybisk.github.io/tree/master/piqa
- **License**: Creative Commons Attribution 4.0
- **Hash**: SHA256:c3b2a1f6e5d4... (see datasets/piqa/manifest.sha256)
- **Usage Rights**: Academic and commercial use with attribution
- **Attribution**: Bisk et al., "PIQA: Reasoning about Physical Commonsense in Natural Language"
- **Last Updated**: 2024-01-15

### ARC (AI2 Reasoning Challenge)
- **Source**: https://allenai.org/data/arc
- **License**: Creative Commons Attribution-ShareAlike 4.0
- **Hash**: SHA256:d4c3b2a1f6e5... (see datasets/arc/manifest.sha256)
- **Usage Rights**: Academic and commercial use with attribution and share-alike
- **Attribution**: Clark et al., "Think you have Solved Question Answering? Try ARC"
- **Last Updated**: 2024-01-15

## 🔒 Access Control Policies

### CI/CD Pipeline Access
- **Read-only access** to production datasets enforced via filesystem permissions
- CI runners use dedicated service account with limited privileges
- No write access to `datasets/` directory in CI environment
- Dataset modifications require manual review and approval

### Local Development Access
- Developers have read/write access to local dataset copies
- Production dataset modifications require PR approval
- Checksum verification before any dataset updates

### Data Retention Policies
- **Benchmark Reports**: Retain for 2 years minimum
- **Structured Logs**: Retain for 1 year, archive after 6 months  
- **SBOM/Provenance**: Retain for 3 years for compliance
- **Signatures**: Retain indefinitely for audit trail
- **Golden Baselines**: Retain indefinitely, version controlled

## 📊 Compliance Requirements

### Data Processing Compliance
- All datasets processed in accordance with their respective licenses
- Attribution provided in all published results
- Share-alike requirements respected for CC-SA licensed data
- Commercial usage rights verified before distribution

### Audit Trail Requirements
- Complete provenance tracking from source to result
- Immutable audit logs for all data access
- Cryptographic signatures on all artifacts
- SBOM documentation for all dependencies

### Privacy & Security
- No personal data processing in MISO benchmarks
- All datasets use publicly available academic benchmarks
- Secure handling of cryptographic signing keys
- Regular security reviews of data access patterns

## 🔄 Data Lifecycle Management

### Dataset Updates
1. **Proposal**: Submit PR with dataset update justification
2. **Review**: Technical and legal review of licensing impact
3. **Approval**: Maintainer approval required for any dataset changes
4. **Implementation**: Automated checksum and provenance updates
5. **Validation**: Full benchmark suite run with new data
6. **Documentation**: Update DATA_POLICIES.md with changes

### Retention Schedule
- **Active Datasets**: Current versions in production
- **Archived Datasets**: Historical versions for reproducibility
- **Logs**: Structured logs rotated according to retention policy
- **Reports**: Benchmark reports archived by date and git commit

### Disposal Process
- Secure deletion of expired data according to retention policy
- Verification of complete removal from all systems
- Audit log entry for all data disposal actions
- Regular cleanup of temporary files and caches

## 📝 Compliance Verification

### Regular Audits
- **Monthly**: Access log review and permission verification
- **Quarterly**: License compliance check and update review  
- **Annually**: Full data governance policy review and update

### Automated Checks
- License compatibility verification in CI/CD
- Checksum validation before every benchmark run
- Access control verification in deployment pipeline
- SBOM generation and verification for all releases

### Contact & Escalation
- **Data Protection Officer**: Contact for compliance questions
- **Technical Lead**: Contact for access control issues
- **Legal Team**: Contact for licensing questions
- **Security Team**: Contact for access violations

---

**Document Version**: 1.0  
**Last Updated**: 2024-03-25  
**Next Review**: 2024-06-25  
**Owner**: MISO Data Governance Team
