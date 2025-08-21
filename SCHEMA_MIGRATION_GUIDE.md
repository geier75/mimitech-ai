# MISO Schema Migration Guide

This guide provides instructions for migrating between different schema versions.

## Current Version: v1.0.0

This is the initial schema version. No migration is required for new implementations.

## Migration Instructions

### From v0.x to v1.0.0 (Initial Release)

This is the first official schema release. If you have been using pre-release schemas:

#### Required Changes:
1. **Add `schema_version` field**: All reports must include `"schema_version": "v1.0.0"`
2. **Add `reproducibility` block**: Required in all benchmark reports
3. **Update `compute_mode`**: Must be one of `"full"`, `"light"`, or `"stub"`
4. **Timestamp format**: Must be ISO 8601 with 'Z' suffix

#### Migration Script Example:
```python
def migrate_to_v1_0_0(old_report):
    """Migrate pre-v1.0.0 report to v1.0.0"""
    new_report = old_report.copy()
    
    # Add required schema version
    new_report['schema_version'] = 'v1.0.0'
    
    # Add reproducibility block if missing
    if 'reproducibility' not in new_report:
        from miso.reproducibility.repro_utils import ReproducibilityCollector
        collector = ReproducibilityCollector()
        new_report['reproducibility'] = collector.collect_all()
    
    # Ensure compute_mode is set
    if 'compute_mode' not in new_report.get('reproducibility', {}):
        new_report['reproducibility']['compute_mode'] = 'stub'
    
    # Update timestamp format
    if 'timestamp' in new_report:
        from datetime import datetime
        # Convert to ISO 8601 if needed
        new_report['timestamp'] = datetime.fromisoformat(
            new_report['timestamp'].replace('Z', '')
        ).isoformat() + 'Z'
    
    return new_report
```

## Future Migration Patterns

### Major Version Changes (Breaking)

When migrating across major versions (e.g., v1.x.x → v2.0.0):

1. **Read this migration guide** for specific breaking changes
2. **Update schema_version** in all reports
3. **Run validation** against new schema
4. **Test thoroughly** with new requirements
5. **Update CI/CD** to use new schema version

### Minor Version Changes (Non-Breaking)

Minor version updates (e.g., v1.0.x → v1.1.0) are backward compatible:

1. **Optional**: Update to new schema version
2. **Recommended**: Use new features if available
3. **Validation**: Old reports remain valid

### Patch Version Changes (Bug Fixes)

Patch updates (e.g., v1.0.0 → v1.0.1) only fix bugs or clarify documentation:

1. **No migration required**
2. **Update recommended** for clarity
3. **Full compatibility** maintained

## Validation Tools

### Schema Version Checker
```bash
python -m miso.versioning.check_version --report=benchmark_report.json
```

### Migration Validator
```bash
python -m miso.versioning.migrate --from=v0.9.0 --to=v1.0.0 --input=old_report.json
```

### Compatibility Matrix
```bash
python -m miso.versioning.compatibility --matrix
```

## Breaking Changes History

### v1.0.0 → v2.0.0 (Future)
*No breaking changes planned yet. This section will be updated when v2.0.0 is planned.*

**Potential future breaking changes might include:**
- Removing deprecated fields
- Changing required field semantics
- Modifying data type requirements
- Restructuring nested objects

## Best Practices

### For Schema Producers (MISO Core)
1. **Always increment** schema version for any change
2. **Document breaking changes** thoroughly
3. **Provide migration scripts** for major versions
4. **Maintain backward compatibility** for at least 2 major versions
5. **Test migrations** against real-world data

### For Schema Consumers (Users)
1. **Pin schema versions** in production
2. **Test migrations** in staging environment
3. **Validate reports** after migration
4. **Monitor compatibility** matrix regularly
5. **Plan upgrades** around release cycles

## Support

### Getting Help
- Check the [Schema Changelog](SCHEMA_CHANGELOG.md) for recent changes
- Review validation error messages for specific guidance
- Contact the MISO team for complex migration scenarios

### Reporting Issues
- Schema validation failures
- Migration script errors
- Compatibility matrix discrepancies
- Performance issues with large report migrations

## Automation

### CI/CD Integration
```yaml
- name: Validate Schema Version
  run: |
    python -c "
    from miso.versioning import SchemaVersionManager
    manager = SchemaVersionManager()
    report = manager.create_version_report()
    if not report['schema_version_status']['schema_files_valid']:
        print('❌ Schema validation failed')
        exit(1)
    print('✅ Schema version validation passed')
    "
```

### Pre-commit Hooks
```bash
# Add to .pre-commit-config.yaml
- repo: local
  hooks:
    - id: schema-version-check
      name: Schema Version Validation
      entry: python -m miso.versioning.validate
      language: python
      files: schemas/.*\.json$
```
