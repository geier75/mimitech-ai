# VXOR Case Consistency Release Notes

## Breaking Change: Case-consistent module paths

### New Module Paths ✅
- `miso.vxor_modules` - Main VXOR modules access
- `vxor.ai.vx_matrix` - VX Matrix AI components  
- `vxor.core.nexus_os` - NEXUS OS core
- `vxor.agents.vx_memex` - VX Memory agents

### Deprecated Paths ⚠️ 
- `miso.vXor_Modules` → **deprecated** (use `miso.vxor_modules`)
- `VX-MATRIX` → **deprecated** (use `vx_matrix`)
- `VXOR_Logs` → **deprecated** (use `vxor_logs`)

## Developer Action Required

### 1. Update Import Statements
```bash
# Search & Replace in your codebase:
find . -name "*.py" -exec sed -i '' 's/miso\.vXor_Modules/miso.vxor_modules/g' {} \;
find . -name "*.py" -exec sed -i '' 's/VXORAdapter/vxorAdapter/g' {} \;
```

### 2. Enable Pre-commit Hooks
```bash
pre-commit install
```

### 3. Fix Import Errors
If you encounter import errors after upgrading:
```bash
bash tools/final_sanity.sh
```

## Backward Compatibility

### Shim Layer Active
- Legacy imports still work via compatibility shim
- **DeprecationWarning** issued on legacy import usage
- Shim will be removed in vNext (see Sunset Plan below)

### Example Migration
```python
# Old (deprecated) ⚠️
from miso.vXor_Modules import adapter
adapter = VXORAdapter()

# New (recommended) ✅  
from miso.vxor_modules import adapter
adapter = vxorAdapter()
```

## Benefits

### Case-Sensitive Filesystem Compatibility
- Eliminates import failures on Linux/container environments
- Prevents case collision conflicts in Git
- Improves cross-platform development experience

### CI/CD Improvements
- Automated case-consistency validation
- Pre-commit hooks prevent regressions
- Fail-fast detection of legacy patterns

## Sunset Plan for Legacy Shim

### Timeline
- **Now**: Shim active, DeprecationWarning on import
- **T+2 Releases**: Warning escalated to Error in CI (PR gate), local remains Warning
- **vNext**: Shim removed completely; Major release with migration guide

### Recommended Tickets
- **IP-212**: "Raise-on-Legacy in CI ab Release+2"
- **IP-213**: "Shim entfernen in vNext" 
- **DOC-87**: "Migrations-Guide Case-Consistency"

## Monitoring

### Weekly Metrics
- `legacy_import_findings`: Count of CI blockers (target: 0)
- `sanity_artifact_cleaned`: Apple Double/DS_Store removals (target: 0)
- `case_collision_alerts`: Case conflict detections (target: 0)

### Alerts
- Slack #vxor-ci @owner if `legacy_import_findings > 0`

## Known Edge Cases

### Windows/Network Shares
- Case-insensitive filesystems require extra care
- Always run `tools/final_sanity.sh` before commit

### Editable Installs
- `pip install -e .` may cache old paths
- Fix with: `pip uninstall <pkg> -y && pip install .`

### Package Resources
- Check MANIFEST.in and package_data for old paths
- Update resource references to new naming

## Support

For migration issues or questions:
- Check tools/final_sanity.sh output
- Review pre-commit hook messages  
- Contact: #vxor-support

---
**Migration Deadline**: Please migrate by vNext to avoid breaking changes.
