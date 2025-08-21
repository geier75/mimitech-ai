# VXOR Case-Consistency Migration Guide
**Version:** 1.0  
**Target:** MISO Ultimate 15.32.28  
**Approach:** Nano-Step TDD  

## 🚀 Quick Start

### Prerequisites

```bash
# 1. Clone the toolkit
cd /Users/gecko365/vxor_naming_toolkit

# 2. Check dependencies  
make check-dependencies

# 3. Initialize toolkit
make init

# 4. Quick status check
make quick-start
```

### 5-Minute Migration

```bash
# Complete automated migration
make all-categories

# Or step-by-step
make fix-directories  # Fix directory names
make fix-files        # Fix file names  
make fix-imports      # Fix import statements
make fix-symbols      # Fix class/function names
make verify           # Validate changes
```

## 🎯 Migration Strategy

### Phase-Based Approach

The migration follows a strict category order to minimize dependencies:

```
1. Directories  → 2. Files → 3. Imports → 4. Symbols → 5. Shims → 6. Docs
   (Foundation)    (Names)   (Stability)  (Code)      (Compat)   (References)
```

**Why this order?**
- **Directories first**: Foundation for all other references
- **Files next**: Enable correct imports  
- **Imports third**: Ensure stability before symbol changes
- **Symbols fourth**: Classes/functions depend on imports
- **Shims fifth**: Backward compatibility for legacy code
- **Docs last**: Update references after code is stable

## 🧪 Nano-Step TDD Process

### The Nano-Step Loop

Each violation follows this strict TDD cycle:

```
📍 1. SCAN       → Find next violation
🧪 2. TEST(FAIL) → Generate failing test  
🔧 3. FIX        → Apply minimal fix
🧪 4. TEST(PASS) → Verify test now passes
✅ 5. COMMIT     → Atomic commit with ID
📊 6. REPORT     → Update progress log
🔄 7. NEXT       → Loop to next violation
```

### Manual Nano-Step

```bash
# Get the next violation
make scan-single CURRENT_CATEGORY=directories

# Fix exactly one violation (full TDD cycle)
make fix-one CURRENT_CATEGORY=directories

# The fix-one command does all 7 steps automatically
```

### Automated Categories

```bash
# Process all violations in a category
make fix-directories   # ~10-50 violations
make fix-files         # ~20-80 violations  
make fix-imports       # ~5-30 violations
make fix-symbols       # ~15-100 violations

# Process everything
make all-categories    # Full migration
```

## 📊 Migration Assessment

### Pre-Migration Scan

```bash
# Comprehensive violation scan
make scan > migration-assessment.txt

# Category breakdown
make status

# Generate detailed report
make report
```

**Expected Violation Counts** (MISO Ultimate):
- **Directories**: 15-25 violations
- **Files**: 40-60 violations
- **Imports**: 20-30 violations  
- **Symbols**: 80-120 violations
- **Total**: ~200-300 violations

### Risk Assessment

| Category | Risk Level | Impact | Automation |
|----------|------------|--------|------------|
| Directories | 🟡 Medium | High | Full |
| Files | 🟡 Medium | High | Full |
| Imports | 🟢 Low | Medium | Full |
| Symbols | 🔴 High | Low | Partial |
| Shims | 🟢 Low | Low | Full |
| Docs | 🟢 Low | Low | Full |

## 🔧 Category-Specific Guides

### 1. Directories (`make fix-directories`)

**What it fixes:**
```bash
# Before
VXOR_Core/
VxorAI/  
vX_lang/

# After  
vxor_core/
vxor_ai/
vxor_lang/
```

**Process:**
1. Rename directory on filesystem
2. Update all import statements referencing the directory
3. Update relative imports within affected modules
4. Verify import stability

**Manual intervention needed:** None - fully automated

### 2. Files (`make fix-files`)

**What it fixes:**
```bash  
# Before
VXORProcessor.py
vxor_AI_Engine.py
VX_Controller.py

# After
vxor_processor.py
vxor_ai_engine.py  
vxor_controller.py
```

**Process:**
1. Rename file on filesystem
2. Update imports from old module name to new
3. Update relative imports and references
4. Validate import chains

**Manual intervention needed:** None - fully automated

### 3. Imports (`make fix-imports`)

**What it fixes:**
```python
# Before
from VXOR_Core import VXORProcessor
import VxorUtils
from vX.lang import MCode

# After
from vxor_core import VxorProcessor  
import vxor_utils
from vxor.lang import mcode
```

**Process:**
1. Parse AST for import statements
2. Apply canonical naming rules
3. Update module references  
4. Test import stability

**Manual intervention needed:** Rare - some complex imports may need review

### 4. Symbols (`make fix-symbols`)

**What it fixes:**
```python
# Before  
class VXORProcessor:        → class VxorProcessor:
def processVXORData():      → def process_vxor_data(): 
VXOR_MAX_SIZE = 1000        → VXOR_MAX_SIZE = 1000  # (correct)
vxor_processor_instance     → vxor_processor_instance  # (correct)
```

**Process:**
1. AST-based symbol detection
2. Apply naming conventions per symbol type
3. Rename all references within scope
4. Generate tests for each change

**Manual intervention needed:** Sometimes - complex inheritance or dynamic references

### 5. Shims (`make fix-shims`)

**What it creates:**
```python
# Backward compatibility aliases
import warnings

# New canonical implementation
class VxorProcessor:
    pass

# Deprecated alias with warning
class VXORProcessor(VxorProcessor):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "VXORProcessor is deprecated, use VxorProcessor",
            DeprecationWarning, stacklevel=2
        )
        super().__init__(*args, **kwargs)
```

**Process:**
1. Identify deprecated patterns still in use
2. Generate compatibility shims
3. Add deprecation warnings
4. Update documentation

**Manual intervention needed:** Minimal - may need custom shims for complex cases

### 6. Documentation (`make fix-docs`)

**What it fixes:**
```markdown
# Before
## VXOR-CORE Module
Use `VXORProcessor` class...
Run: `VXOR-scan --all`

# After  
## vxor_core Module
Use `VxorProcessor` class...
Run: `vxor-scan --all`
```

**Process:**
1. Update README files
2. Fix code examples in documentation
3. Update CLI command references
4. Correct class/function names in docs

**Manual intervention needed:** Some - documentation may need contextual review

## 🛠️ Advanced Usage

### Custom Migration Workflows

#### Selective Category Migration
```bash
# Only fix critical issues
make fix-directories
make fix-files
make verify

# Skip less critical categories
# make fix-symbols  # Skip for now
# make fix-shims    # Skip for now
```

#### Safe Mode Migration
```bash
# Preview all changes first
python vxor_naming_nano_cli.py scan --category all --dry-run

# Fix with confirmation prompts
make fix-one CURRENT_CATEGORY=directories  # Interactive
make fix-one CURRENT_CATEGORY=files       # Interactive
```

#### Batch Mode Migration  
```bash
# Fully automated (use with caution)
export VXOR_AUTO_FIX=true
make all-categories
```

### Handling Edge Cases

#### Complex Import Chains
```python
# Problem: Dynamic imports or complex module structures  
module_name = f"vxor_{engine_type}"
imported_module = importlib.import_module(module_name)

# Solution: Update string templates
module_name = f"vxor_{engine_type.lower()}"  # Ensure lowercase
```

#### Legacy Code Integration
```python
# Problem: Third-party code expects old names
external_lib.register_processor(VXORProcessor)

# Solution: Use shim for external compatibility
VXORProcessor = VxorProcessor  # Alias without deprecation warning
```

#### Case-Sensitive Filesystem Issues
```bash
# Problem: Git not detecting case-only renames on macOS
git mv oldFile.py tempfile.py
git mv tempfile.py newfile.py

# Solution: Use toolkit's safe rename (handles automatically)
make fix-files  # Handles cross-platform case renames
```

## 🧪 Testing and Validation

### Test Strategy

1. **Pre-migration Tests**
   ```bash
   make test                    # Baseline test run
   python -m pytest -x         # Stop on first failure
   ```

2. **During Migration Tests**  
   ```bash
   make test-violation VIOLATION_ID=vxor_001  # Test specific fix
   make verify                  # Import stability check
   ```

3. **Post-migration Tests**
   ```bash
   make test                    # Full test suite
   make ci-test                 # CI simulation
   ```

### Validation Checklist

- [ ] All tests pass: `make test`
- [ ] Import stability: `make verify`  
- [ ] No violations found: `make scan`
- [ ] CI pipeline passes: Check GitHub Actions
- [ ] Documentation updated: Review changed files
- [ ] Backward compatibility: Test legacy imports

### Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Import errors | `ModuleNotFoundError` | Run `make fix-imports` |
| Case collisions | Git conflicts on macOS | Use `make fix-files` (handles safely) |
| Test failures | Tests can't find modules | Update test imports manually |
| Legacy breaks | Old code stops working | Add shims: `make fix-shims` |

## 📊 Progress Tracking

### Real-time Monitoring

```bash
# Current status
make status

# Detailed progress
make report

# Watch progress during migration
watch -n 5 'make status'
```

### Report Formats

- **Console**: `make status` - Quick overview
- **Markdown**: `reports/summary.md` - Comprehensive report  
- **JSON**: `reports/metrics.json` - Machine-readable data
- **JSONL**: `reports/nano_steps.jsonl` - Detailed change log

### Progress Metrics

The toolkit tracks:
- **Violations found/fixed/pending** by category
- **Success rate** of nano-step fixes  
- **Average fix time** per violation
- **Test pass rate** after fixes
- **Import stability** across modules

## 🚨 Rollback and Recovery

### Safe Points

Create git commits at key milestones:

```bash
git add -A && git commit -m "chore: pre-migration baseline"
make fix-directories
git add -A && git commit -m "chore: directories migrated"
make fix-files  
git add -A && git commit -m "chore: files migrated"
# Continue per category...
```

### Rollback Strategy

```bash
# Rollback to previous category
git reset --hard HEAD~1

# Rollback to start of migration
git reset --hard <baseline-commit-hash>

# Selective rollback of specific files
git checkout HEAD~1 -- path/to/problematic/file.py
```

### Recovery Procedures

#### If Migration Breaks Tests
```bash
# 1. Identify failing tests
make test 2>&1 | grep FAILED

# 2. Fix imports in test files
make fix-imports

# 3. Re-run tests
make test

# 4. Manual fixes if needed
# Edit test files to use new names
```

#### If Import Chain Breaks
```bash
# 1. Check import stability
make verify

# 2. Fix detected issues
make fix-imports

# 3. Add missing shims
make fix-shims

# 4. Re-verify
make verify
```

## 📋 Migration Checklist

### Pre-Migration
- [ ] Backup current codebase: `git tag pre-vxor-migration`
- [ ] Install toolkit dependencies: `make check-dependencies`
- [ ] Run baseline tests: `make test`
- [ ] Generate violation report: `make scan > pre-migration-scan.txt`

### Migration Execution  
- [ ] Initialize toolkit: `make init`
- [ ] Fix directories: `make fix-directories`
- [ ] Fix files: `make fix-files`
- [ ] Fix imports: `make fix-imports`  
- [ ] Fix symbols: `make fix-symbols`
- [ ] Add shims: `make fix-shims`
- [ ] Update docs: `make fix-docs`

### Post-Migration
- [ ] Verify no violations: `make scan` (should show 0)
- [ ] Run full test suite: `make test`
- [ ] Check import stability: `make verify`
- [ ] Generate completion report: `make report`
- [ ] Tag completion: `git tag vxor-migration-complete`

### CI/CD Integration
- [ ] Update pre-commit hooks: Copy `.pre-commit-config.yaml` to repo
- [ ] Install hooks: `pre-commit install`
- [ ] Update GitHub Actions: Copy workflow file
- [ ] Test CI pipeline: Make test commit

## 🎉 Success Criteria

Migration is considered successful when:

1. **Zero violations**: `make scan` returns no violations
2. **All tests pass**: `make test` shows 100% pass rate  
3. **Import stability**: `make verify` confirms all imports work
4. **CI pipeline passes**: GitHub Actions workflow succeeds
5. **Documentation current**: All references use new naming conventions

### Final Validation

```bash
# Comprehensive final check
make ci-test    # Simulate CI environment
make report     # Generate completion report
make status     # Verify clean state

# Should show:
# ✅ Clean (0 violations) for all categories
# ✅ All tests passing
# ✅ Import stability verified
# ✅ CI pipeline ready
```

## 📞 Support and Troubleshooting

### Command Reference
```bash
make help                    # Show all commands
make help-nano-step         # Nano-step workflow help  
make help-categories        # Category processing help
```

### Debug Commands
```bash
make debug-violation VIOLATION_ID=<id>  # Debug specific violation
make show-categories                      # List available categories
python vxor_naming_nano_cli.py --help   # CLI help
```

### Getting Help
1. **Command documentation**: `make help`
2. **Naming policy**: See `naming-policy.md`
3. **CLI reference**: `python vxor_naming_nano_cli.py --help`
4. **Issue tracking**: Check toolkit reports in `reports/`

---

**Success Path**: `make all-categories` → `make verify` → `make test` → Done! 🎉
