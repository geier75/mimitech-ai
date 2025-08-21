# VXOR Naming Policy
**Version:** 1.0  
**Status:** Production  
**Enforcement:** Strict  

## 📋 Overview

This document defines the official naming policy for VXOR/vxor/VX/vx identifiers across the entire Python codebase. All code must comply with these conventions to ensure consistency, maintainability, and compatibility across platforms.

## 🎯 Policy Objectives

- **Consistency**: Unified naming conventions across all modules
- **Platform Safety**: Compatible with case-sensitive and case-insensitive filesystems
- **Import Stability**: Reliable import behavior across environments  
- **Developer Experience**: Clear, predictable naming patterns
- **CI Enforcement**: Automated validation in pre-commit and CI pipelines

## 📏 Naming Conventions

### 1. Packages and Modules

**Rule:** Use `snake_case` with lowercase `vxor` prefix

✅ **Correct:**
```python
vxor_core
vxor_utils
vxor_ai_engine
vxor.lang.mcode
```

❌ **Incorrect:**
```python
VXOR_core
VxorCore  
vXor_utils
VX_engine
```

### 2. Classes

**Rule:** Use `PascalCase` with `Vxor` prefix (capital V, lowercase x-o-r)

✅ **Correct:**
```python
class VxorProcessor:
    pass

class VxorAiEngine:
    pass

class VxorNexusController:
    pass
```

❌ **Incorrect:**
```python
class VXORProcessor:    # All caps
class vxorEngine:       # lowercase prefix
class VXor_Controller:  # mixed case with underscore
```

### 3. Functions and Methods

**Rule:** Use `snake_case` with lowercase `vxor` where appropriate

✅ **Correct:**
```python
def process_vxor_data():
    pass

def initialize_vxor_system():
    pass

def vxor_transform(data):
    pass
```

❌ **Incorrect:**
```python
def processVXORData():      # camelCase
def initialize_VXOR_system(): # mixed case
def VxorTransform():        # PascalCase for function
```

### 4. Constants

**Rule:** Use `UPPER_SNAKE_CASE` with `VXOR_` prefix

✅ **Correct:**
```python
VXOR_VERSION = "1.0.0"
VXOR_MAX_ITERATIONS = 1000
VXOR_DEFAULT_CONFIG = {...}
```

❌ **Incorrect:**
```python
vxor_version = "1.0.0"     # lowercase
VxorMaxIterations = 1000   # PascalCase
VX_CONFIG = {...}          # abbreviated prefix
```

### 5. Variables

**Rule:** Use `snake_case` with descriptive names

✅ **Correct:**
```python
vxor_instance = VxorProcessor()
current_vxor_state = "running"
vxor_results = []
```

❌ **Incorrect:**
```python
VxorInstance = VxorProcessor()  # PascalCase
currentVXORState = "running"    # camelCase with caps
vx_res = []                     # abbreviated
```

### 6. File and Directory Names

**Rule:** Use `snake_case` with lowercase throughout

✅ **Correct:**
```
vxor_core/
  __init__.py
  vxor_processor.py
  vxor_utils.py
  tests/
    test_vxor_core.py
```

❌ **Incorrect:**
```
VXOR_Core/
VxorCore/
vXor_utils/
test_VXORCore.py
```

### 7. CLI Commands and Scripts

**Rule:** Use `kebab-case` with lowercase

✅ **Correct:**
```bash
vxor-scan
vxor-fix-naming
vxor-generate-report
```

❌ **Incorrect:**
```bash
VXOR-scan
VxorScan
vxor_scan_files
```

## 🏗️ Legacy and Compatibility

### Deprecated Patterns

These patterns are deprecated and will be flagged by the linter:

```python
# ❌ Deprecated
VXOR (without underscore prefix for constants)
vXor, VXor, vXOR (mixed case variations)
VX (abbreviation in new code)

# ✅ Migration
VXOR_CONSTANT   # Use VXOR_ prefix for constants
vxor            # Use lowercase for modules/functions  
VxorClass       # Use PascalCase for classes
```

### Backward Compatibility Shims

Legacy aliases are maintained via deprecation warnings:

```python
# In legacy module
import warnings

# New canonical name
VxorProcessor = VxorProcessor

# Deprecated alias with warning
def VXOR_Process(*args, **kwargs):
    warnings.warn(
        "VXOR_Process is deprecated, use VxorProcessor instead",
        DeprecationWarning,
        stacklevel=2
    )
    return VxorProcessor(*args, **kwargs)
```

## 🔍 Validation and Enforcement

### Automated Checks

The naming policy is enforced through multiple layers:

1. **Pre-commit Hooks**: Block commits with violations
2. **CI Pipeline**: Fail builds on naming violations  
3. **IDE Integration**: Real-time linting feedback
4. **Test Suite**: Automated compliance verification

### Scanning Commands

```bash
# Scan for violations
make scan

# Fix specific category
make fix-directories
make fix-files
make fix-imports

# Fix all violations
make all-categories

# Generate compliance report
make report
```

### Violation Categories

| Category | Description | Priority |
|----------|-------------|----------|
| `directories` | Directory naming violations | High |
| `files` | File naming violations | High |
| `imports` | Import statement inconsistencies | High |
| `symbols` | Class/function/constant violations | Medium |
| `shims` | Missing backward compatibility | Low |
| `docs_cli` | Documentation references | Low |

## 🚨 CI Integration

### Pre-commit Gate

The VXOR naming policy gate runs on every commit:

```yaml
- id: vxor-naming-gate
  name: "🚨 VXOR Naming Policy Gate"
  entry: python vxor_naming_nano_cli.py gate --strict
```

### GitHub Actions

Matrix testing across platforms ensures consistent behavior:

- **Ubuntu**: Case-sensitive filesystem testing
- **macOS**: Case-insensitive filesystem testing  
- **Validation**: Import stability, test suite, linting

## 🔧 Developer Workflow

### Making Changes

1. **Before coding**: Review naming conventions
2. **During development**: Use descriptive, compliant names
3. **Pre-commit**: Automated validation catches violations
4. **CI/CD**: Cross-platform validation ensures compatibility

### Fixing Violations

Use the nano-step TDD approach:

```bash
# 1. Find violations  
make scan-single

# 2. Fix one violation (TDD cycle)
make fix-one

# 3. Verify fix
make test-violation VIOLATION_ID=<id>

# 4. Repeat until clean
make all-categories
```

### IDE Setup

Configure your IDE with these tools:

- **Linter**: Ruff with VXOR naming rules
- **Formatter**: Black for consistent style
- **Type Checker**: MyPy for type safety
- **Pre-commit**: Automatic hook installation

## 📚 Examples

### Complete Example

```python
"""
vxor_ai_engine.py - VXOR AI Engine Module
"""

# Constants
VXOR_ENGINE_VERSION = "2.1.0"  
VXOR_MAX_WORKERS = 8
VXOR_DEFAULT_TIMEOUT = 30.0

# Imports  
from vxor_core import VxorProcessor
from vxor_utils import vxor_logging

# Classes
class VxorAiEngine:
    """Main VXOR AI processing engine."""
    
    def __init__(self, config: dict):
        self.vxor_config = config
        self.vxor_processor = VxorProcessor()
        self.vxor_logger = vxor_logging.get_logger(__name__)
    
    def process_vxor_data(self, data: list) -> dict:
        """Process data through VXOR AI pipeline."""
        vxor_results = {}
        
        for vxor_item in data:
            processed_item = self.vxor_processor.transform(vxor_item)
            vxor_results[vxor_item.id] = processed_item
            
        return vxor_results
    
    def get_vxor_status(self) -> str:
        """Get current VXOR engine status."""
        return "running" if self.vxor_processor.is_active else "stopped"

# Factory function
def create_vxor_engine(config_path: str) -> VxorAiEngine:
    """Create and configure VXOR AI engine instance."""
    with open(config_path) as f:
        vxor_config = json.load(f)
    
    return VxorAiEngine(vxor_config)
```

## 🔄 Migration Strategy

### Phase 1: Assessment
- Run comprehensive scan: `make scan`
- Review violation report
- Prioritize by category and impact

### Phase 2: Automated Fixes
- Fix directories: `make fix-directories`
- Fix files: `make fix-files`  
- Fix imports: `make fix-imports`

### Phase 3: Manual Review
- Fix symbols: `make fix-symbols`
- Add shims: `make fix-shims`
- Update docs: `make fix-docs`

### Phase 4: Validation  
- Full test suite: `make test`
- Import stability: `make verify`
- CI pipeline validation

## ❓ FAQ

**Q: Why not use all caps VXOR everywhere?**  
A: PEP 8 reserves ALL_CAPS for constants only. Classes use PascalCase, modules use snake_case.

**Q: Can I abbreviate to VX?**  
A: Avoid VX in new code. Use full vxor for clarity. Existing VX may be grandfathered.

**Q: What about third-party dependencies?**  
A: External packages keep their naming. Only internal VXOR code follows this policy.

**Q: How do I handle legacy code?**  
A: Use gradual migration with deprecation warnings. The toolkit provides automated fixes.

**Q: Why the nano-step approach?**  
A: Small, atomic changes reduce risk and enable precise testing and rollback.

---

**Enforcement:** This policy is automatically enforced via pre-commit hooks and CI pipelines.  
**Support:** Use `make help` for toolkit commands or see the [Migration Guide](migration-guide.md).
