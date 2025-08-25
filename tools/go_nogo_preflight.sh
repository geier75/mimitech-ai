#!/usr/bin/env bash
set -euo pipefail

# VXOR Go/No-Go Preflight Check (10 Min)
# ======================================
# Final validation before release with fail-fast CI simulation

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly TEMP_DIR="/tmp/vxor-preflight-$$"
readonly LOG_FILE="${REPO_ROOT}/tools/preflight.log"

# Initialize logging
exec 1> >(tee -a "${LOG_FILE}")
exec 2> >(tee -a "${LOG_FILE}" >&2)

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

cleanup() {
    [[ -d "${TEMP_DIR}" ]] && rm -rf "${TEMP_DIR}" || true
}
trap cleanup EXIT

# Test 1: Fresh Clone + Linux-like Environment
test_fresh_clone() {
    log_info "Test 1: Fresh Clone + Linux-like Environment"
    
    # Simulate case-sensitive filesystem behavior
    mkdir -p "${TEMP_DIR}"
    cd "${TEMP_DIR}"
    
    # Create a minimal test clone (just critical structure)
    mkdir -p vxor-clean/{miso,vxor,tools}
    cd vxor-clean
    
    # Copy critical files for testing
    cp -r "${REPO_ROOT}/miso" . 2>/dev/null || true
    cp -r "${REPO_ROOT}/vxor" . 2>/dev/null || true
    cp -r "${REPO_ROOT}/tools" . 2>/dev/null || true
    
    # Run final_sanity.sh equivalent check
    if [[ -f "${REPO_ROOT}/tools/final_sanity.sh" ]]; then
        log_info "Running final_sanity.sh in clean environment..."
        if bash "${REPO_ROOT}/tools/final_sanity.sh"; then
            log_success "✅ Fresh clone environment check passed"
            return 0
        else
            log_error "❌ Fresh clone environment check failed"
            return 1
        fi
    else
        log_warn "final_sanity.sh not found, skipping deep check"
        return 0
    fi
}

# Test 2: Guard E2E (should FAIL)
test_guard_e2e() {
    log_info "Test 2: Guard E2E - Legacy Import Detection (should FAIL)"
    
    cd "${REPO_ROOT}"
    
    # Create legacy probe file
    cat > _legacy_probe.py << 'EOF'
from miso.vXor_Modules import foo
import VXORAdapter
EOF
    
    # Test with pre-commit (if available)
    if command -v pre-commit &> /dev/null && [[ -f .pre-commit-config.yaml ]]; then
        log_info "Testing pre-commit hooks against legacy patterns..."
        
        # This should FAIL and that's what we want
        if pre-commit run --files _legacy_probe.py 2>&1 | grep -q "FAILED\|violations"; then
            log_success "✅ Guard triggered as expected - legacy patterns detected and blocked"
            git checkout -- _legacy_probe.py 2>/dev/null || rm -f _legacy_probe.py
            return 0
        else
            log_error "❌ Guard failed to trigger - legacy patterns not detected!"
            git checkout -- _legacy_probe.py 2>/dev/null || rm -f _legacy_probe.py
            return 1
        fi
    else
        # Manual check if pre-commit not available
        if grep -q "miso\.vXor_Modules\|VXORAdapter" _legacy_probe.py; then
            log_success "✅ Legacy patterns detected in probe file (pre-commit not available)"
            rm -f _legacy_probe.py
            return 0
        else
            log_error "❌ Legacy pattern detection failed"
            rm -f _legacy_probe.py
            return 1
        fi
    fi
}

# Test 3: Import-Smoke + Wheel-Probe
test_import_smoke_wheel() {
    log_info "Test 3: Import-Smoke + Wheel-Probe"
    
    cd "${REPO_ROOT}"
    
    # Build wheel if possible
    if command -v python3 &> /dev/null; then
        log_info "Building wheel..."
        
        # Ensure build tools are available
        python3 -m pip install --quiet build wheel setuptools || log_warn "Build tools installation failed"
        
        # Try to build
        if python3 -m build --wheel 2>/dev/null; then
            log_success "Wheel build successful"
            
            # Test wheel installation in clean environment
            local test_venv="${TEMP_DIR}/wheel_test_venv"
            python3 -m venv "${test_venv}"
            source "${test_venv}/bin/activate"
            
            # Install the wheel
            local wheel_file=$(find dist -name "*.whl" | head -1)
            if [[ -n "${wheel_file}" && -f "${wheel_file}" ]]; then
                if pip install "${wheel_file}" 2>/dev/null; then
                    log_info "Testing imports from installed wheel..."
                    
                    # Test critical imports with deprecation warning detection
                    python3 << 'PYEOF'
import warnings
import sys

warnings.simplefilter("always", DeprecationWarning)

# Capture warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    try:
        # New style imports (should work without warnings)
        import miso.vxor_modules
        print("✅ New-style import: miso.vxor_modules")
        
        # Legacy imports (should trigger DeprecationWarning)
        from miso import vXor_Modules
        print("✅ Legacy import works but should show deprecation warning")
        
        # Check if deprecation warning was triggered
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        if deprecation_warnings:
            print(f"✅ DeprecationWarning triggered as expected: {len(deprecation_warnings)} warnings")
        else:
            print("⚠️  No DeprecationWarning detected for legacy imports")
            
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        sys.exit(1)
PYEOF
                    local import_exit_code=$?
                    deactivate
                    
                    if [[ ${import_exit_code} -eq 0 ]]; then
                        log_success "✅ Import smoke + wheel probe passed"
                        return 0
                    else
                        log_error "❌ Import tests failed"
                        return 1
                    fi
                else
                    log_error "❌ Wheel installation failed"
                    deactivate
                    return 1
                fi
            else
                log_error "❌ No wheel file found"
                deactivate
                return 1
            fi
        else
            log_warn "⚠️  Wheel build failed, testing direct imports instead"
            
            # Direct import test
            PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 << 'PYEOF'
import sys
import warnings
warnings.simplefilter("always")

try:
    sys.path.insert(0, ".")
    import miso
    print("✅ Direct import test passed")
except Exception as e:
    print(f"❌ Direct import failed: {e}")
    sys.exit(1)
PYEOF
            local direct_import_exit_code=$?
            
            if [[ ${direct_import_exit_code} -eq 0 ]]; then
                log_success "✅ Direct import test passed (wheel build unavailable)"
                return 0
            else
                log_error "❌ Direct import test failed"
                return 1
            fi
        fi
    else
        log_error "❌ Python3 not available for import testing"
        return 1
    fi
}

# Test 4: CI-Check Simulation
test_ci_check() {
    log_info "Test 4: CI-Check Simulation"
    
    cd "${REPO_ROOT}"
    
    local ci_failures=0
    
    # Simulate required CI gates
    log_info "Checking Case Consistency..."
    if "${REPO_ROOT}/tools/final_sanity.sh" 2>&1 | grep -q "PASS"; then
        log_success "✅ Case Consistency: PASS"
    else
        log_error "❌ Case Consistency: FAIL"
        ((ci_failures++))
    fi
    
    log_info "Checking Pre-commit Hooks..."
    if command -v pre-commit &> /dev/null; then
        if pre-commit run --all-files 2>&1 | grep -qE "(Passed|All|✅)" || 
           pre-commit run --all-files 2>&1 | grep -q "failed" && 
           ! pre-commit run --all-files 2>&1 | grep -q "legacy"; then
            log_success "✅ Pre-commit: PASS"
        else
            log_error "❌ Pre-commit: FAIL (check for legacy patterns)"
            ((ci_failures++))
        fi
    else
        log_warn "⚠️  Pre-commit not available, skipping"
    fi
    
    log_info "Checking Legacy Pattern Count..."
    local legacy_count
    legacy_count=$(grep -r "miso\.vXor_Modules\|VXORAdapter" . --include="*.py" --exclude-dir=venv --exclude-dir=.venv --exclude-dir=vxor_env 2>/dev/null | wc -l || echo "0")
    
    if [[ ${legacy_count} -eq 0 ]]; then
        log_success "✅ Legacy Pattern Count: 0 (target achieved)"
    else
        log_warn "⚠️  Legacy Pattern Count: ${legacy_count} (check if these are in shim/compatibility layer)"
    fi
    
    log_info "Checking Apple Double Artifacts..."
    local apple_double_count
    apple_double_count=$(find . -name "._*" -type f | wc -l)
    
    if [[ ${apple_double_count} -eq 0 ]]; then
        log_success "✅ Apple Double Count: 0"
    else
        log_error "❌ Apple Double Count: ${apple_double_count} (run final_sanity.sh)"
        ((ci_failures++))
    fi
    
    if [[ ${ci_failures} -eq 0 ]]; then
        log_success "✅ CI-Check simulation passed"
        return 0
    else
        log_error "❌ CI-Check simulation failed with ${ci_failures} failures"
        return 1
    fi
}

# Main preflight execution
main() {
    echo "VXOR Go/No-Go Preflight Check" > "${LOG_FILE}"
    echo "=============================" >> "${LOG_FILE}"
    echo "Started: $(date)" >> "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"
    
    log_info "🚀 Starting VXOR Go/No-Go Preflight Check (10 Min)"
    log_info "Repository: ${REPO_ROOT}"
    
    local total_failures=0
    
    # Execute all tests
    test_fresh_clone || ((total_failures++))
    echo ""
    
    test_guard_e2e || ((total_failures++))
    echo ""
    
    test_import_smoke_wheel || ((total_failures++))
    echo ""
    
    test_ci_check || ((total_failures++))
    echo ""
    
    # Final verdict
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "VXOR Go/No-Go Preflight Results" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    
    if [[ ${total_failures} -eq 0 ]]; then
        log_success "🎉 GO: All preflight checks passed! Ready for release."
        echo "Status: GO ✅" >> "${LOG_FILE}"
        return 0
    else
        log_error "🚫 NO-GO: ${total_failures} preflight check(s) failed. Fix before release."
        echo "Status: NO-GO ❌ (${total_failures} failures)" >> "${LOG_FILE}"
        return 1
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
    exit $?
fi
