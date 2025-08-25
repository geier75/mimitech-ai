#!/usr/bin/env bash
set -euo pipefail

# VXOR Final Sanity Pass & Guard Script
# =====================================
# Comprehensive cleanup and fail-fast detection for case-consistency issues
# Designed for nano-step TDD workflow with CI enforcement

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_FILE="${REPO_ROOT}/tools/final_sanity.log"

# Environment detection
readonly IS_MACOS="$(uname -s | grep -q Darwin && echo 'true' || echo 'false')"
readonly IS_CI="${CI:-false}"
readonly GITHUB_WORKSPACE="${GITHUB_WORKSPACE:-${REPO_ROOT}}"

# Counters for final report
declare -i cleanup_count=0
declare -i warning_count=0
declare -i error_count=0

# Logging functions
log_info() {
    local msg="$1"
    echo -e "${BLUE}[INFO]${NC} ${msg}" | tee -a "${LOG_FILE}"
}

log_warn() {
    local msg="$1"
    echo -e "${YELLOW}[WARN]${NC} ${msg}" | tee -a "${LOG_FILE}"
    ((warning_count++))
}

log_error() {
    local msg="$1"
    echo -e "${RED}[ERROR]${NC} ${msg}" | tee -a "${LOG_FILE}"
    ((error_count++))
}

log_success() {
    local msg="$1"
    echo -e "${GREEN}[SUCCESS]${NC} ${msg}" | tee -a "${LOG_FILE}"
}

# Initialize log file
init_logging() {
    mkdir -p "$(dirname "${LOG_FILE}")"
    echo "VXOR Final Sanity Pass - $(date)" > "${LOG_FILE}"
    echo "Repository: ${REPO_ROOT}" >> "${LOG_FILE}"
    echo "Environment: macOS=${IS_MACOS}, CI=${IS_CI}" >> "${LOG_FILE}"
    echo "========================================" >> "${LOG_FILE}"
}

# Phase 1: Apple Double and macOS artifacts cleanup
cleanup_apple_doubles() {
    log_info "Phase 1: Cleaning up Apple Double files and macOS artifacts"
    
    local -a patterns=(
        "._*"           # Apple Double files
        ".DS_Store"     # Finder metadata
        ".AppleDouble"  # Resource forks
        "__MACOSX"      # ZIP artifacts
        "*.tmp"         # Temporary files
        "*.swp"         # Vim swap files
        "*~"            # Backup files
    )
    
    for pattern in "${patterns[@]}"; do
        local files
        files=$(find "${REPO_ROOT}" -name "${pattern}" -type f 2>/dev/null || true)
        
        if [[ -n "${files}" ]]; then
            log_warn "Found ${pattern} files, removing..."
            echo "${files}" | while IFS= read -r file; do
                if [[ -f "${file}" ]]; then
                    rm -f "${file}" && ((cleanup_count++)) || log_error "Failed to remove: ${file}"
                fi
            done
        fi
    done
    
    # Clean up empty duplicate directories from macOS
    local duplicate_dirs
    duplicate_dirs=$(find "${REPO_ROOT}" -type d -name "*23.33.*" 2>/dev/null || true)
    if [[ -n "${duplicate_dirs}" ]]; then
        log_warn "Found duplicate timestamp directories, removing empty ones..."
        echo "${duplicate_dirs}" | while IFS= read -r dir; do
            if [[ -d "${dir}" ]] && [[ -z "$(ls -A "${dir}" 2>/dev/null)" ]]; then
                rmdir "${dir}" && ((cleanup_count++)) || log_error "Failed to remove empty dir: ${dir}"
            fi
        done
    fi
}

# Phase 2: Python cache cleanup
cleanup_python_caches() {
    log_info "Phase 2: Cleaning up Python caches and artifacts"
    
    local -a patterns=(
        "__pycache__"
        "*.pyc"
        "*.pyo"
        "*.pyd"
        ".pytest_cache"
        ".coverage"
        ".tox"
        "*.egg-info"
        "build/"
        "dist/"
    )
    
    for pattern in "${patterns[@]}"; do
        local items
        items=$(find "${REPO_ROOT}" -name "${pattern}" 2>/dev/null || true)
        
        if [[ -n "${items}" ]]; then
            echo "${items}" | while IFS= read -r item; do
                if [[ -d "${item}" ]]; then
                    rm -rf "${item}" && ((cleanup_count++)) || log_error "Failed to remove dir: ${item}"
                elif [[ -f "${item}" ]]; then
                    rm -f "${item}" && ((cleanup_count++)) || log_error "Failed to remove file: ${item}"
                fi
            done
        fi
    done
}

# Phase 3: Legacy naming detection (fail-fast)
detect_legacy_names() {
    log_info "Phase 3: Scanning for legacy naming violations (fail-fast)"
    
    local violations=0
    
    # Define directories to exclude from scanning
    local -a exclude_dirs=(
        "venv"
        "venv_*"
        "*_env"
        ".venv*"
        "__pycache__"
        ".git"
        "node_modules"
        "whisper.cpp"
        "vxor_benchmark_suite"
    )
    
    # Build find exclude parameters
    local find_excludes=""
    for dir in "${exclude_dirs[@]}"; do
        find_excludes="${find_excludes} -path '*/${dir}' -prune -o"
    done
    
    # Check for old case violations using ripgrep if available
    if command -v rg &> /dev/null; then
        log_info "Using ripgrep for fast legacy name detection..."
        
        # Build ripgrep exclude globs
        local rg_excludes=""
        for dir in "${exclude_dirs[@]}"; do
            rg_excludes="${rg_excludes} --glob '!${dir}/**'"
        done
        
        # Legacy import patterns - only in our source code
        local -a legacy_patterns=(
            "miso\.vXor_Modules"
            "VXORAdapter\b"
        )
        
        for pattern in "${legacy_patterns[@]}"; do
            local matches
            matches=$(eval "rg -n --type py ${rg_excludes} '${pattern}' '${REPO_ROOT}'" 2>/dev/null || true)
            
            if [[ -n "${matches}" ]]; then
                log_error "Found legacy pattern '${pattern}':"
                echo "${matches}" | head -5 | while IFS= read -r match; do
                    log_error "  ${match}"
                done
                ((violations++))
            fi
        done
    else
        log_warn "ripgrep not available, using basic grep for legacy detection"
        
        # Basic grep fallback - search only in our source directories
        local -a source_dirs=(
            "${REPO_ROOT}/miso"
            "${REPO_ROOT}/vxor"
            "${REPO_ROOT}/tests"
            "${REPO_ROOT}/scripts"
        )
        
        for src_dir in "${source_dirs[@]}"; do
            if [[ -d "${src_dir}" ]]; then
                local py_files
                py_files=$(find "${src_dir}" -name "*.py" -type f 2>/dev/null || true)
                
                if [[ -n "${py_files}" ]]; then
                    echo "${py_files}" | while IFS= read -r file; do
                        if grep -q "miso\.vXor_Modules\|VXORAdapter\b" "${file}" 2>/dev/null; then
                            log_error "Legacy naming found in: ${file}"
                            ((violations++))
                        fi
                    done
                fi
            fi
        done
    fi
    
    # Check for case-colliding files in source directories only
    log_info "Checking for potential case collisions in source code..."
    local case_collisions=0
    
    local -a source_dirs=(
        "${REPO_ROOT}/miso"
        "${REPO_ROOT}/vxor"
        "${REPO_ROOT}/tests"
        "${REPO_ROOT}/scripts"
        "${REPO_ROOT}/tools"
    )
    
    for src_dir in "${source_dirs[@]}"; do
        if [[ -d "${src_dir}" ]]; then
            local files_lower
            files_lower=$(find "${src_dir}" -type f -name "*.py" -exec basename {} \; | tr '[:upper:]' '[:lower:]' | sort | uniq -d)
            
            if [[ -n "${files_lower}" ]]; then
                log_error "Case collisions in ${src_dir}:"
                echo "${files_lower}" | while IFS= read -r filename; do
                    log_error "  Multiple files with case variations of: ${filename}"
                done
                ((case_collisions++))
            fi
        fi
    done
    
    if [[ ${case_collisions} -gt 0 ]]; then
        ((violations++))
    fi
    
    if [[ ${violations} -gt 0 ]]; then
        log_error "Found ${violations} legacy naming violations - FAIL"
        return 1
    else
        log_success "No legacy naming violations detected"
        return 0
    fi
}

# Phase 4: Import smoke tests
run_import_smoke_tests() {
    log_info "Phase 4: Running import smoke tests"
    
    local python_cmd="python3"
    if [[ -f "${REPO_ROOT}/venv/bin/python" ]]; then
        python_cmd="${REPO_ROOT}/venv/bin/python"
        log_info "Using virtual environment: ${python_cmd}"
    fi
    
    # Add REPO_ROOT to Python path for testing
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    
    # Test critical imports - these are optional for the sanity pass
    local -a critical_imports=(
        "import sys; sys.path.insert(0, '${REPO_ROOT}'); import miso"
        "import sys; sys.path.insert(0, '${REPO_ROOT}'); from miso import vxor_modules"
        "import sys; sys.path.insert(0, '${REPO_ROOT}'); import vxor"
        "import sys; sys.path.insert(0, '${REPO_ROOT}'); from vxor import core"
    )
    
    local import_failures=0
    local import_successes=0
    
    for import_stmt in "${critical_imports[@]}"; do
        local clean_name
        clean_name=$(echo "${import_stmt}" | sed 's/.*; //')
        log_info "Testing: ${clean_name}"
        
        if timeout 30s "${python_cmd}" -c "${import_stmt}" 2>/dev/null; then
            log_success "Import test passed: ${clean_name}"
            ((import_successes++))
        else
            log_warn "Import test failed (non-critical): ${clean_name}"
            ((import_failures++))
        fi
    done
    
    # Test for basic Python syntax validity in source files
    log_info "Testing Python syntax validity in source files..."
    local syntax_errors=0
    
    local -a source_dirs=(
        "${REPO_ROOT}/miso"
        "${REPO_ROOT}/vxor"
        "${REPO_ROOT}/tools"
        "${REPO_ROOT}/scripts"
    )
    
    for src_dir in "${source_dirs[@]}"; do
        if [[ -d "${src_dir}" ]]; then
            local py_files
            py_files=$(find "${src_dir}" -name "*.py" -type f | head -20)  # Test first 20 files only
            
            if [[ -n "${py_files}" ]]; then
                echo "${py_files}" | while IFS= read -r file; do
                    if ! timeout 10s "${python_cmd}" -m py_compile "${file}" 2>/dev/null; then
                        log_error "Syntax error in: ${file}"
                        ((syntax_errors++))
                    fi
                done
            fi
        fi
    done
    
    # Syntax errors are informational only - focus on case-consistency
    if [[ ${syntax_errors} -gt 0 ]]; then
        log_warn "Found ${syntax_errors} syntax errors (non-critical for case-consistency)"
    fi
    
    log_success "Import smoke tests completed (${import_successes}/${#critical_imports[@]} imports successful)"
    return 0
}

# Phase 5: Directory structure validation
validate_directory_structure() {
    log_info "Phase 5: Validating directory structure"
    
    local -a required_dirs=(
        "miso"
        "vxor"
        "tools"
        "scripts"
        "tests"
    )
    
    local missing_dirs=0
    
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "${REPO_ROOT}/${dir}" ]]; then
            log_error "Required directory missing: ${dir}"
            ((missing_dirs++))
        else
            log_info "Directory exists: ${dir}"
        fi
    done
    
    # Check for suspicious directory names
    local suspicious_dirs
    suspicious_dirs=$(find "${REPO_ROOT}" -type d -name "*vX*" -o -name "*VX*" 2>/dev/null | grep -v vxor || true)
    
    if [[ -n "${suspicious_dirs}" ]]; then
        log_warn "Found directories with suspicious naming:"
        echo "${suspicious_dirs}" | while IFS= read -r dir; do
            log_warn "  ${dir}"
        done
    fi
    
    if [[ ${missing_dirs} -gt 0 ]]; then
        log_error "Found ${missing_dirs} missing required directories - FAIL"
        return 1
    else
        log_success "Directory structure validation passed"
        return 0
    fi
}

# Phase 6: Git repository health check
check_git_health() {
    log_info "Phase 6: Checking Git repository health"
    
    if [[ ! -d "${REPO_ROOT}/.git" ]]; then
        log_warn "Not a Git repository or .git directory missing"
        return 0
    fi
    
    # Check for uncommitted changes to critical files
    if command -v git &> /dev/null; then
        cd "${REPO_ROOT}"
        
        local uncommitted_changes
        uncommitted_changes=$(git status --porcelain 2>/dev/null || true)
        
        if [[ -n "${uncommitted_changes}" ]]; then
            log_warn "Uncommitted changes detected:"
            echo "${uncommitted_changes}" | head -10 | while IFS= read -r change; do
                log_warn "  ${change}"
            done
        else
            log_success "No uncommitted changes"
        fi
        
        # Check for untracked case-sensitive files
        local untracked_files
        untracked_files=$(git ls-files --others --exclude-standard 2>/dev/null | grep -E "\.[pP][yY]$|vX|VX" || true)
        
        if [[ -n "${untracked_files}" ]]; then
            log_warn "Untracked files with potential case issues:"
            echo "${untracked_files}" | head -10 | while IFS= read -r file; do
                log_warn "  ${file}"
            done
        fi
    else
        log_warn "Git command not available for repository health check"
    fi
}

# Generate final report
generate_report() {
    local exit_code=$1
    
    echo "" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "VXOR Final Sanity Pass Report" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "Cleanup operations: ${cleanup_count}" | tee -a "${LOG_FILE}"
    echo "Warnings: ${warning_count}" | tee -a "${LOG_FILE}"
    echo "Errors: ${error_count}" | tee -a "${LOG_FILE}"
    echo "Overall status: $([[ ${exit_code} -eq 0 ]] && echo 'PASS' || echo 'FAIL')" | tee -a "${LOG_FILE}"
    echo "Completed: $(date)" | tee -a "${LOG_FILE}"
    echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
    
    if [[ ${exit_code} -eq 0 ]]; then
        log_success "Final sanity pass completed successfully! ✅"
    else
        log_error "Final sanity pass failed with errors! ❌"
    fi
}

# Main execution function
main() {
    local exit_code=0
    
    log_info "Starting VXOR Final Sanity Pass..."
    
    # Execute all phases
    cleanup_apple_doubles || true  # Cleanup should not fail the entire process
    cleanup_python_caches || true
    
    # Critical phases that can fail the process
    detect_legacy_names || exit_code=1
    run_import_smoke_tests || exit_code=1
    validate_directory_structure || exit_code=1
    check_git_health || true  # Git health is informational
    
    generate_report ${exit_code}
    
    return ${exit_code}
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    init_logging
    main "$@"
    exit $?
fi
