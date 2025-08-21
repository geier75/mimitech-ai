#!/usr/bin/env python3
"""
Access Control - Phase 13
CI/CD access control and read-only enforcement for datasets
"""

import os
import stat
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess

class AccessLevel(Enum):
    """Access levels for different contexts"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    NO_ACCESS = "no_access"

@dataclass 
class AccessPolicy:
    """Access policy configuration"""
    name: str
    paths: List[str]
    ci_access_level: AccessLevel
    local_access_level: AccessLevel
    enforce_permissions: bool = True
    allowed_operations: List[str] = None

class AccessController:
    """
    Manages access control for datasets and sensitive files
    Phase 13: Daten-Governance & Compliance
    """
    
    def __init__(self, policies_config: Dict[str, Any] = None):
        self.is_ci_environment = self._detect_ci_environment()
        self.policies = self._create_default_policies()
        
        if policies_config:
            self._load_policies_from_config(policies_config)
    
    def enforce_access_policies(self, base_dir: Path = None) -> Dict[str, bool]:
        """
        Enforce access policies on configured paths
        
        Args:
            base_dir: Base directory for relative paths
            
        Returns:
            Dictionary mapping policy names to enforcement success
        """
        
        if base_dir is None:
            base_dir = Path.cwd()
        
        results = {}
        
        for policy_name, policy in self.policies.items():
            try:
                success = self._enforce_single_policy(policy, base_dir)
                results[policy_name] = success
                
                if success:
                    print(f"✅ Policy enforced: {policy_name}")
                else:
                    print(f"❌ Failed to enforce policy: {policy_name}")
                    
            except Exception as e:
                results[policy_name] = False
                print(f"❌ Error enforcing policy {policy_name}: {e}")
        
        return results
    
    def verify_access_compliance(self, base_dir: Path = None) -> Dict[str, Dict[str, Any]]:
        """
        Verify current access permissions comply with policies
        
        Args:
            base_dir: Base directory for relative paths
            
        Returns:
            Detailed compliance report per policy
        """
        
        if base_dir is None:
            base_dir = Path.cwd()
        
        compliance_report = {}
        
        for policy_name, policy in self.policies.items():
            policy_report = {
                "policy": asdict(policy),
                "compliant": True,
                "violations": [],
                "checked_paths": []
            }
            
            # Get expected access level for current environment
            expected_level = policy.ci_access_level if self.is_ci_environment else policy.local_access_level
            
            for path_pattern in policy.paths:
                for path in base_dir.glob(path_pattern):
                    if path.exists():
                        policy_report["checked_paths"].append(str(path))
                        
                        # Check if current permissions match expected level
                        current_perms = self._get_path_permissions(path)
                        is_compliant = self._check_permission_compliance(
                            current_perms, expected_level, path
                        )
                        
                        if not is_compliant:
                            policy_report["compliant"] = False
                            policy_report["violations"].append({
                                "path": str(path),
                                "expected": expected_level.value,
                                "actual": current_perms
                            })
            
            compliance_report[policy_name] = policy_report
        
        return compliance_report
    
    def setup_ci_permissions(self, base_dir: Path = None) -> bool:
        """
        Setup read-only permissions for CI environment
        
        Args:
            base_dir: Base directory for operations
            
        Returns:
            True if setup successful
        """
        
        if not self.is_ci_environment:
            print("⚠️ Not in CI environment, skipping CI permission setup")
            return True
        
        if base_dir is None:
            base_dir = Path.cwd()
        
        print("🔒 Setting up read-only permissions for CI environment...")
        
        success_count = 0
        total_count = 0
        
        # Set datasets to read-only
        datasets_dir = base_dir / "datasets"
        if datasets_dir.exists():
            for dataset_path in datasets_dir.iterdir():
                total_count += 1
                try:
                    self._set_readonly_recursive(dataset_path)
                    success_count += 1
                    print(f"  🔒 Read-only: {dataset_path.name}")
                except Exception as e:
                    print(f"  ❌ Failed to set read-only: {dataset_path.name} - {e}")
        
        # Set baseline directory to read-only if it exists
        baseline_dir = base_dir / "baseline"
        if baseline_dir.exists():
            total_count += 1
            try:
                self._set_readonly_recursive(baseline_dir)
                success_count += 1
                print(f"  🔒 Read-only: baseline/")
            except Exception as e:
                print(f"  ❌ Failed to set read-only: baseline/ - {e}")
        
        print(f"📊 CI permissions setup: {success_count}/{total_count} successful")
        return success_count == total_count
    
    def _detect_ci_environment(self) -> bool:
        """Detect if running in CI/CD environment"""
        
        ci_indicators = [
            "CI",
            "CONTINUOUS_INTEGRATION", 
            "GITHUB_ACTIONS",
            "JENKINS_URL",
            "GITLAB_CI",
            "TRAVIS",
            "CIRCLECI"
        ]
        
        return any(os.getenv(indicator) for indicator in ci_indicators)
    
    def _enforce_single_policy(self, policy: AccessPolicy, base_dir: Path) -> bool:
        """Enforce a single access policy"""
        
        if not policy.enforce_permissions:
            return True
        
        expected_level = policy.ci_access_level if self.is_ci_environment else policy.local_access_level
        
        success = True
        
        for path_pattern in policy.paths:
            for path in base_dir.glob(path_pattern):
                if path.exists():
                    try:
                        self._set_path_permissions(path, expected_level)
                    except Exception as e:
                        print(f"  ❌ Failed to set permissions on {path}: {e}")
                        success = False
        
        return success
    
    def _set_path_permissions(self, path: Path, access_level: AccessLevel):
        """Set permissions on a path according to access level"""
        
        if access_level == AccessLevel.READ_ONLY:
            self._set_readonly_recursive(path)
        elif access_level == AccessLevel.READ_WRITE:
            self._set_readwrite_recursive(path)
        elif access_level == AccessLevel.NO_ACCESS:
            self._set_no_access_recursive(path)
    
    def _set_readonly_recursive(self, path: Path):
        """Set path to read-only recursively"""
        
        if path.is_file():
            # Remove write permissions for owner, group, others
            current_mode = path.stat().st_mode
            readonly_mode = current_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            path.chmod(readonly_mode)
        elif path.is_dir():
            # Set directory permissions (read + execute, no write)
            current_mode = path.stat().st_mode
            readonly_mode = (current_mode | stat.S_IRUSR | stat.S_IXUSR) & ~stat.S_IWUSR
            path.chmod(readonly_mode)
            
            # Recursively set permissions on contents
            for child in path.iterdir():
                self._set_readonly_recursive(child)
    
    def _set_readwrite_recursive(self, path: Path):
        """Set path to read-write recursively"""
        
        if path.is_file():
            # Add read/write permissions for owner
            current_mode = path.stat().st_mode
            readwrite_mode = current_mode | stat.S_IRUSR | stat.S_IWUSR
            path.chmod(readwrite_mode)
        elif path.is_dir():
            # Set directory permissions (read + write + execute)
            current_mode = path.stat().st_mode
            readwrite_mode = current_mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
            path.chmod(readwrite_mode)
            
            # Recursively set permissions on contents
            for child in path.iterdir():
                self._set_readwrite_recursive(child)
    
    def _set_no_access_recursive(self, path: Path):
        """Remove all permissions from path"""
        
        # Remove all permissions (dangerous operation - use with caution)
        path.chmod(0o000)
        
        if path.is_dir():
            for child in path.iterdir():
                self._set_no_access_recursive(child)
    
    def _get_path_permissions(self, path: Path) -> Dict[str, bool]:
        """Get current permissions for a path"""
        
        try:
            mode = path.stat().st_mode
            return {
                "owner_read": bool(mode & stat.S_IRUSR),
                "owner_write": bool(mode & stat.S_IWUSR),
                "owner_execute": bool(mode & stat.S_IXUSR),
                "group_read": bool(mode & stat.S_IRGRP),
                "group_write": bool(mode & stat.S_IWGRP),
                "group_execute": bool(mode & stat.S_IXGRP),
                "other_read": bool(mode & stat.S_IROTH),
                "other_write": bool(mode & stat.S_IWOTH),
                "other_execute": bool(mode & stat.S_IXOTH)
            }
        except OSError:
            return {}
    
    def _check_permission_compliance(self, current_perms: Dict[str, bool], 
                                   expected_level: AccessLevel, path: Path) -> bool:
        """Check if current permissions comply with expected access level"""
        
        if expected_level == AccessLevel.READ_ONLY:
            # Should have read but not write permissions
            return (current_perms.get("owner_read", False) and 
                   not current_perms.get("owner_write", False))
        
        elif expected_level == AccessLevel.READ_WRITE:
            # Should have both read and write permissions
            return (current_perms.get("owner_read", False) and 
                   current_perms.get("owner_write", False))
        
        elif expected_level == AccessLevel.NO_ACCESS:
            # Should have no permissions
            return not any(current_perms.values())
        
        return True
    
    def _create_default_policies(self) -> Dict[str, AccessPolicy]:
        """Create default access policies for MISO"""
        
        policies = {
            "datasets_readonly": AccessPolicy(
                name="datasets_readonly",
                paths=["datasets/*", "datasets/**/*"],
                ci_access_level=AccessLevel.READ_ONLY,
                local_access_level=AccessLevel.READ_WRITE,
                enforce_permissions=True,
                allowed_operations=["read", "checksum"]
            ),
            "baseline_readonly": AccessPolicy(
                name="baseline_readonly", 
                paths=["baseline/*", "baseline/**/*"],
                ci_access_level=AccessLevel.READ_ONLY,
                local_access_level=AccessLevel.READ_WRITE,
                enforce_permissions=True,
                allowed_operations=["read", "verify"]
            ),
            "schemas_readonly": AccessPolicy(
                name="schemas_readonly",
                paths=["schemas/*.json"],
                ci_access_level=AccessLevel.READ_ONLY,
                local_access_level=AccessLevel.READ_WRITE,
                enforce_permissions=True,
                allowed_operations=["read", "validate"]
            ),
            "artifacts_readwrite": AccessPolicy(
                name="artifacts_readwrite",
                paths=["*.json", "logs/*", "reports/*", "signatures/*"],
                ci_access_level=AccessLevel.READ_WRITE,
                local_access_level=AccessLevel.READ_WRITE,
                enforce_permissions=False,  # Allow CI to create artifacts
                allowed_operations=["read", "write", "create"]
            )
        }
        
        return policies
    
    def _load_policies_from_config(self, config: Dict[str, Any]):
        """Load additional policies from configuration"""
        
        for policy_name, policy_config in config.items():
            try:
                # Convert string access levels to enums
                policy_config["ci_access_level"] = AccessLevel(policy_config["ci_access_level"])
                policy_config["local_access_level"] = AccessLevel(policy_config["local_access_level"])
                
                self.policies[policy_name] = AccessPolicy(**policy_config)
                
            except (ValueError, TypeError, KeyError) as e:
                print(f"⚠️ Failed to load policy {policy_name}: {e}")
    
    def generate_access_report(self, base_dir: Path = None) -> Dict[str, Any]:
        """Generate comprehensive access control report"""
        
        if base_dir is None:
            base_dir = Path.cwd()
        
        report = {
            "generated_at": os.getenv("CI_BUILD_TIME", "local"),
            "environment": "ci" if self.is_ci_environment else "local",
            "base_directory": str(base_dir),
            "compliance_summary": {},
            "violations": [],
            "recommendations": []
        }
        
        # Get compliance status
        compliance = self.verify_access_compliance(base_dir)
        
        compliant_policies = 0
        total_policies = len(compliance)
        
        for policy_name, policy_report in compliance.items():
            report["compliance_summary"][policy_name] = policy_report["compliant"]
            
            if policy_report["compliant"]:
                compliant_policies += 1
            else:
                report["violations"].extend(policy_report["violations"])
        
        # Add recommendations
        if report["violations"]:
            if self.is_ci_environment:
                report["recommendations"].append(
                    "Run 'python scripts/setup_ci_permissions.py' to fix CI permissions"
                )
            else:
                report["recommendations"].append(
                    "Run access controller enforcement to fix local permissions"
                )
        
        report["overall_compliance"] = compliant_policies / total_policies if total_policies > 0 else 1.0
        
        return report
