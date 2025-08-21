"""
Schema version management and compatibility checking
Enforces semantic versioning and migration requirements
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VersionChangeType(Enum):
    """Types of schema version changes"""
    MAJOR = "major"    # Breaking changes
    MINOR = "minor"    # Backward compatible additions
    PATCH = "patch"    # Bug fixes, clarifications

@dataclass
class VersionInfo:
    """Semantic version information"""
    major: int
    minor: int
    patch: int
    
    @classmethod
    def from_string(cls, version_str: str) -> 'VersionInfo':
        """Parse version string like 'v1.2.3'"""
        match = re.match(r'^v?(\d+)\.(\d+)\.(\d+)$', version_str)
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3))
        )
    
    def to_string(self) -> str:
        """Convert to version string"""
        return f"v{self.major}.{self.minor}.{self.patch}"
    
    def compare(self, other: 'VersionInfo') -> int:
        """Compare versions: -1 if self < other, 0 if equal, 1 if self > other"""
        if self.major != other.major:
            return -1 if self.major < other.major else 1
        if self.minor != other.minor:
            return -1 if self.minor < other.minor else 1
        if self.patch != other.patch:
            return -1 if self.patch < other.patch else 1
        return 0
    
    def is_compatible_with(self, other: 'VersionInfo') -> bool:
        """Check if versions are compatible (same major version)"""
        return self.major == other.major

@dataclass
class SchemaChange:
    """Documentation of a schema change"""
    version: str
    change_type: VersionChangeType
    description: str
    migration_required: bool
    breaking_changes: List[str]
    changelog_entry: Optional[str] = None

class SchemaVersionManager:
    """Manages schema versions and enforces compatibility"""
    
    # Current supported schema version
    CURRENT_VERSION = "v1.0.0"
    
    # Minimum supported version for backward compatibility
    MIN_SUPPORTED_VERSION = "v1.0.0"
    
    def __init__(self, schemas_dir: Path = None):
        """Initialize version manager"""
        if schemas_dir is None:
            project_root = Path(__file__).parent.parent.parent
            schemas_dir = project_root / "schemas"
        
        self.schemas_dir = schemas_dir
        self.version_history = self._load_version_history()
        
    def _load_version_history(self) -> Dict[str, SchemaChange]:
        """Load version history from changelog"""
        version_file = self.schemas_dir.parent / "SCHEMA_CHANGELOG.md"
        history = {}
        
        if not version_file.exists():
            logger.warning(f"Schema changelog not found: {version_file}")
            return history
        
        # Parse changelog for version entries
        # This is a simplified parser - in practice you'd want more robust parsing
        content = version_file.read_text()
        version_blocks = re.findall(r'## (v\d+\.\d+\.\d+).*?\n(.*?)(?=\n## |\n*$)', content, re.DOTALL)
        
        for version_str, description in version_blocks:
            change_type = VersionChangeType.PATCH
            if "BREAKING" in description.upper() or "breaking" in description:
                change_type = VersionChangeType.MAJOR
            elif "ADD" in description.upper() or "new" in description.lower():
                change_type = VersionChangeType.MINOR
            
            breaking_changes = []
            if change_type == VersionChangeType.MAJOR:
                breaking_changes = [line.strip() for line in description.split('\n') 
                                  if 'breaking' in line.lower() or 'removed' in line.lower()]
            
            history[version_str] = SchemaChange(
                version=version_str,
                change_type=change_type,
                description=description.strip(),
                migration_required=change_type == VersionChangeType.MAJOR,
                breaking_changes=breaking_changes,
                changelog_entry=description.strip()
            )
        
        return history
    
    def validate_schema_version(self, reported_version: str) -> Tuple[bool, str]:
        """Validate that schema version in report is supported"""
        try:
            reported = VersionInfo.from_string(reported_version)
            current = VersionInfo.from_string(self.CURRENT_VERSION)
            min_supported = VersionInfo.from_string(self.MIN_SUPPORTED_VERSION)
            
            # Check if version is too old
            if reported.compare(min_supported) < 0:
                return False, f"Schema version {reported_version} is too old. Minimum supported: {self.MIN_SUPPORTED_VERSION}"
            
            # Check if version is too new
            if reported.compare(current) > 0:
                return False, f"Schema version {reported_version} is newer than current: {self.CURRENT_VERSION}"
            
            # Check compatibility
            if not reported.is_compatible_with(current):
                return False, f"Schema version {reported_version} is incompatible with current: {self.CURRENT_VERSION}"
            
            return True, f"Schema version {reported_version} is valid and compatible"
            
        except ValueError as e:
            return False, f"Invalid schema version format: {e}"
    
    def check_migration_requirements(self, from_version: str, to_version: str) -> List[str]:
        """Check what migrations are required between versions"""
        try:
            from_ver = VersionInfo.from_string(from_version)
            to_ver = VersionInfo.from_string(to_version)
        except ValueError as e:
            return [f"Invalid version format: {e}"]
        
        if from_ver.compare(to_ver) >= 0:
            return []  # No migration needed for same or older version
        
        required_migrations = []
        
        # Check each version between from and to
        for version_str, change in self.version_history.items():
            try:
                change_ver = VersionInfo.from_string(version_str)
                
                # If this change is between from and to versions
                if (change_ver.compare(from_ver) > 0 and 
                    change_ver.compare(to_ver) <= 0 and
                    change.migration_required):
                    
                    required_migrations.append(
                        f"Migration required for {version_str}: {change.description}"
                    )
                    
            except ValueError:
                continue
        
        return required_migrations
    
    def enforce_release_gate(self, proposed_changes: List[str]) -> Tuple[bool, List[str]]:
        """Enforce release gate requirements for schema changes"""
        errors = []
        
        # Check if changelog exists and is updated
        changelog_path = self.schemas_dir.parent / "SCHEMA_CHANGELOG.md"
        if not changelog_path.exists():
            errors.append("SCHEMA_CHANGELOG.md is required but missing")
        
        # Check if migration guide exists for major changes
        migration_guide_path = self.schemas_dir.parent / "SCHEMA_MIGRATION_GUIDE.md"
        has_breaking_changes = any("breaking" in change.lower() or "remove" in change.lower() 
                                 for change in proposed_changes)
        
        if has_breaking_changes and not migration_guide_path.exists():
            errors.append("SCHEMA_MIGRATION_GUIDE.md is required for breaking changes but missing")
        
        # Check if version was bumped appropriately
        current_version = VersionInfo.from_string(self.CURRENT_VERSION)
        
        # Simulate version bump validation
        if has_breaking_changes:
            # Major version bump required
            required_version = f"v{current_version.major + 1}.0.0"
            errors.append(f"Breaking changes detected. Major version bump required: {required_version}")
        elif any("add" in change.lower() or "new" in change.lower() for change in proposed_changes):
            # Minor version bump required
            required_version = f"v{current_version.major}.{current_version.minor + 1}.0"
            errors.append(f"New features detected. Minor version bump required: {required_version}")
        
        return len(errors) == 0, errors
    
    def generate_compatibility_matrix(self) -> Dict[str, Any]:
        """Generate version compatibility matrix"""
        current = VersionInfo.from_string(self.CURRENT_VERSION)
        min_supported = VersionInfo.from_string(self.MIN_SUPPORTED_VERSION)
        
        compatible_versions = []
        deprecated_versions = []
        unsupported_versions = []
        
        for version_str in self.version_history.keys():
            try:
                version = VersionInfo.from_string(version_str)
                
                if version.is_compatible_with(current) and version.compare(min_supported) >= 0:
                    compatible_versions.append(version_str)
                elif version.compare(min_supported) < 0:
                    unsupported_versions.append(version_str)
                else:
                    deprecated_versions.append(version_str)
                    
            except ValueError:
                unsupported_versions.append(version_str)
        
        return {
            "current_version": self.CURRENT_VERSION,
            "min_supported_version": self.MIN_SUPPORTED_VERSION,
            "compatible_versions": sorted(compatible_versions),
            "deprecated_versions": sorted(deprecated_versions),
            "unsupported_versions": sorted(unsupported_versions),
            "migration_paths": self._generate_migration_paths()
        }
    
    def _generate_migration_paths(self) -> Dict[str, List[str]]:
        """Generate migration paths between versions"""
        migration_paths = {}
        
        for from_version in self.version_history.keys():
            for to_version in self.version_history.keys():
                if from_version != to_version:
                    migrations = self.check_migration_requirements(from_version, to_version)
                    if migrations:
                        migration_paths[f"{from_version} -> {to_version}"] = migrations
        
        return migration_paths
    
    def validate_schema_files(self) -> List[str]:
        """Validate that all schema files have consistent versions"""
        errors = []
        schema_files = list(self.schemas_dir.glob("*.schema.json"))
        
        for schema_file in schema_files:
            try:
                with open(schema_file) as f:
                    schema = json.load(f)
                
                # Check for version consistency
                schema_id = schema.get("$id", "")
                if schema_id:
                    # Extract version from $id (e.g., "https://miso.ai/schemas/benchmark_report/v1.0.0")
                    version_match = re.search(r'/v(\d+\.\d+\.\d+)$', schema_id)
                    if version_match:
                        file_version = f"v{version_match.group(1)}"
                        if file_version != self.CURRENT_VERSION:
                            errors.append(f"{schema_file.name}: Version mismatch. Found {file_version}, expected {self.CURRENT_VERSION}")
                    else:
                        errors.append(f"{schema_file.name}: No version found in $id field")
                else:
                    errors.append(f"{schema_file.name}: Missing $id field")
                    
            except Exception as e:
                errors.append(f"{schema_file.name}: Failed to parse - {e}")
        
        return errors
    
    def create_version_report(self) -> Dict[str, Any]:
        """Create comprehensive version status report"""
        validation_errors = self.validate_schema_files()
        compatibility_matrix = self.generate_compatibility_matrix()
        
        return {
            "schema_version_status": {
                "current_version": self.CURRENT_VERSION,
                "min_supported_version": self.MIN_SUPPORTED_VERSION,
                "schema_files_valid": len(validation_errors) == 0,
                "validation_errors": validation_errors
            },
            "compatibility_matrix": compatibility_matrix,
            "version_history_count": len(self.version_history),
            "release_gate_status": {
                "changelog_exists": (self.schemas_dir.parent / "SCHEMA_CHANGELOG.md").exists(),
                "migration_guide_exists": (self.schemas_dir.parent / "SCHEMA_MIGRATION_GUIDE.md").exists()
            }
        }
