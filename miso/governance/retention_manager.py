#!/usr/bin/env python3
"""
Retention Manager - Phase 13
Automated data retention and cleanup policies
"""

import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class RetentionAction(Enum):
    """Actions to take when retention period expires"""
    DELETE = "delete"
    ARCHIVE = "archive"
    COMPRESS = "compress"
    MOVE = "move"

@dataclass
class RetentionPolicy:
    """Data retention policy configuration"""
    name: str
    file_patterns: List[str]
    retention_days: int
    action: RetentionAction
    archive_location: Optional[str] = None
    grace_period_days: int = 7
    enabled: bool = True

class RetentionManager:
    """
    Manages data retention policies and automated cleanup
    Phase 13: Daten-Governance & Compliance
    """
    
    def __init__(self, policies_file: Path = None):
        self.policies_file = policies_file or Path("retention_policies.json")
        self.policies = self._load_policies()
        
        # Default retention policies
        self._create_default_policies()
    
    def add_policy(self, policy: RetentionPolicy):
        """Add a new retention policy"""
        self.policies[policy.name] = policy
        self._save_policies()
        print(f"✅ Retention policy added: {policy.name}")
    
    def remove_policy(self, policy_name: str) -> bool:
        """Remove a retention policy"""
        if policy_name in self.policies:
            del self.policies[policy_name]
            self._save_policies()
            print(f"🗑️ Retention policy removed: {policy_name}")
            return True
        return False
    
    def list_policies(self) -> List[RetentionPolicy]:
        """List all retention policies"""
        return list(self.policies.values())
    
    def enforce_retention(self, base_dir: Path = None, dry_run: bool = False) -> Dict[str, int]:
        """
        Enforce retention policies on files
        
        Args:
            base_dir: Base directory to scan (default: current)
            dry_run: If True, only report what would be done
            
        Returns:
            Dictionary with counts of actions taken per policy
        """
        
        if base_dir is None:
            base_dir = Path.cwd()
        
        results = {}
        now = datetime.now()
        
        for policy_name, policy in self.policies.items():
            if not policy.enabled:
                continue
            
            action_count = 0
            
            print(f"🔍 Enforcing policy: {policy_name}")
            
            # Find files matching patterns
            for pattern in policy.file_patterns:
                for file_path in base_dir.rglob(pattern):
                    if self._should_process_file(file_path, policy, now):
                        
                        if dry_run:
                            print(f"  [DRY RUN] Would {policy.action.value}: {file_path}")
                            action_count += 1
                        else:
                            try:
                                if self._process_file(file_path, policy):
                                    action_count += 1
                            except Exception as e:
                                print(f"❌ Failed to process {file_path}: {e}")
            
            results[policy_name] = action_count
            
            if action_count > 0:
                print(f"  📊 {action_count} files processed")
            else:
                print(f"  ✅ No files require processing")
        
        return results
    
    def _should_process_file(self, file_path: Path, policy: RetentionPolicy, now: datetime) -> bool:
        """Check if file should be processed according to policy"""
        
        try:
            # Get file modification time
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            # Calculate age
            age = now - mtime
            retention_threshold = timedelta(days=policy.retention_days)
            
            # Check if file is older than retention period
            return age > retention_threshold
            
        except (OSError, ValueError):
            return False
    
    def _process_file(self, file_path: Path, policy: RetentionPolicy) -> bool:
        """Process file according to retention policy"""
        
        try:
            if policy.action == RetentionAction.DELETE:
                file_path.unlink()
                print(f"🗑️ Deleted: {file_path}")
                
            elif policy.action == RetentionAction.ARCHIVE:
                if policy.archive_location:
                    archive_dir = Path(policy.archive_location)
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Preserve directory structure in archive
                    relative_path = file_path.relative_to(Path.cwd())
                    archive_path = archive_dir / relative_path
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.move(str(file_path), str(archive_path))
                    print(f"📦 Archived: {file_path} → {archive_path}")
                else:
                    print(f"⚠️ No archive location specified for {file_path}")
                    return False
                    
            elif policy.action == RetentionAction.COMPRESS:
                # Compress file in place
                compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
                
                import gzip
                with open(file_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                file_path.unlink()  # Remove original
                print(f"🗜️ Compressed: {file_path} → {compressed_path}")
                
            elif policy.action == RetentionAction.MOVE:
                if policy.archive_location:
                    move_dir = Path(policy.archive_location)
                    move_dir.mkdir(parents=True, exist_ok=True)
                    
                    move_path = move_dir / file_path.name
                    shutil.move(str(file_path), str(move_path))
                    print(f"📁 Moved: {file_path} → {move_path}")
                else:
                    print(f"⚠️ No move location specified for {file_path}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to {policy.action.value} {file_path}: {e}")
            return False
    
    def _load_policies(self) -> Dict[str, RetentionPolicy]:
        """Load retention policies from file"""
        
        if not self.policies_file.exists():
            return {}
        
        try:
            with open(self.policies_file) as f:
                policies_data = json.load(f)
            
            policies = {}
            for name, data in policies_data.get("policies", {}).items():
                # Convert action string back to enum
                data["action"] = RetentionAction(data["action"])
                policies[name] = RetentionPolicy(**data)
            
            return policies
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"⚠️ Failed to load retention policies: {e}")
            return {}
    
    def _save_policies(self):
        """Save retention policies to file"""
        
        # Convert to JSON-serializable format
        policies_data = {}
        for name, policy in self.policies.items():
            data = asdict(policy)
            data["action"] = policy.action.value
            policies_data[name] = data
        
        # Add metadata
        policies_with_metadata = {
            "_metadata": {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "total_policies": len(self.policies)
            },
            "policies": policies_data
        }
        
        with open(self.policies_file, 'w') as f:
            json.dump(policies_with_metadata, f, indent=2)
    
    def _create_default_policies(self):
        """Create default retention policies for MISO"""
        
        default_policies = [
            RetentionPolicy(
                name="benchmark_reports",
                file_patterns=["benchmark_report_*.json", "reports/*.json"],
                retention_days=730,  # 2 years
                action=RetentionAction.ARCHIVE,
                archive_location="archive/reports"
            ),
            RetentionPolicy(
                name="structured_logs",
                file_patterns=["logs/*.jsonl", "*.jsonl"],
                retention_days=365,  # 1 year
                action=RetentionAction.COMPRESS,
                archive_location="archive/logs"
            ),
            RetentionPolicy(
                name="temporary_files",
                file_patterns=["tmp/*", "temp/*", "*.tmp"],
                retention_days=7,  # 1 week
                action=RetentionAction.DELETE
            ),
            RetentionPolicy(
                name="sbom_provenance",
                file_patterns=["miso_sbom_*.json", "build_provenance_*.json"],
                retention_days=1095,  # 3 years
                action=RetentionAction.ARCHIVE,
                archive_location="archive/compliance"
            ),
            RetentionPolicy(
                name="signatures",
                file_patterns=["signatures/*.sig", "signatures/*.miso_sig"],
                retention_days=-1,  # Never expire (indefinite retention)
                action=RetentionAction.ARCHIVE,
                archive_location="archive/signatures",
                enabled=False  # Disabled by default for indefinite retention
            ),
            RetentionPolicy(
                name="drift_reports", 
                file_patterns=["drift_report_*.json"],
                retention_days=180,  # 6 months
                action=RetentionAction.COMPRESS,
                archive_location="archive/drift"
            ),
            RetentionPolicy(
                name="cache_files",
                file_patterns=["__pycache__/*", "*.pyc", ".pytest_cache/*"],
                retention_days=1,  # Daily cleanup
                action=RetentionAction.DELETE
            )
        ]
        
        # Only add policies that don't already exist
        added_count = 0
        for policy in default_policies:
            if policy.name not in self.policies:
                self.policies[policy.name] = policy
                added_count += 1
        
        if added_count > 0:
            self._save_policies()
            print(f"📋 Created {added_count} default retention policies")
    
    def generate_retention_report(self, base_dir: Path = None) -> Dict[str, Any]:
        """Generate report on files affected by retention policies"""
        
        if base_dir is None:
            base_dir = Path.cwd()
        
        now = datetime.now()
        report = {
            "generated_at": now.isoformat(),
            "base_directory": str(base_dir),
            "policies": {}
        }
        
        for policy_name, policy in self.policies.items():
            if not policy.enabled:
                continue
            
            policy_report = {
                "retention_days": policy.retention_days,
                "action": policy.action.value,
                "files_to_process": [],
                "total_size_bytes": 0
            }
            
            # Scan for files that would be affected
            for pattern in policy.file_patterns:
                for file_path in base_dir.rglob(pattern):
                    if self._should_process_file(file_path, policy, now):
                        file_info = {
                            "path": str(file_path),
                            "size_bytes": file_path.stat().st_size,
                            "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        }
                        policy_report["files_to_process"].append(file_info)
                        policy_report["total_size_bytes"] += file_info["size_bytes"]
            
            report["policies"][policy_name] = policy_report
        
        return report
