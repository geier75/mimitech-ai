#!/usr/bin/env python3
"""
T0: Freeze UNTRAINED_BASELINE
Captures current model state as immutable reference for training comparisons
"""

import json
import subprocess
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys

class UntrainedBaselineFreezer:
    """
    Creates immutable UNTRAINED_BASELINE reference for training progression tracking
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.baseline_dir = self.project_root / "baseline" / "untrained"
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
    def freeze_baseline(self) -> Dict[str, Any]:
        """
        Create comprehensive UNTRAINED_BASELINE freeze
        
        Returns:
            Dict with baseline metadata and artifact locations
        """
        
        print("🧊 Freezing UNTRAINED_BASELINE...")
        print("=" * 60)
        
        # Get current git state
        git_info = self._capture_git_state()
        
        # Capture evaluation artifacts
        eval_artifacts = self._capture_evaluation_artifacts()
        
        # Capture model/system state  
        system_state = self._capture_system_state()
        
        # Capture dataset manifests
        dataset_state = self._capture_dataset_state()
        
        # Create baseline manifest
        baseline_manifest = {
            "baseline_id": "UNTRAINED_BASELINE",
            "created_at": datetime.now().isoformat(),
            "git_info": git_info,
            "system_state": system_state,
            "evaluation_artifacts": eval_artifacts,
            "dataset_state": dataset_state,
            "artifact_hashes": {},
            "notes": "Initial untrained baseline before any training interventions"
        }
        
        # Hash all artifacts
        baseline_manifest["artifact_hashes"] = self._hash_all_artifacts()
        
        # Save baseline manifest
        manifest_path = self.baseline_dir / "baseline_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(baseline_manifest, f, indent=2)
        
        print(f"\n✅ UNTRAINED_BASELINE frozen successfully")
        print(f"📁 Artifacts saved to: {self.baseline_dir}")
        print(f"📋 Manifest: {manifest_path}")
        
        return baseline_manifest
    
    def _capture_git_state(self) -> Dict[str, str]:
        """Capture current git commit and state"""
        
        try:
            # Get commit hash
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                cwd=self.project_root,
                text=True
            ).strip()
            
            # Get branch name
            branch = subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=self.project_root, 
                text=True
            ).strip()
            
            # Check if working directory is clean
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                text=True
            ).strip()
            
            is_clean = len(status) == 0
            
            # Get last commit message
            commit_msg = subprocess.check_output(
                ["git", "log", "-1", "--pretty=%B"],
                cwd=self.project_root,
                text=True
            ).strip()
            
            print(f"📌 Git State: {commit_hash[:8]} on {branch} {'(clean)' if is_clean else '(dirty)'}")
            
            return {
                "commit_hash": commit_hash,
                "branch": branch,
                "is_clean": is_clean,
                "commit_message": commit_msg,
                "uncommitted_changes": status if not is_clean else None
            }
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Git state capture failed: {e}")
            return {"error": str(e)}
    
    def _capture_evaluation_artifacts(self) -> Dict[str, Any]:
        """Copy current evaluation reports and logs"""
        
        artifacts = {
            "benchmark_reports": [],
            "structured_logs": [],
            "summary_reports": []
        }
        
        # Copy benchmark reports
        reports_dir = self.project_root / "tests" / "reports"
        if reports_dir.exists():
            baseline_reports_dir = self.baseline_dir / "reports"
            baseline_reports_dir.mkdir(exist_ok=True)
            
            # Copy JSON reports
            for report_file in reports_dir.rglob("*.json"):
                rel_path = report_file.relative_to(reports_dir)
                dest_path = baseline_reports_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(report_file, dest_path)
                artifacts["benchmark_reports"].append(str(rel_path))
            
            print(f"📊 Copied {len(artifacts['benchmark_reports'])} benchmark reports")
        
        # Copy structured logs
        logs_dir = self.project_root / "logs"
        if logs_dir.exists():
            baseline_logs_dir = self.baseline_dir / "logs"
            baseline_logs_dir.mkdir(exist_ok=True)
            
            for log_file in logs_dir.rglob("*.jsonl"):
                rel_path = log_file.relative_to(logs_dir)
                dest_path = baseline_logs_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(log_file, dest_path)
                artifacts["structured_logs"].append(str(rel_path))
                
            print(f"📝 Copied {len(artifacts['structured_logs'])} log files")
        
        # Copy summary reports  
        for summary_file in self.project_root.glob("SUMMARY*.md"):
            dest_path = self.baseline_dir / summary_file.name
            shutil.copy2(summary_file, dest_path)
            artifacts["summary_reports"].append(summary_file.name)
        
        return artifacts
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture system configuration and environment"""
        
        system_state = {
            "python_version": sys.version,
            "platform": sys.platform,
            "environment_variables": {}
        }
        
        # Capture reproducibility-critical env vars
        repro_env_vars = [
            "PYTHONHASHSEED",
            "OMP_NUM_THREADS", 
            "MKL_NUM_THREADS",
            "CUDA_VISIBLE_DEVICES"
        ]
        
        for var in repro_env_vars:
            system_state["environment_variables"][var] = os.getenv(var)
        
        # Capture package versions if requirements.txt exists
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            shutil.copy2(requirements_file, self.baseline_dir / "requirements.txt")
            system_state["requirements_captured"] = True
        
        print(f"🖥️ System state captured: Python {sys.version_info.major}.{sys.version_info.minor}")
        
        return system_state
    
    def _capture_dataset_state(self) -> Dict[str, Any]:
        """Capture dataset manifests and checksums"""
        
        dataset_state = {
            "manifests": [],
            "checksums": {}
        }
        
        datasets_dir = self.project_root / "datasets"
        if datasets_dir.exists():
            baseline_datasets_dir = self.baseline_dir / "datasets"
            baseline_datasets_dir.mkdir(exist_ok=True)
            
            # Copy all manifest files
            for manifest_file in datasets_dir.rglob("manifest.sha256"):
                rel_path = manifest_file.relative_to(datasets_dir)
                dest_path = baseline_datasets_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(manifest_file, dest_path)
                dataset_state["manifests"].append(str(rel_path))
            
            print(f"📂 Captured {len(dataset_state['manifests'])} dataset manifests")
        
        return dataset_state
    
    def _hash_all_artifacts(self) -> Dict[str, str]:
        """Generate SHA256 hashes for all baseline artifacts"""
        
        hashes = {}
        
        for file_path in self.baseline_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(self.baseline_dir)
                
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                
                hashes[str(rel_path)] = file_hash
        
        print(f"🔐 Generated hashes for {len(hashes)} artifacts")
        
        return hashes
    
    def verify_baseline_integrity(self) -> bool:
        """Verify baseline artifacts haven't been tampered with"""
        
        manifest_path = self.baseline_dir / "baseline_manifest.json"
        if not manifest_path.exists():
            print("❌ Baseline manifest not found")
            return False
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        original_hashes = manifest.get("artifact_hashes", {})
        current_hashes = self._hash_all_artifacts()
        
        # Compare hashes
        mismatches = []
        for file_path, original_hash in original_hashes.items():
            current_hash = current_hashes.get(file_path)
            if current_hash != original_hash:
                mismatches.append(file_path)
        
        if mismatches:
            print(f"❌ Integrity check failed: {len(mismatches)} files modified")
            for file_path in mismatches:
                print(f"  - {file_path}")
            return False
        
        print("✅ Baseline integrity verified - no modifications detected")
        return True

def main():
    import os
    
    freezer = UntrainedBaselineFreezer()
    
    print("🧊 MISO UNTRAINED_BASELINE Freezer")
    print("=" * 60)
    
    # Create baseline freeze
    baseline_manifest = freezer.freeze_baseline()
    
    # Verify integrity immediately
    print("\n🔍 Verifying baseline integrity...")
    integrity_ok = freezer.verify_baseline_integrity()
    
    print("\n" + "=" * 60)
    print("📋 BASELINE FREEZE SUMMARY")
    print("=" * 60)
    
    print(f"**Baseline ID**: {baseline_manifest['baseline_id']}")
    print(f"**Created**: {baseline_manifest['created_at']}")
    print(f"**Git Commit**: {baseline_manifest['git_info'].get('commit_hash', 'N/A')[:12]}")
    print(f"**Branch**: {baseline_manifest['git_info'].get('branch', 'N/A')}")
    print(f"**Artifacts**: {len(baseline_manifest['artifact_hashes'])} files")
    print(f"**Integrity**: {'✅ VERIFIED' if integrity_ok else '❌ FAILED'}")
    
    print("\n📁 Artifact Locations:")
    print(f"- Reports: baseline/untrained/reports/")
    print(f"- Logs: baseline/untrained/logs/")  
    print(f"- Datasets: baseline/untrained/datasets/")
    print(f"- Manifest: baseline/untrained/baseline_manifest.json")
    
    print("\n🔄 Next Steps:")
    print("1. Commit this baseline to git: `git add baseline/untrained/`")
    print("2. Tag the commit: `git tag UNTRAINED_BASELINE`")
    print("3. Begin training track T1: Define metric contract")
    print("4. Reference this baseline in all future A/B evaluations")
    
    if integrity_ok:
        print("\n🎉 UNTRAINED_BASELINE successfully frozen and verified!")
        sys.exit(0)
    else:
        print("\n❌ Baseline freeze failed integrity check!")
        sys.exit(1)

if __name__ == "__main__":
    main()
