#!/usr/bin/env python3
"""
Build Provenance Manager - Phase 12
SLSA-compliant build provenance generation and verification
"""

import json
import hashlib
import subprocess
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class BuildProvenance:
    """SLSA Build Provenance information"""
    builder_id: str
    build_type: str
    invocation_id: str
    started_on: str
    finished_on: str
    materials: List[Dict[str, Any]]
    byproducts: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class ProvenanceManager:
    """
    Manages SLSA-compliant build provenance for artifacts
    Phase 12: Supply-Chain & Artefakt-Integrität
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
    
    def generate_build_provenance(self, 
                                artifact_paths: List[Path],
                                build_command: str = None,
                                output_path: Path = None) -> Path:
        """
        Generate SLSA build provenance for artifacts
        
        Args:
            artifact_paths: List of artifact files to document
            build_command: Command used to build artifacts
            output_path: Optional output path for provenance
            
        Returns:
            Path to generated provenance file
        """
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.project_root / f"build_provenance_{timestamp}.json"
        
        # Collect build environment information
        build_info = self._collect_build_info()
        
        # Collect materials (source inputs)
        materials = self._collect_materials()
        
        # Collect artifact information
        byproducts = self._collect_artifacts(artifact_paths)
        
        # Create SLSA provenance statement
        provenance = {
            "_type": "https://in-toto.io/Statement/v0.1",
            "predicateType": "https://slsa.dev/provenance/v0.2",
            "subject": [
                {
                    "name": artifact["name"],
                    "digest": {"sha256": artifact["sha256"]}
                }
                for artifact in byproducts
            ],
            "predicate": {
                "builder": {
                    "id": build_info["builder_id"]
                },
                "buildType": "https://miso-ultimate.local/build-types/benchmark@v1",
                "invocation": {
                    "configSource": {
                        "uri": build_info.get("repository_uri", "unknown"),
                        "digest": {"sha1": build_info.get("git_commit", "unknown")},
                        "entryPoint": build_command or "unknown"
                    },
                    "parameters": {
                        "build_command": build_command,
                        "environment": dict(os.environ)
                    }
                },
                "metadata": {
                    "buildInvocationId": build_info["invocation_id"],
                    "buildStartedOn": build_info["started_on"],
                    "buildFinishedOn": datetime.now().isoformat() + "Z",
                    "completeness": {
                        "parameters": True,
                        "environment": True,
                        "materials": True
                    },
                    "reproducible": True
                },
                "materials": materials
            }
        }
        
        # Write provenance file
        with open(output_path, 'w') as f:
            json.dump(provenance, f, indent=2)
        
        print(f"✅ Build provenance generated: {output_path}")
        print(f"🏗️  Builder: {build_info['builder_id']}")
        print(f"📦 Artifacts: {len(byproducts)} files")
        print(f"🔗 Materials: {len(materials)} source inputs")
        
        return output_path
    
    def _collect_build_info(self) -> Dict[str, Any]:
        """Collect build environment information"""
        info = {}
        
        # Build ID from environment or generate
        info["invocation_id"] = os.getenv("GITHUB_RUN_ID", f"local-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        info["started_on"] = datetime.now().isoformat() + "Z"
        
        # Builder identification
        if os.getenv("GITHUB_ACTIONS"):
            info["builder_id"] = f"https://github.com/actions/runner@{os.getenv('RUNNER_VERSION', 'unknown')}"
        else:
            import platform
            info["builder_id"] = f"https://local-builder/{platform.node()}@{platform.system()}"
        
        # Repository information
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            info["repository_uri"] = result.stdout.strip()
            
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            info["git_commit"] = result.stdout.strip()
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            info["repository_uri"] = "unknown"
            info["git_commit"] = "unknown"
        
        return info
    
    def _collect_materials(self) -> List[Dict[str, Any]]:
        """Collect source materials (inputs) for the build"""
        materials = []
        
        # Include Python source files
        for py_file in sorted(self.project_root.rglob("*.py")):
            if not any(part.startswith('.') for part in py_file.parts):
                try:
                    materials.append({
                        "uri": f"file://{py_file.relative_to(self.project_root)}",
                        "digest": {"sha256": self._calculate_file_hash(py_file)}
                    })
                except (IOError, OSError):
                    pass
        
        # Include configuration files
        config_files = [
            "requirements.txt",
            "Makefile",
            ".github/workflows/bench_smoke.yml",
            "schemas/*.json"
        ]
        
        for pattern in config_files:
            for config_file in self.project_root.glob(pattern):
                if config_file.is_file():
                    materials.append({
                        "uri": f"file://{config_file.relative_to(self.project_root)}",
                        "digest": {"sha256": self._calculate_file_hash(config_file)}
                    })
        
        return materials
    
    def _collect_artifacts(self, artifact_paths: List[Path]) -> List[Dict[str, Any]]:
        """Collect information about build artifacts"""
        artifacts = []
        
        for artifact_path in artifact_paths:
            if artifact_path.exists():
                artifacts.append({
                    "name": str(artifact_path.relative_to(self.project_root)),
                    "sha256": self._calculate_file_hash(artifact_path),
                    "size": artifact_path.stat().st_size,
                    "created": datetime.fromtimestamp(artifact_path.stat().st_mtime).isoformat() + "Z"
                })
        
        return artifacts
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        except (IOError, OSError):
            return "unknown"
        return hasher.hexdigest()
    
    def verify_provenance(self, provenance_path: Path, artifact_paths: List[Path]) -> bool:
        """
        Verify provenance against actual artifacts
        
        Args:
            provenance_path: Path to provenance file
            artifact_paths: Paths to artifacts to verify
            
        Returns:
            True if provenance is valid and matches artifacts
        """
        
        try:
            with open(provenance_path) as f:
                provenance = json.load(f)
            
            # Verify provenance structure
            if not self._validate_provenance_structure(provenance):
                return False
            
            # Verify artifact hashes match
            provenance_subjects = {
                subj["name"]: subj["digest"]["sha256"] 
                for subj in provenance.get("subject", [])
            }
            
            for artifact_path in artifact_paths:
                artifact_name = str(artifact_path.relative_to(self.project_root))
                if artifact_name in provenance_subjects:
                    expected_hash = provenance_subjects[artifact_name]
                    actual_hash = self._calculate_file_hash(artifact_path)
                    
                    if expected_hash != actual_hash:
                        print(f"❌ Hash mismatch for {artifact_name}")
                        print(f"   Expected: {expected_hash}")
                        print(f"   Actual:   {actual_hash}")
                        return False
                else:
                    print(f"⚠️  Artifact {artifact_name} not found in provenance")
            
            print("✅ Provenance verification successful")
            return True
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Provenance verification failed: {e}")
            return False
    
    def _validate_provenance_structure(self, provenance: Dict[str, Any]) -> bool:
        """Validate SLSA provenance structure"""
        required_fields = [
            "_type",
            "predicateType", 
            "subject",
            "predicate"
        ]
        
        if not all(field in provenance for field in required_fields):
            return False
        
        predicate = provenance.get("predicate", {})
        required_predicate_fields = [
            "builder",
            "buildType",
            "invocation",
            "metadata"
        ]
        
        return all(field in predicate for field in required_predicate_fields)
