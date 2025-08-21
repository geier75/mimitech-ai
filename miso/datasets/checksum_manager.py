"""
Dataset checksum management for integrity validation
Creates and validates checksums for dataset files
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ChecksumManager:
    """Manages dataset file checksums for integrity validation"""
    
    def __init__(self, datasets_root: Path):
        self.datasets_root = Path(datasets_root)
        self.manifest_file = self.datasets_root / "checksums_manifest.json"
        
    def calculate_file_checksum(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Calculate checksum for a file"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        hash_obj = hashlib.new(algorithm)
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            raise
    
    def generate_checksums_manifest(self) -> Dict[str, Any]:
        """Generate checksums manifest for all dataset files"""
        if not self.datasets_root.exists():
            logger.warning(f"Datasets root does not exist: {self.datasets_root}")
            return {"files": {}, "metadata": {"generated_at": datetime.now().isoformat(), "total_files": 0}}
        
        manifest = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "algorithm": "sha256",
                "total_files": 0,
                "datasets_root": str(self.datasets_root)
            },
            "files": {}
        }
        
        # Find all dataset files
        dataset_extensions = ['.json', '.jsonl', '.csv', '.tsv', '.txt', '.allow']
        dataset_files = []
        
        for ext in dataset_extensions:
            dataset_files.extend(self.datasets_root.glob(f"**/*{ext}"))
        
        # Calculate checksums
        for file_path in dataset_files:
            if file_path.is_file():
                try:
                    relative_path = file_path.relative_to(self.datasets_root)
                    checksum = self.calculate_file_checksum(file_path)
                    
                    manifest["files"][str(relative_path)] = {
                        "checksum": checksum,
                        "size_bytes": file_path.stat().st_size,
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
        
        manifest["metadata"]["total_files"] = len(manifest["files"])
        
        # Save manifest
        try:
            with open(self.manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"✅ Generated checksums manifest: {len(manifest['files'])} files")
            
        except Exception as e:
            logger.error(f"Error saving checksums manifest: {e}")
            raise
        
        return manifest
    
    def load_checksums_manifest(self) -> Optional[Dict[str, Any]]:
        """Load existing checksums manifest"""
        if not self.manifest_file.exists():
            logger.debug(f"Checksums manifest not found: {self.manifest_file}")
            return None
        
        try:
            with open(self.manifest_file, 'r') as f:
                manifest = json.load(f)
            
            logger.debug(f"Loaded checksums manifest: {manifest['metadata']['total_files']} files")
            return manifest
            
        except Exception as e:
            logger.error(f"Error loading checksums manifest: {e}")
            return None
    
    def verify_file_integrity(self, file_path: Path) -> bool:
        """Verify integrity of a single file against manifest"""
        manifest = self.load_checksums_manifest()
        if manifest is None:
            logger.warning("No checksums manifest found - generating new one")
            manifest = self.generate_checksums_manifest()
        
        try:
            relative_path = file_path.relative_to(self.datasets_root)
            relative_path_str = str(relative_path)
            
            if relative_path_str not in manifest["files"]:
                logger.warning(f"File not in manifest: {relative_path}")
                return False
            
            expected_checksum = manifest["files"][relative_path_str]["checksum"]
            actual_checksum = self.calculate_file_checksum(file_path)
            
            if expected_checksum == actual_checksum:
                logger.debug(f"✅ Checksum valid: {relative_path}")
                return True
            else:
                logger.error(f"❌ Checksum mismatch for {relative_path}: expected {expected_checksum}, got {actual_checksum}")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying file integrity for {file_path}: {e}")
            return False
    
    def verify_all_files(self) -> Dict[str, bool]:
        """Verify integrity of all files in manifest"""
        manifest = self.load_checksums_manifest()
        if manifest is None:
            logger.error("Cannot verify files - no manifest available")
            return {}
        
        results = {}
        
        for relative_path_str in manifest["files"]:
            file_path = self.datasets_root / relative_path_str
            
            if not file_path.exists():
                logger.error(f"❌ File missing: {relative_path_str}")
                results[relative_path_str] = False
                continue
            
            results[relative_path_str] = self.verify_file_integrity(file_path)
        
        passed = sum(results.values())
        total = len(results)
        
        if passed == total:
            logger.info(f"✅ All files passed integrity check: {passed}/{total}")
        else:
            logger.error(f"❌ Integrity check failed: {passed}/{total} files passed")
        
        return results
    
    def update_manifest_for_file(self, file_path: Path):
        """Update manifest entry for a single file"""
        manifest = self.load_checksums_manifest()
        if manifest is None:
            manifest = self.generate_checksums_manifest()
            return
        
        try:
            relative_path = file_path.relative_to(self.datasets_root)
            relative_path_str = str(relative_path)
            
            checksum = self.calculate_file_checksum(file_path)
            
            manifest["files"][relative_path_str] = {
                "checksum": checksum,
                "size_bytes": file_path.stat().st_size,
                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            
            manifest["metadata"]["generated_at"] = datetime.now().isoformat()
            manifest["metadata"]["total_files"] = len(manifest["files"])
            
            with open(self.manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"✅ Updated manifest for: {relative_path}")
            
        except Exception as e:
            logger.error(f"Error updating manifest for {file_path}: {e}")
            raise
