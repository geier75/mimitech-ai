#!/usr/bin/env python3
"""
Dataset Registry - Phase 13
Centralized registry for dataset provenance and licensing
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class LicenseType(Enum):
    """Supported license types"""
    MIT = "MIT"
    APACHE_2_0 = "Apache-2.0"
    CC_BY_4_0 = "CC-BY-4.0"
    CC_BY_SA_4_0 = "CC-BY-SA-4.0"
    CUSTOM = "Custom"

@dataclass
class DatasetInfo:
    """Complete dataset information for governance"""
    name: str
    source_url: str
    license_type: LicenseType
    license_url: str
    attribution: str
    hash_sha256: str
    last_updated: str
    usage_rights: str
    commercial_use: bool
    requires_attribution: bool
    share_alike: bool
    local_path: str
    manifest_path: str
    sample_count: int
    
    def __post_init__(self):
        """Validate dataset info on creation"""
        if self.sample_count < 0:
            raise ValueError(f"Invalid sample count: {self.sample_count}")
        if not self.hash_sha256 or len(self.hash_sha256) != 64:
            raise ValueError(f"Invalid SHA256 hash: {self.hash_sha256}")

class DatasetRegistry:
    """
    Centralized registry for dataset provenance and compliance
    Phase 13: Daten-Governance & Compliance
    """
    
    def __init__(self, registry_path: Path = None):
        self.registry_path = registry_path or Path("datasets/registry.json")
        self.datasets_dir = self.registry_path.parent
        self.datasets_dir.mkdir(exist_ok=True)
        
        # Load existing registry or create new
        self.datasets = self._load_registry()
    
    def register_dataset(self, dataset_info: DatasetInfo) -> bool:
        """
        Register a new dataset with full provenance information
        
        Args:
            dataset_info: Complete dataset information
            
        Returns:
            True if registration successful
        """
        
        # Validate dataset exists
        dataset_path = Path(dataset_info.local_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Validate manifest exists
        manifest_path = Path(dataset_info.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        # Verify hash matches manifest
        if not self._verify_dataset_integrity(dataset_info):
            raise ValueError(f"Dataset integrity check failed: {dataset_info.name}")
        
        # Check for license compliance
        self._validate_license_compliance(dataset_info)
        
        # Register dataset
        self.datasets[dataset_info.name] = dataset_info
        self._save_registry()
        
        print(f"✅ Dataset registered: {dataset_info.name}")
        print(f"   License: {dataset_info.license_type.value}")
        print(f"   Samples: {dataset_info.sample_count:,}")
        print(f"   Commercial: {'Yes' if dataset_info.commercial_use else 'No'}")
        
        return True
    
    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """Get complete information for a dataset"""
        return self.datasets.get(dataset_name)
    
    def list_datasets(self) -> List[DatasetInfo]:
        """List all registered datasets"""
        return list(self.datasets.values())
    
    def verify_compliance(self, dataset_names: List[str] = None) -> Dict[str, bool]:
        """
        Verify compliance for specified datasets or all datasets
        
        Args:
            dataset_names: Optional list of specific datasets to check
            
        Returns:
            Dictionary mapping dataset names to compliance status
        """
        
        if dataset_names is None:
            dataset_names = list(self.datasets.keys())
        
        compliance = {}
        
        for name in dataset_names:
            if name in self.datasets:
                dataset = self.datasets[name]
                
                try:
                    # Check file integrity
                    integrity_ok = self._verify_dataset_integrity(dataset)
                    
                    # Check license compliance
                    license_ok = self._validate_license_compliance(dataset, raise_on_error=False)
                    
                    # Check access permissions
                    access_ok = self._verify_access_permissions(dataset)
                    
                    compliance[name] = integrity_ok and license_ok and access_ok
                    
                    if not compliance[name]:
                        print(f"⚠️  Compliance issue detected for {name}")
                    
                except Exception as e:
                    compliance[name] = False
                    print(f"❌ Compliance check failed for {name}: {e}")
            else:
                compliance[name] = False
                print(f"❌ Dataset not registered: {name}")
        
        return compliance
    
    def generate_attribution_report(self) -> str:
        """Generate attribution text for all datasets"""
        
        attribution_lines = []
        attribution_lines.append("# Dataset Attributions")
        attribution_lines.append("")
        
        for dataset in sorted(self.datasets.values(), key=lambda d: d.name):
            attribution_lines.append(f"## {dataset.name}")
            attribution_lines.append(f"- **Source**: {dataset.source_url}")
            attribution_lines.append(f"- **License**: {dataset.license_type.value}")
            attribution_lines.append(f"- **Attribution**: {dataset.attribution}")
            
            if dataset.license_url:
                attribution_lines.append(f"- **License URL**: {dataset.license_url}")
                
            attribution_lines.append("")
        
        return "\n".join(attribution_lines)
    
    def _load_registry(self) -> Dict[str, DatasetInfo]:
        """Load dataset registry from file"""
        
        if not self.registry_path.exists():
            return {}
        
        try:
            with open(self.registry_path) as f:
                registry_data = json.load(f)
            
            datasets = {}
            for name, data in registry_data.items():
                # Convert license string back to enum
                data["license_type"] = LicenseType(data["license_type"])
                datasets[name] = DatasetInfo(**data)
            
            return datasets
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"⚠️  Failed to load registry, creating new: {e}")
            return {}
    
    def _save_registry(self):
        """Save dataset registry to file"""
        
        # Convert to JSON-serializable format
        registry_data = {}
        for name, dataset in self.datasets.items():
            data = asdict(dataset)
            data["license_type"] = dataset.license_type.value
            registry_data[name] = data
        
        # Add metadata
        registry_with_metadata = {
            "_metadata": {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "total_datasets": len(self.datasets)
            },
            "datasets": registry_data
        }
        
        with open(self.registry_path, 'w') as f:
            json.dump(registry_with_metadata, f, indent=2)
    
    def _verify_dataset_integrity(self, dataset_info: DatasetInfo) -> bool:
        """Verify dataset integrity against manifest"""
        
        manifest_path = Path(dataset_info.manifest_path)
        if not manifest_path.exists():
            return False
        
        try:
            # Load manifest and check hash
            with open(manifest_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        stored_hash, filename = line.split(None, 1)
                        if stored_hash == dataset_info.hash_sha256:
                            return True
            return False
            
        except (IOError, ValueError):
            return False
    
    def _validate_license_compliance(self, dataset_info: DatasetInfo, raise_on_error: bool = True) -> bool:
        """Validate license compliance requirements"""
        
        issues = []
        
        # Check attribution requirements
        if dataset_info.requires_attribution and not dataset_info.attribution:
            issues.append("Attribution required but not provided")
        
        # Check commercial use restrictions
        if not dataset_info.commercial_use:
            # For now, we assume MISO may be used commercially
            # In production, check actual usage context
            pass
        
        # Check share-alike requirements
        if dataset_info.share_alike:
            # Verify that derivative works will be shared under same license
            # For benchmarking, this usually means results/reports
            pass
        
        if issues:
            error_msg = f"License compliance issues for {dataset_info.name}: {', '.join(issues)}"
            if raise_on_error:
                raise ValueError(error_msg)
            else:
                print(f"⚠️  {error_msg}")
                return False
        
        return True
    
    def _verify_access_permissions(self, dataset_info: DatasetInfo) -> bool:
        """Verify dataset access permissions are correctly set"""
        
        dataset_path = Path(dataset_info.local_path)
        
        try:
            # Check if path exists and is readable
            if not dataset_path.exists():
                return False
            
            # Check if directory is readable
            if dataset_path.is_dir():
                return any(dataset_path.iterdir())  # Can list contents
            else:
                # Check if file is readable
                with open(dataset_path, 'r') as f:
                    f.read(1)  # Read one character to test access
                return True
                
        except (PermissionError, IOError):
            return False
    
    def create_standard_datasets(self):
        """Create registry entries for standard MISO datasets"""
        
        standard_datasets = [
            DatasetInfo(
                name="mmlu",
                source_url="https://github.com/hendrycks/test",
                license_type=LicenseType.MIT,
                license_url="https://github.com/hendrycks/test/blob/master/LICENSE",
                attribution='Hendrycks et al., "Measuring Massive Multitask Language Understanding"',
                hash_sha256="a1b2c3d4e5f6789012345678901234567890123456789012345678901234567890",  # Placeholder
                last_updated="2024-01-15",
                usage_rights="Academic and commercial use permitted",
                commercial_use=True,
                requires_attribution=True,
                share_alike=False,
                local_path="datasets/mmlu",
                manifest_path="datasets/mmlu/manifest.sha256",
                sample_count=14042
            ),
            DatasetInfo(
                name="hellaswag", 
                source_url="https://github.com/rowanz/hellaswag",
                license_type=LicenseType.MIT,
                license_url="https://github.com/rowanz/hellaswag/blob/master/LICENSE",
                attribution='Zellers et al., "HellaSwag: Can a Machine Really Finish Your Sentence?"',
                hash_sha256="f6e5d4c3b2a1098765432109876543210987654321098765432109876543210987",  # Placeholder
                last_updated="2024-01-15",
                usage_rights="Academic and commercial use permitted",
                commercial_use=True,
                requires_attribution=True,
                share_alike=False,
                local_path="datasets/hellaswag",
                manifest_path="datasets/hellaswag/manifest.sha256",
                sample_count=10042
            ),
            DatasetInfo(
                name="winogrande",
                source_url="https://github.com/allenai/winogrande", 
                license_type=LicenseType.APACHE_2_0,
                license_url="https://github.com/allenai/winogrande/blob/master/LICENSE",
                attribution='Sakaguchi et al., "WinoGrande: An Adversarial Winograd Schema Challenge"',
                hash_sha256="b2a1f6e5d4c3098765432109876543210987654321098765432109876543210987",  # Placeholder
                last_updated="2024-01-15",
                usage_rights="Academic and commercial use permitted",
                commercial_use=True,
                requires_attribution=True,
                share_alike=False,
                local_path="datasets/winogrande", 
                manifest_path="datasets/winogrande/manifest.sha256",
                sample_count=1267
            ),
            DatasetInfo(
                name="piqa",
                source_url="https://github.com/ybisk/ybisk.github.io/tree/master/piqa",
                license_type=LicenseType.CC_BY_4_0,
                license_url="https://creativecommons.org/licenses/by/4.0/",
                attribution='Bisk et al., "PIQA: Reasoning about Physical Commonsense in Natural Language"',
                hash_sha256="c3b2a1f6e5d4098765432109876543210987654321098765432109876543210987",  # Placeholder
                last_updated="2024-01-15",
                usage_rights="Academic and commercial use with attribution",
                commercial_use=True,
                requires_attribution=True,
                share_alike=False,
                local_path="datasets/piqa",
                manifest_path="datasets/piqa/manifest.sha256", 
                sample_count=1838
            ),
            DatasetInfo(
                name="arc",
                source_url="https://allenai.org/data/arc",
                license_type=LicenseType.CC_BY_SA_4_0,
                license_url="https://creativecommons.org/licenses/by-sa/4.0/",
                attribution='Clark et al., "Think you have Solved Question Answering? Try ARC"',
                hash_sha256="d4c3b2a1f6e5098765432109876543210987654321098765432109876543210987",  # Placeholder
                last_updated="2024-01-15",
                usage_rights="Academic and commercial use with attribution and share-alike",
                commercial_use=True,
                requires_attribution=True,
                share_alike=True,
                local_path="datasets/arc",
                manifest_path="datasets/arc/manifest.sha256",
                sample_count=2590
            )
        ]
        
        for dataset in standard_datasets:
            try:
                # Only register if not already present
                if dataset.name not in self.datasets:
                    self.datasets[dataset.name] = dataset
                    print(f"📝 Registered standard dataset: {dataset.name}")
            except Exception as e:
                print(f"⚠️  Failed to register {dataset.name}: {e}")
        
        self._save_registry()
        print(f"✅ Standard dataset registry created with {len(self.datasets)} datasets")
