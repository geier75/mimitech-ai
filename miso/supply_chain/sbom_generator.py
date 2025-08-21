#!/usr/bin/env python3
"""
SBOM Generator - Phase 12
Software Bill of Materials generation for supply chain security
"""

import json
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict

class SBOMFormat(Enum):
    """Supported SBOM formats"""
    SPDX_JSON = "spdx+json"
    CYCLONE_DX = "cyclonedx+json"

@dataclass
class ComponentInfo:
    """Information about a software component"""
    name: str
    version: str
    supplier: Optional[str] = None
    download_location: Optional[str] = None
    files_analyzed: bool = True
    license_concluded: Optional[str] = None
    license_declared: Optional[str] = None
    copyright_text: Optional[str] = None
    checksum: Optional[str] = None

class SBOMGenerator:
    """
    Generates Software Bill of Materials (SBOM) for MISO project
    Phase 12: Supply-Chain & Artefakt-Integrität
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.requirements_file = self.project_root / "requirements.txt"
        
    def generate_sbom(self, 
                     output_format: SBOMFormat = SBOMFormat.SPDX_JSON,
                     output_path: Path = None) -> Path:
        """
        Generate SBOM for the MISO project
        
        Args:
            output_format: Format for SBOM output
            output_path: Optional output file path
            
        Returns:
            Path to generated SBOM file
        """
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"miso_sbom_{timestamp}.{output_format.value.split('+')[1]}"
            output_path = self.project_root / filename
        
        if output_format == SBOMFormat.SPDX_JSON:
            return self._generate_spdx_sbom(output_path)
        elif output_format == SBOMFormat.CYCLONE_DX:
            return self._generate_cyclone_dx_sbom(output_path)
        else:
            raise ValueError(f"Unsupported SBOM format: {output_format}")
    
    def _generate_spdx_sbom(self, output_path: Path) -> Path:
        """Generate SPDX format SBOM"""
        
        # Collect project information
        project_info = self._get_project_info()
        dependencies = self._get_dependencies()
        
        # Create SPDX document
        spdx_document = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": f"MISO Ultimate SBOM {datetime.now().strftime('%Y-%m-%d')}",
            "documentNamespace": f"https://miso-ultimate.local/sbom/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "creationInfo": {
                "licenseListVersion": "3.21",
                "creators": ["Tool: MISO SBOM Generator"],
                "created": datetime.now().isoformat() + "Z"
            },
            "packages": []
        }
        
        # Add main project package
        main_package = {
            "SPDXID": "SPDXRef-Package-MISO-Ultimate",
            "name": "MISO Ultimate",
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": True,
            "licenseConcluded": "NOASSERTION", 
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "versionInfo": project_info.get("version", "unknown"),
            "supplier": "Organization: MISO Team",
            "checksums": [
                {
                    "algorithm": "SHA256",
                    "checksumValue": self._calculate_project_checksum()
                }
            ]
        }
        spdx_document["packages"].append(main_package)
        
        # Add dependency packages
        for i, dep in enumerate(dependencies, 1):
            package_id = f"SPDXRef-Package-{dep.name.replace('-', '')}-{i}"
            dep_package = {
                "SPDXID": package_id,
                "name": dep.name,
                "downloadLocation": dep.download_location or "NOASSERTION",
                "filesAnalyzed": dep.files_analyzed,
                "licenseConcluded": dep.license_concluded or "NOASSERTION",
                "licenseDeclared": dep.license_declared or "NOASSERTION", 
                "copyrightText": dep.copyright_text or "NOASSERTION",
                "versionInfo": dep.version,
                "supplier": dep.supplier or "NOASSERTION"
            }
            
            if dep.checksum:
                dep_package["checksums"] = [
                    {
                        "algorithm": "SHA256",
                        "checksumValue": dep.checksum
                    }
                ]
            
            spdx_document["packages"].append(dep_package)
        
        # Add relationships
        relationships = [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": "SPDXRef-Package-MISO-Ultimate"
            }
        ]
        
        for i, dep in enumerate(dependencies, 1):
            package_id = f"SPDXRef-Package-{dep.name.replace('-', '')}-{i}"
            relationships.append({
                "spdxElementId": "SPDXRef-Package-MISO-Ultimate",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": package_id
            })
        
        spdx_document["relationships"] = relationships
        
        # Write SBOM file
        with open(output_path, 'w') as f:
            json.dump(spdx_document, f, indent=2)
        
        print(f"✅ SPDX SBOM generated: {output_path}")
        print(f"📦 Main package: {main_package['name']} v{main_package['versionInfo']}")
        print(f"🔗 Dependencies: {len(dependencies)} packages")
        
        return output_path
    
    def _generate_cyclone_dx_sbom(self, output_path: Path) -> Path:
        """Generate CycloneDX format SBOM"""
        
        # Collect project information
        project_info = self._get_project_info()
        dependencies = self._get_dependencies()
        
        # Create CycloneDX document
        cyclone_document = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "serialNumber": f"urn:uuid:miso-ultimate-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "version": 1,
            "metadata": {
                "timestamp": datetime.now().isoformat() + "Z",
                "tools": [
                    {
                        "vendor": "MISO Team",
                        "name": "MISO SBOM Generator",
                        "version": "1.0.0"
                    }
                ],
                "component": {
                    "type": "application",
                    "bom-ref": "miso-ultimate",
                    "name": "MISO Ultimate",
                    "version": project_info.get("version", "unknown"),
                    "supplier": {
                        "name": "MISO Team"
                    }
                }
            },
            "components": []
        }
        
        # Add dependency components
        for dep in dependencies:
            component = {
                "type": "library",
                "bom-ref": f"{dep.name}@{dep.version}",
                "name": dep.name,
                "version": dep.version,
                "purl": f"pkg:pypi/{dep.name}@{dep.version}"
            }
            
            if dep.supplier:
                component["supplier"] = {"name": dep.supplier}
            
            if dep.license_declared:
                component["licenses"] = [{"license": {"name": dep.license_declared}}]
            
            if dep.checksum:
                component["hashes"] = [{"alg": "SHA-256", "content": dep.checksum}]
            
            cyclone_document["components"].append(component)
        
        # Write SBOM file
        with open(output_path, 'w') as f:
            json.dump(cyclone_document, f, indent=2)
        
        print(f"✅ CycloneDX SBOM generated: {output_path}")
        print(f"📦 Main component: {cyclone_document['metadata']['component']['name']}")
        print(f"🔗 Components: {len(cyclone_document['components'])} packages")
        
        return output_path
    
    def _get_project_info(self) -> Dict[str, Any]:
        """Get project information from git and setup files"""
        info = {}
        
        try:
            # Get git commit info
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            info["git_commit"] = result.stdout.strip()
            
            # Get git tag if available
            result = subprocess.run(
                ["git", "describe", "--tags", "--exact-match"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                info["version"] = result.stdout.strip()
            else:
                info["version"] = info["git_commit"][:8]
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            info["git_commit"] = "unknown"
            info["version"] = "unknown"
        
        return info
    
    def _get_dependencies(self) -> List[ComponentInfo]:
        """Extract dependencies from requirements.txt and pip"""
        dependencies = []
        
        # Parse requirements.txt if available
        if self.requirements_file.exists():
            with open(self.requirements_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Basic parsing of package==version
                        if '==' in line:
                            name, version = line.split('==', 1)
                        elif '>=' in line:
                            name, version = line.split('>=', 1)
                            version = f">={version}"
                        else:
                            name, version = line, "unknown"
                        
                        dependencies.append(ComponentInfo(
                            name=name.strip(),
                            version=version.strip(),
                            download_location=f"https://pypi.org/project/{name.strip()}/",
                            files_analyzed=False,
                            license_declared="NOASSERTION"
                        ))
        
        # Enhance with pip show information
        self._enhance_with_pip_info(dependencies)
        
        return dependencies
    
    def _enhance_with_pip_info(self, dependencies: List[ComponentInfo]):
        """Enhance dependency info with pip show output"""
        for dep in dependencies:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "show", dep.name],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.startswith('Version:'):
                        dep.version = line.split(':', 1)[1].strip()
                    elif line.startswith('License:'):
                        license_text = line.split(':', 1)[1].strip()
                        if license_text and license_text != "UNKNOWN":
                            dep.license_declared = license_text
                    elif line.startswith('Author:'):
                        author = line.split(':', 1)[1].strip()
                        if author and author != "UNKNOWN":
                            dep.supplier = f"Person: {author}"
                            
            except (subprocess.CalledProcessError, FileNotFoundError):
                # pip show failed, keep existing info
                pass
    
    def _calculate_project_checksum(self) -> str:
        """Calculate SHA256 checksum for project source files"""
        hasher = hashlib.sha256()
        
        # Include Python source files
        for py_file in sorted(self.project_root.rglob("*.py")):
            if not any(part.startswith('.') for part in py_file.parts):
                try:
                    with open(py_file, 'rb') as f:
                        hasher.update(f.read())
                except (IOError, OSError):
                    pass
        
        # Include requirements.txt if exists
        if self.requirements_file.exists():
            with open(self.requirements_file, 'rb') as f:
                hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def verify_sbom(self, sbom_path: Path) -> bool:
        """Verify SBOM file integrity and format"""
        try:
            with open(sbom_path) as f:
                sbom_data = json.load(f)
            
            # Check for required SPDX fields
            if "spdxVersion" in sbom_data:
                required_fields = ["spdxVersion", "dataLicense", "SPDXID", "name"]
                return all(field in sbom_data for field in required_fields)
            
            # Check for required CycloneDX fields  
            elif "bomFormat" in sbom_data:
                required_fields = ["bomFormat", "specVersion", "serialNumber", "version"]
                return all(field in sbom_data for field in required_fields)
            
            return False
            
        except (json.JSONDecodeError, FileNotFoundError):
            return False
