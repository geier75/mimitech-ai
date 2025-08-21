#!/usr/bin/env python3
"""
Generate SBOM - Phase 12
Script to generate Software Bill of Materials
"""

import argparse
import sys
from pathlib import Path
from miso.supply_chain.sbom_generator import SBOMGenerator, SBOMFormat

def main():
    parser = argparse.ArgumentParser(description="Generate SBOM for MISO project")
    parser.add_argument("--format", "-f", 
                       choices=["spdx+json", "cyclonedx+json"],
                       default="spdx+json",
                       help="SBOM format (default: spdx+json)")
    parser.add_argument("--output", "-o",
                       help="Output file path (default: auto-generated)")
    parser.add_argument("--project-root",
                       help="Project root directory (default: current)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify generated SBOM")
    
    args = parser.parse_args()
    
    # Initialize SBOM generator
    project_root = Path(args.project_root) if args.project_root else None
    generator = SBOMGenerator(project_root)
    
    # Parse format
    if args.format == "spdx+json":
        sbom_format = SBOMFormat.SPDX_JSON
    elif args.format == "cyclonedx+json":
        sbom_format = SBOMFormat.CYCLONE_DX
    else:
        print(f"❌ Unsupported format: {args.format}")
        sys.exit(1)
    
    try:
        # Generate SBOM
        print("📦 Generating Software Bill of Materials...")
        
        output_path = Path(args.output) if args.output else None
        sbom_path = generator.generate_sbom(sbom_format, output_path)
        
        # Verify SBOM if requested
        if args.verify:
            print("\n🔍 Verifying SBOM...")
            is_valid = generator.verify_sbom(sbom_path)
            
            if is_valid:
                print("✅ SBOM verification successful")
            else:
                print("❌ SBOM verification failed")
                sys.exit(1)
        
        print(f"\n📄 SBOM generated successfully: {sbom_path}")
        
    except Exception as e:
        print(f"❌ SBOM generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
