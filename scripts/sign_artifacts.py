#!/usr/bin/env python3
"""
Sign Artifacts - Phase 12
Script to cryptographically sign build artifacts
"""

import argparse
import sys
from pathlib import Path
from miso.supply_chain.artifact_signer import ArtifactSigner, SignatureVerifier

def main():
    parser = argparse.ArgumentParser(description="Sign and verify build artifacts")
    parser.add_argument("command", choices=["sign", "verify"], 
                       help="Command to execute")
    parser.add_argument("artifacts", nargs="+",
                       help="Artifact files to sign/verify")
    parser.add_argument("--method", "-m", 
                       choices=["cosign", "gpg", "internal"],
                       default="internal",
                       help="Signing method (default: internal)")
    parser.add_argument("--signatures-dir",
                       help="Directory for signatures (default: ./signatures)")
    
    args = parser.parse_args()
    
    # Parse artifact paths
    artifact_paths = [Path(p) for p in args.artifacts]
    
    # Check if artifacts exist
    missing = [p for p in artifact_paths if not p.exists()]
    if missing:
        print(f"❌ Artifacts not found: {', '.join(str(p) for p in missing)}")
        sys.exit(1)
    
    try:
        if args.command == "sign":
            # Initialize signer
            signatures_dir = Path(args.signatures_dir) if args.signatures_dir else None
            if signatures_dir:
                signatures_dir.mkdir(exist_ok=True)
            signer = ArtifactSigner()
            
            print(f"🔐 Signing {len(artifact_paths)} artifacts with {args.method}...")
            
            signatures = signer.sign_artifacts(artifact_paths, args.method)
            
            print(f"\n✅ Signed {len(signatures)} artifacts successfully")
            for sig in signatures:
                print(f"   {Path(sig.artifact_path).name} → {Path(sig.signature_path).name}")
                
        elif args.command == "verify":
            # Initialize verifier
            signatures_dir = Path(args.signatures_dir) if args.signatures_dir else None
            verifier = SignatureVerifier(signatures_dir)
            
            print(f"🔍 Verifying {len(artifact_paths)} artifact signatures...")
            
            results = verifier.verify_artifacts(artifact_paths)
            
            verified_count = sum(1 for valid in results.values() if valid)
            print(f"\n📊 Verification Results: {verified_count}/{len(results)} valid")
            
            if verified_count < len(results):
                print("\n❌ Some signatures failed verification")
                sys.exit(1)
            else:
                print("\n✅ All signatures verified successfully")
                
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
