#!/usr/bin/env python3
"""
Artifact Signer - Phase 12
Cryptographic signing and verification of build artifacts
"""

import json
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import os

@dataclass
class SignatureInfo:
    """Information about an artifact signature"""
    artifact_path: str
    signature_path: str
    hash_algorithm: str
    hash_value: str
    signed_at: str
    signer: str
    signature_format: str

class ArtifactSigner:
    """
    Signs build artifacts using cryptographic signatures
    Phase 12: Supply-Chain & Artefakt-Integrität
    """
    
    def __init__(self, key_path: Path = None):
        self.key_path = key_path
        self.signatures_dir = Path("signatures")
        self.signatures_dir.mkdir(exist_ok=True)
    
    def sign_artifacts(self, 
                      artifact_paths: List[Path],
                      signing_method: str = "cosign") -> List[SignatureInfo]:
        """
        Sign multiple artifacts
        
        Args:
            artifact_paths: List of artifact files to sign
            signing_method: Signing method ("cosign", "gpg", or "internal")
            
        Returns:
            List of signature information
        """
        
        signatures = []
        
        for artifact_path in artifact_paths:
            if not artifact_path.exists():
                print(f"⚠️  Artifact not found: {artifact_path}")
                continue
            
            try:
                signature_info = self._sign_single_artifact(artifact_path, signing_method)
                signatures.append(signature_info)
                print(f"✅ Signed: {artifact_path.name}")
            except Exception as e:
                print(f"❌ Failed to sign {artifact_path.name}: {e}")
        
        # Create signature manifest
        self._create_signature_manifest(signatures)
        
        return signatures
    
    def _sign_single_artifact(self, artifact_path: Path, method: str) -> SignatureInfo:
        """Sign a single artifact"""
        
        # Calculate artifact hash
        hash_value = self._calculate_file_hash(artifact_path)
        
        if method == "cosign":
            return self._sign_with_cosign(artifact_path, hash_value)
        elif method == "gpg":
            return self._sign_with_gpg(artifact_path, hash_value)
        else:
            return self._sign_internal(artifact_path, hash_value)
    
    def _sign_with_cosign(self, artifact_path: Path, hash_value: str) -> SignatureInfo:
        """Sign with cosign (Sigstore)"""
        
        signature_path = self.signatures_dir / f"{artifact_path.name}.sig"
        
        try:
            # Use cosign to sign (keyless signing)
            subprocess.run([
                "cosign", "sign-blob",
                "--output-signature", str(signature_path),
                str(artifact_path)
            ], check=True, capture_output=True)
            
            return SignatureInfo(
                artifact_path=str(artifact_path),
                signature_path=str(signature_path),
                hash_algorithm="sha256",
                hash_value=hash_value,
                signed_at=datetime.now().isoformat() + "Z",
                signer="cosign-keyless",
                signature_format="cosign"
            )
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to internal signing if cosign not available
            print("⚠️  cosign not available, using internal signing")
            return self._sign_internal(artifact_path, hash_value)
    
    def _sign_with_gpg(self, artifact_path: Path, hash_value: str) -> SignatureInfo:
        """Sign with GPG"""
        
        signature_path = self.signatures_dir / f"{artifact_path.name}.gpg"
        
        try:
            subprocess.run([
                "gpg", "--armor", "--detach-sign",
                "--output", str(signature_path),
                str(artifact_path)
            ], check=True, capture_output=True)
            
            return SignatureInfo(
                artifact_path=str(artifact_path),
                signature_path=str(signature_path),
                hash_algorithm="sha256",
                hash_value=hash_value,
                signed_at=datetime.now().isoformat() + "Z",
                signer="gpg",
                signature_format="pgp"
            )
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to internal signing if GPG not available
            print("⚠️  GPG not available, using internal signing")
            return self._sign_internal(artifact_path, hash_value)
    
    def _sign_internal(self, artifact_path: Path, hash_value: str) -> SignatureInfo:
        """Internal HMAC-based signing (for development/testing)"""
        
        signature_path = self.signatures_dir / f"{artifact_path.name}.miso_sig"
        
        # Create internal signature (HMAC + metadata)
        # In production, use proper cryptographic keys
        secret_key = os.getenv("MISO_SIGNING_KEY", "miso-development-key-do-not-use-in-production")
        
        # Create signature payload
        payload = {
            "artifact_path": str(artifact_path),
            "hash_algorithm": "sha256",
            "hash_value": hash_value,
            "signed_at": datetime.now().isoformat() + "Z",
            "signer": "miso-internal"
        }
        
        # Calculate HMAC
        import hmac
        signature = hmac.new(
            secret_key.encode(),
            json.dumps(payload, sort_keys=True).encode(),
            hashlib.sha256
        ).hexdigest()
        
        payload["signature"] = signature
        
        # Write signature file
        with open(signature_path, 'w') as f:
            json.dump(payload, f, indent=2)
        
        return SignatureInfo(
            artifact_path=str(artifact_path),
            signature_path=str(signature_path),
            hash_algorithm="sha256",
            hash_value=hash_value,
            signed_at=payload["signed_at"],
            signer="miso-internal",
            signature_format="miso-hmac"
        )
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _create_signature_manifest(self, signatures: List[SignatureInfo]):
        """Create manifest of all signatures"""
        
        manifest_path = self.signatures_dir / "signature_manifest.json"
        
        manifest = {
            "generated_at": datetime.now().isoformat() + "Z",
            "signatures": [asdict(sig) for sig in signatures]
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"📋 Signature manifest created: {manifest_path}")

class SignatureVerifier:
    """
    Verifies artifact signatures
    Phase 12: Supply-Chain & Artefakt-Integrität  
    """
    
    def __init__(self, signatures_dir: Path = None):
        self.signatures_dir = signatures_dir or Path("signatures")
    
    def verify_artifacts(self, artifact_paths: List[Path]) -> Dict[str, bool]:
        """
        Verify signatures for multiple artifacts
        
        Args:
            artifact_paths: List of artifacts to verify
            
        Returns:
            Dictionary mapping artifact names to verification status
        """
        
        results = {}
        
        for artifact_path in artifact_paths:
            if not artifact_path.exists():
                results[artifact_path.name] = False
                print(f"❌ Artifact not found: {artifact_path}")
                continue
            
            try:
                is_valid = self._verify_single_artifact(artifact_path)
                results[artifact_path.name] = is_valid
                
                if is_valid:
                    print(f"✅ Signature verified: {artifact_path.name}")
                else:
                    print(f"❌ Signature invalid: {artifact_path.name}")
                    
            except Exception as e:
                results[artifact_path.name] = False
                print(f"❌ Verification failed for {artifact_path.name}: {e}")
        
        return results
    
    def _verify_single_artifact(self, artifact_path: Path) -> bool:
        """Verify signature for a single artifact"""
        
        # Check for different signature formats
        cosign_sig = self.signatures_dir / f"{artifact_path.name}.sig"
        gpg_sig = self.signatures_dir / f"{artifact_path.name}.gpg"
        miso_sig = self.signatures_dir / f"{artifact_path.name}.miso_sig"
        
        if cosign_sig.exists():
            return self._verify_cosign_signature(artifact_path, cosign_sig)
        elif gpg_sig.exists():
            return self._verify_gpg_signature(artifact_path, gpg_sig)
        elif miso_sig.exists():
            return self._verify_miso_signature(artifact_path, miso_sig)
        else:
            print(f"⚠️  No signature found for {artifact_path.name}")
            return False
    
    def _verify_cosign_signature(self, artifact_path: Path, signature_path: Path) -> bool:
        """Verify cosign signature"""
        try:
            result = subprocess.run([
                "cosign", "verify-blob",
                "--signature", str(signature_path),
                str(artifact_path)
            ], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _verify_gpg_signature(self, artifact_path: Path, signature_path: Path) -> bool:
        """Verify GPG signature"""
        try:
            result = subprocess.run([
                "gpg", "--verify",
                str(signature_path),
                str(artifact_path)
            ], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _verify_miso_signature(self, artifact_path: Path, signature_path: Path) -> bool:
        """Verify internal MISO signature"""
        try:
            # Load signature file
            with open(signature_path) as f:
                sig_data = json.load(f)
            
            # Recalculate artifact hash
            current_hash = self._calculate_file_hash(artifact_path)
            
            # Check hash matches
            if sig_data.get("hash_value") != current_hash:
                return False
            
            # Verify HMAC signature
            secret_key = os.getenv("MISO_SIGNING_KEY", "miso-development-key-do-not-use-in-production")
            
            payload = {k: v for k, v in sig_data.items() if k != "signature"}
            
            import hmac
            expected_signature = hmac.new(
                secret_key.encode(),
                json.dumps(payload, sort_keys=True).encode(),
                hashlib.sha256
            ).hexdigest()
            
            return sig_data.get("signature") == expected_signature
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def verify_manifest(self) -> bool:
        """Verify the signature manifest integrity"""
        
        manifest_path = self.signatures_dir / "signature_manifest.json"
        
        if not manifest_path.exists():
            print("⚠️  No signature manifest found")
            return False
        
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            signatures = manifest.get("signatures", [])
            verified_count = 0
            
            for sig_info in signatures:
                artifact_path = Path(sig_info["artifact_path"])
                if artifact_path.exists():
                    is_valid = self._verify_single_artifact(artifact_path)
                    if is_valid:
                        verified_count += 1
            
            print(f"📋 Manifest verification: {verified_count}/{len(signatures)} signatures valid")
            return verified_count == len(signatures)
            
        except (FileNotFoundError, json.JSONDecodeError):
            print("❌ Invalid signature manifest")
            return False
