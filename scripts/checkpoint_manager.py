#!/usr/bin/env python3
"""Advanced checkpoint and backup management system"""

import os, json, shutil, hashlib
from pathlib import Path
from datetime import datetime
import subprocess

ROOT = Path(os.environ.get("ROOT", "."))
RUNS_DIR = ROOT/"runs"
BACKUP_DIR = ROOT/"backups"

class CheckpointManager:
    def __init__(self):
        self.backup_dir = BACKUP_DIR
        self.backup_dir.mkdir(exist_ok=True)
        
    def get_checkpoint_info(self, checkpoint_path):
        """Extract information from a checkpoint"""
        trainer_state = checkpoint_path / "trainer_state.json"
        if not trainer_state.exists():
            return None
            
        with open(trainer_state) as f:
            state = json.load(f)
            
        return {
            'path': checkpoint_path,
            'step': state.get('global_step', 0),
            'epoch': state.get('epoch', 0),
            'loss': state.get('log_history', [{}])[-1].get('loss'),
            'size_mb': self.get_directory_size(checkpoint_path) / (1024*1024),
            'created': datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
        }
    
    def get_directory_size(self, path):
        """Get total size of directory in bytes"""
        total = 0
        for entry in Path(path).rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
        return total
    
    def find_all_checkpoints(self, run_dir=None):
        """Find all checkpoints in runs directory"""
        if run_dir is None:
            search_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
        else:
            search_dirs = [Path(run_dir)]
            
        checkpoints = []
        for run_dir in search_dirs:
            for checkpoint in run_dir.glob("checkpoint-*"):
                if checkpoint.is_dir():
                    info = self.get_checkpoint_info(checkpoint)
                    if info:
                        checkpoints.append(info)
        
        return sorted(checkpoints, key=lambda x: x['step'])
    
    def create_checkpoint_backup(self, checkpoint_path, backup_name=None):
        """Create compressed backup of checkpoint"""
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"checkpoint_{checkpoint_path.name}_{timestamp}"
            
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        print(f"Creating backup: {backup_name}")
        try:
            # Create compressed archive
            cmd = ["tar", "-czf", str(backup_path), "-C", str(checkpoint_path.parent), checkpoint_path.name]
            subprocess.run(cmd, check=True)
            
            # Calculate checksum
            checksum = self.calculate_checksum(backup_path)
            
            # Save metadata
            metadata = {
                'original_path': str(checkpoint_path),
                'backup_path': str(backup_path),
                'checksum': checksum,
                'created': datetime.now().isoformat(),
                'size_bytes': backup_path.stat().st_size
            }
            
            metadata_path = backup_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"✅ Backup created: {backup_path}")
            print(f"📊 Size: {backup_path.stat().st_size / (1024*1024):.1f} MB")
            return backup_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Backup failed: {e}")
            return None
    
    def calculate_checksum(self, file_path):
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def cleanup_old_checkpoints(self, run_dir, keep_count=3):
        """Clean up old checkpoints, keeping only the most recent ones"""
        checkpoints = []
        for checkpoint in Path(run_dir).glob("checkpoint-*"):
            if checkpoint.is_dir():
                info = self.get_checkpoint_info(checkpoint)
                if info:
                    checkpoints.append(info)
        
        # Sort by step number and keep only the most recent
        checkpoints.sort(key=lambda x: x['step'])
        to_remove = checkpoints[:-keep_count] if len(checkpoints) > keep_count else []
        
        total_freed = 0
        for checkpoint_info in to_remove:
            size = checkpoint_info['size_mb']
            shutil.rmtree(checkpoint_info['path'])
            total_freed += size
            print(f"🗑️  Removed checkpoint: {checkpoint_info['path'].name} ({size:.1f} MB)")
        
        if total_freed > 0:
            print(f"💾 Space freed: {total_freed:.1f} MB")
        
        return len(to_remove)
    
    def verify_backup_integrity(self, backup_path):
        """Verify backup integrity using checksum"""
        metadata_path = Path(str(backup_path).replace('.tar.gz', '.json'))
        
        if not metadata_path.exists():
            print(f"❌ No metadata found for {backup_path}")
            return False
            
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        current_checksum = self.calculate_checksum(backup_path)
        stored_checksum = metadata['checksum']
        
        if current_checksum == stored_checksum:
            print(f"✅ Backup integrity verified: {backup_path.name}")
            return True
        else:
            print(f"❌ Backup corrupted: {backup_path.name}")
            return False

def main():
    manager = CheckpointManager()
    
    print("🔧 CHECKPOINT MANAGEMENT SYSTEM")
    print("=" * 50)
    
    # Find all checkpoints
    checkpoints = manager.find_all_checkpoints()
    
    if not checkpoints:
        print("❌ No checkpoints found")
        return
    
    print(f"📂 Found {len(checkpoints)} checkpoints:")
    
    for i, cp in enumerate(checkpoints[-5:], 1):  # Show last 5
        print(f"{i:2d}. {cp['path'].name} - Step {cp['step']:,} - {cp['size_mb']:.1f} MB - Loss: {cp['loss']}")
    
    # Get latest checkpoint
    latest = checkpoints[-1]
    print(f"\n🎯 Latest checkpoint: {latest['path'].name}")
    print(f"   Step: {latest['step']:,}")
    print(f"   Size: {latest['size_mb']:.1f} MB")
    print(f"   Created: {latest['created']}")
    
    # Create backup of latest
    backup_path = manager.create_checkpoint_backup(latest['path'])
    
    if backup_path:
        # Verify backup
        manager.verify_backup_integrity(backup_path)
    
    # Cleanup old checkpoints (keep last 3)
    for run_dir in RUNS_DIR.iterdir():
        if run_dir.is_dir() and any(run_dir.glob("checkpoint-*")):
            print(f"\n🧹 Cleaning up {run_dir.name}")
            removed = manager.cleanup_old_checkpoints(run_dir, keep_count=3)
            if removed == 0:
                print("   No cleanup needed")

if __name__ == "__main__":
    main()
