#!/usr/bin/env python3
"""Comprehensive post-training evaluation pipeline"""

import os, json, sys
from pathlib import Path
import subprocess
from datetime import datetime

ROOT = Path(os.environ.get("ROOT", "."))
RUNS_DIR = ROOT/"runs/phase2_mps_real"

def find_final_checkpoint():
    """Find the final/best checkpoint"""
    checkpoints = list(RUNS_DIR.glob("checkpoint-*"))
    if not checkpoints:
        return None
    
    # Sort by step number and get the latest
    final = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
    return final

def run_validation_gates(checkpoint_path):
    """Run the comprehensive validation gates"""
    print("🔍 Running validation gates...")
    
    try:
        # Run the existing validation script
        cmd = [
            sys.executable, "scripts/validate_gates_real.py",
            "--checkpoint_path", str(checkpoint_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Validation gates: PASSED")
            return True, result.stdout
        else:
            print("❌ Validation gates: FAILED")
            print(result.stderr)
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print("⏰ Validation gates: TIMEOUT")
        return False, "Validation timed out"
    except Exception as e:
        print(f"❌ Validation gates: ERROR - {e}")
        return False, str(e)

def merge_and_prepare_model(checkpoint_path):
    """Merge LoRA adapters and prepare final model"""
    print("🔧 Merging model adapters...")
    
    try:
        cmd = [
            sys.executable, "scripts/merge_adapters_real.py",
            "--checkpoint_path", str(checkpoint_path),
            "--output_dir", str(RUNS_DIR / "final_merged_model")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Model merge: SUCCESS")
            return True, str(RUNS_DIR / "final_merged_model")
        else:
            print("❌ Model merge: FAILED")
            print(result.stderr)
            return False, result.stderr
            
    except Exception as e:
        print(f"❌ Model merge: ERROR - {e}")
        return False, str(e)

def generate_training_report(checkpoint_path):
    """Generate comprehensive training report"""
    print("📊 Generating training report...")
    
    trainer_state_file = checkpoint_path / "trainer_state.json"
    if not trainer_state_file.exists():
        return None
    
    with open(trainer_state_file) as f:
        state = json.load(f)
    
    # Extract key metrics
    final_step = state.get('global_step', 0)
    final_epoch = state.get('epoch', 0)
    logs = state.get('log_history', [])
    
    if not logs:
        return None
    
    # Calculate training statistics
    losses = [log.get('loss') for log in logs if 'loss' in log]
    learning_rates = [log.get('learning_rate') for log in logs if 'learning_rate' in log]
    
    report = {
        "training_summary": {
            "final_step": final_step,
            "final_epoch": final_epoch,
            "total_epochs": state.get('num_train_epochs', 3),
            "completion_status": "completed" if final_step >= state.get('max_steps', 0) else "incomplete"
        },
        "performance_metrics": {
            "final_loss": losses[-1] if losses else None,
            "best_loss": min(losses) if losses else None,
            "average_loss": sum(losses) / len(losses) if losses else None,
            "loss_improvement": losses[0] - losses[-1] if len(losses) > 1 else None,
            "final_learning_rate": learning_rates[-1] if learning_rates else None
        },
        "training_stability": {
            "total_log_entries": len(logs),
            "gradient_norms": [log.get('grad_norm') for log in logs if 'grad_norm' in log],
            "loss_variance": None
        },
        "timestamp": datetime.now().isoformat(),
        "checkpoint_path": str(checkpoint_path)
    }
    
    # Calculate loss variance for stability assessment
    if len(losses) > 1:
        mean_loss = sum(losses) / len(losses)
        variance = sum((x - mean_loss) ** 2 for x in losses) / len(losses)
        report["training_stability"]["loss_variance"] = variance
    
    # Save report
    report_file = checkpoint_path / "training_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def create_deployment_bundle(merged_model_path, checkpoint_path):
    """Create deployment-ready bundle"""
    print("📦 Creating deployment bundle...")
    
    bundle_dir = RUNS_DIR / "deployment_bundle"
    bundle_dir.mkdir(exist_ok=True)
    
    # Copy essential files
    files_to_copy = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json",
        "special_tokens_map.json",
        "training_report.json"
    ]
    
    for file_name in files_to_copy:
        src_file = checkpoint_path / file_name
        if src_file.exists():
            import shutil
            shutil.copy2(src_file, bundle_dir / file_name)
    
    # Create deployment metadata
    metadata = {
        "model_type": "TinyLlama-1.1B-Chat-v1.0-LoRA",
        "training_method": "Phase2-MPS-Real",
        "created_at": datetime.now().isoformat(),
        "source_checkpoint": str(checkpoint_path),
        "merged_model_path": str(merged_model_path) if merged_model_path else None,
        "deployment_ready": True
    }
    
    with open(bundle_dir / "deployment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Deployment bundle created: {bundle_dir}")
    return bundle_dir

def main():
    print("🎯 POST-TRAINING EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Find final checkpoint
    checkpoint = find_final_checkpoint()
    if not checkpoint:
        print("❌ No checkpoints found!")
        return 1
    
    print(f"📂 Final checkpoint: {checkpoint}")
    
    # Step 1: Generate training report
    report = generate_training_report(checkpoint)
    if report:
        print("✅ Training report generated")
        print(f"   Final Loss: {report['performance_metrics']['final_loss']:.6f}")
        print(f"   Completion: {report['training_summary']['completion_status']}")
    else:
        print("❌ Failed to generate training report")
    
    # Step 2: Run validation gates
    gates_passed, gates_output = run_validation_gates(checkpoint)
    
    # Step 3: Merge model (if validation passed)
    merged_model_path = None
    if gates_passed:
        merge_success, merged_model_path = merge_and_prepare_model(checkpoint)
    else:
        print("⚠️  Skipping model merge due to validation failures")
        merge_success = False
    
    # Step 4: Create deployment bundle
    bundle_path = None
    if merge_success and merged_model_path:
        bundle_path = create_deployment_bundle(merged_model_path, checkpoint)
    
    # Final summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Training Report: {'✅ Generated' if report else '❌ Failed'}")
    print(f"Validation Gates: {'✅ Passed' if gates_passed else '❌ Failed'}")
    print(f"Model Merge: {'✅ Success' if merge_success else '❌ Failed'}")
    print(f"Deployment Bundle: {'✅ Created' if bundle_path else '❌ Not Created'}")
    
    if bundle_path:
        print(f"\n🚀 READY FOR DEPLOYMENT")
        print(f"Bundle Location: {bundle_path}")
    else:
        print(f"\n⚠️  MODEL NEEDS ATTENTION")
        print("Review validation results and fix issues before deployment")
    
    return 0 if gates_passed and merge_success else 1

if __name__ == "__main__":
    sys.exit(main())
