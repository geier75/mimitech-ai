#!/usr/bin/env python3
"""Lightweight training monitor without model loading"""

import os, json, time
from pathlib import Path
from datetime import datetime

ROOT = Path(os.environ.get("ROOT", "."))
RUNS_DIR = ROOT/"runs/phase2_mps_real"

def find_latest_checkpoint():
    """Find the most recent checkpoint"""
    checkpoints = list(RUNS_DIR.glob("checkpoint-*"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
    return latest

def analyze_training_progress():
    """Analyze training progress from trainer_state.json"""
    checkpoint = find_latest_checkpoint()
    if not checkpoint:
        print("❌ No checkpoints found")
        return
    
    trainer_state_file = checkpoint / "trainer_state.json"
    if not trainer_state_file.exists():
        print(f"❌ No trainer_state.json in {checkpoint}")
        return
    
    with open(trainer_state_file) as f:
        state = json.load(f)
    
    # Current progress
    current_step = state.get('global_step', 0)
    max_steps = state.get('max_steps', 1)
    current_epoch = state.get('epoch', 0)
    progress_pct = (current_step / max_steps) * 100
    
    print(f"📊 TRAINING PROGRESS ANALYSIS")
    print(f"=" * 50)
    print(f"Checkpoint: {checkpoint.name}")
    print(f"Progress: {current_step:,} / {max_steps:,} steps ({progress_pct:.1f}%)")
    print(f"Epoch: {current_epoch:.3f} / {state.get('num_train_epochs', 3)}")
    
    # Loss analysis
    if state.get('log_history'):
        logs = state['log_history']
        recent_logs = logs[-10:] if len(logs) >= 10 else logs
        
        losses = [log.get('loss') for log in recent_logs if 'loss' in log]
        if losses:
            print(f"\n📉 LOSS ANALYSIS (last {len(losses)} steps)")
            print(f"Current Loss: {losses[-1]:.6f}")
            print(f"Average Loss: {sum(losses)/len(losses):.6f}")
            print(f"Min Loss: {min(losses):.6f}")
            print(f"Max Loss: {max(losses):.6f}")
            
            if len(losses) > 1:
                trend = "📈 Increasing" if losses[-1] > losses[0] else "📉 Decreasing"
                print(f"Trend: {trend}")
        
        # Learning rate
        latest_log = logs[-1]
        if 'learning_rate' in latest_log:
            print(f"\n⚙️  TRAINING PARAMETERS")
            print(f"Learning Rate: {latest_log['learning_rate']:.2e}")
        
        if 'grad_norm' in latest_log:
            print(f"Gradient Norm: {latest_log['grad_norm']:.6f}")
    
    # Time estimation
    if len(state.get('log_history', [])) > 1:
        first_log = state['log_history'][0]
        latest_log = state['log_history'][-1]
        
        if 'step' in first_log and 'step' in latest_log:
            steps_done = latest_log['step'] - first_log['step']
            steps_remaining = max_steps - current_step
            
            if steps_done > 0:
                # Rough time estimation (assumes consistent step timing)
                print(f"\n⏱️  TIME ESTIMATION")
                print(f"Steps completed: {steps_done:,}")
                print(f"Steps remaining: {steps_remaining:,}")
                
                # ETA based on current pace
                if steps_remaining > 0:
                    eta_hours = (steps_remaining / steps_done) * 6  # Rough estimate
                    print(f"Estimated completion: ~{eta_hours:.1f} hours")

def check_system_health():
    """Check system resources and health"""
    print(f"\n🖥️  SYSTEM HEALTH")
    print(f"=" * 50)
    
    # Check if training process is running
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'train_mps.py' in result.stdout:
            print("✅ Training process: RUNNING")
        else:
            print("❌ Training process: NOT FOUND")
    except:
        print("❓ Training process: UNKNOWN")
    
    # Check monitoring services
    try:
        result = subprocess.run(['curl', '-s', 'http://localhost:9108/metrics'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and 'mimikcompute' in result.stdout:
            print("✅ Prometheus exporter: RUNNING")
        else:
            print("❌ Prometheus exporter: NOT RESPONDING")
    except:
        print("❓ Prometheus exporter: UNKNOWN")

def main():
    print(f"🔍 Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    analyze_training_progress()
    check_system_health()
    
    print(f"\n" + "=" * 70)
    print("Monitor complete. Run again anytime to check progress.")

if __name__ == "__main__":
    main()
