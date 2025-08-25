#!/usr/bin/env python3
"""Simple terminal-based training dashboard"""

import os, json, time
from pathlib import Path
import subprocess
import signal
import sys

ROOT = Path(os.environ.get("ROOT", "."))
RUNS_DIR = ROOT/"runs/phase2_mps_real"

class TrainingDashboard:
    def __init__(self):
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        print("\n👋 Dashboard shutting down...")
        self.running = False
    
    def clear_screen(self):
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def get_training_status(self):
        """Get current training status"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            training_active = 'train_mps.py' in result.stdout
            
            # Get prometheus metrics
            metrics_result = subprocess.run(
                ['curl', '-s', 'http://localhost:9108/metrics'],
                capture_output=True, text=True, timeout=2
            )
            
            metrics = {}
            if metrics_result.returncode == 0:
                for line in metrics_result.stdout.split('\n'):
                    if line.startswith('mimikcompute_train_loss '):
                        metrics['loss'] = float(line.split(' ')[1])
                    elif line.startswith('mimikcompute_train_lr '):
                        metrics['learning_rate'] = float(line.split(' ')[1])
                    elif line.startswith('mimikcompute_train_grad_norm '):
                        metrics['grad_norm'] = float(line.split(' ')[1])
            
            return training_active, metrics
            
        except Exception:
            return False, {}
    
    def get_checkpoint_info(self):
        """Get latest checkpoint information"""
        checkpoints = list(RUNS_DIR.glob("checkpoint-*"))
        if not checkpoints:
            return None
        
        latest = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
        trainer_state_file = latest / "trainer_state.json"
        
        if not trainer_state_file.exists():
            return None
        
        try:
            with open(trainer_state_file) as f:
                state = json.load(f)
            return {
                'checkpoint': latest.name,
                'step': state.get('global_step', 0),
                'max_steps': state.get('max_steps', 1),
                'epoch': state.get('epoch', 0),
                'total_epochs': state.get('num_train_epochs', 3)
            }
        except Exception:
            return None
    
    def format_time_remaining(self, current_step, max_steps, hours_so_far=6):
        """Estimate time remaining"""
        if current_step == 0:
            return "Unknown"
        
        progress = current_step / max_steps
        if progress == 0:
            return "Unknown"
        
        total_estimated_hours = hours_so_far / progress
        remaining_hours = total_estimated_hours - hours_so_far
        
        if remaining_hours < 1:
            return f"{remaining_hours * 60:.0f} minutes"
        else:
            return f"{remaining_hours:.1f} hours"
    
    def display_dashboard(self):
        """Display the main dashboard"""
        self.clear_screen()
        
        print("🚀 MISO ULTIMATE TRAINING DASHBOARD")
        print("=" * 60)
        print(f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Training status
        training_active, metrics = self.get_training_status()
        status_icon = "🟢" if training_active else "🔴"
        print(f"{status_icon} Training Process: {'RUNNING' if training_active else 'STOPPED'}")
        
        # Checkpoint info
        checkpoint_info = self.get_checkpoint_info()
        if checkpoint_info:
            progress = (checkpoint_info['step'] / checkpoint_info['max_steps']) * 100
            
            print(f"📂 Checkpoint: {checkpoint_info['checkpoint']}")
            print(f"📊 Progress: {checkpoint_info['step']:,} / {checkpoint_info['max_steps']:,} ({progress:.1f}%)")
            print(f"🔄 Epoch: {checkpoint_info['epoch']:.3f} / {checkpoint_info['total_epochs']}")
            
            # Progress bar
            bar_width = 40
            filled = int(bar_width * progress / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"▶️  [{bar}] {progress:.1f}%")
            
            # Time estimation
            eta = self.format_time_remaining(checkpoint_info['step'], checkpoint_info['max_steps'])
            print(f"⏱️  ETA: {eta}")
        
        print()
        
        # Metrics
        if metrics:
            print("📈 CURRENT METRICS")
            print("-" * 30)
            print(f"Loss: {metrics.get('loss', 'N/A')}")
            print(f"Learning Rate: {metrics.get('learning_rate', 'N/A'):.2e}")
            print(f"Gradient Norm: {metrics.get('grad_norm', 'N/A'):.4f}")
        else:
            print("📈 METRICS: Not available")
        
        print()
        print("🔧 MONITORING SERVICES")
        print("-" * 30)
        
        # Check services
        try:
            prometheus_result = subprocess.run(
                ['curl', '-s', 'http://localhost:9108/metrics'],
                capture_output=True, text=True, timeout=2
            )
            prometheus_status = "🟢 RUNNING" if prometheus_result.returncode == 0 else "🔴 STOPPED"
        except:
            prometheus_status = "🔴 STOPPED"
        
        print(f"Prometheus Exporter: {prometheus_status}")
        
        # Check caffeinate
        try:
            caffeinate_result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            caffeinate_status = "🟢 ACTIVE" if 'caffeinate' in caffeinate_result.stdout else "🔴 INACTIVE"
        except:
            caffeinate_status = "🔴 UNKNOWN"
        
        print(f"Sleep Prevention: {caffeinate_status}")
        
        print()
        print("Press Ctrl+C to exit dashboard")
        print("Refreshing every 10 seconds...")
    
    def run(self):
        """Run the dashboard"""
        while self.running:
            self.display_dashboard()
            
            # Wait 10 seconds or until interrupted
            for _ in range(100):  # 10 seconds in 0.1s intervals
                if not self.running:
                    break
                time.sleep(0.1)

def main():
    dashboard = TrainingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
