#!/usr/bin/env python3
"""Automated training completion notification system"""

import os, json, time
from pathlib import Path
import subprocess
from datetime import datetime

ROOT = Path(os.environ.get("ROOT", "."))
RUNS_DIR = ROOT/"runs/phase2_mps_real"

class TrainingCompletionNotifier:
    def __init__(self):
        self.last_step = 0
        self.max_steps = 101589
        self.completion_threshold = 0.99  # Consider complete at 99%
        self.stagnation_threshold = 300  # seconds without progress
        
    def get_current_progress(self):
        """Get current training progress"""
        checkpoints = list(RUNS_DIR.glob("checkpoint-*"))
        if not checkpoints:
            return None
            
        latest = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
        trainer_state_file = latest / "trainer_state.json"
        
        if not trainer_state_file.exists():
            return None
            
        with open(trainer_state_file) as f:
            state = json.load(f)
            
        return {
            'step': state.get('global_step', 0),
            'max_steps': state.get('max_steps', self.max_steps),
            'epoch': state.get('epoch', 0),
            'loss': state.get('log_history', [{}])[-1].get('loss'),
            'checkpoint': latest.name
        }
    
    def is_training_active(self):
        """Check if training process is still running"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            return 'train_mps.py' in result.stdout
        except:
            return False
    
    def notify_completion(self, progress_info):
        """Send completion notification"""
        completion_pct = (progress_info['step'] / progress_info['max_steps']) * 100
        
        message = f"""
🎉 TRAINING COMPLETION DETECTED!
================================
Checkpoint: {progress_info['checkpoint']}
Final Step: {progress_info['step']:,} / {progress_info['max_steps']:,}
Progress: {completion_pct:.1f}%
Final Loss: {progress_info['loss']:.6f}
Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Next Steps:
1. Run post-training evaluation
2. Validate model quality
3. Merge and prepare for deployment
"""
        
        print(message)
        
        # Save completion report
        report_file = RUNS_DIR / "training_completion_report.txt"
        with open(report_file, 'w') as f:
            f.write(message)
            
        # Play system sound (macOS)
        try:
            subprocess.run(['afplay', '/System/Library/Sounds/Glass.aiff'], 
                         capture_output=True)
        except:
            pass
            
        return message
    
    def notify_stagnation(self, progress_info, minutes_stagnant):
        """Notify about training stagnation"""
        message = f"""
⚠️ TRAINING STAGNATION DETECTED
==============================
No progress for {minutes_stagnant:.1f} minutes
Current Step: {progress_info['step']:,}
Training Process: {'RUNNING' if self.is_training_active() else 'STOPPED'}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Action Required: Check training logs for errors
"""
        
        print(message)
        return message
    
    def monitor_loop(self, check_interval=60):
        """Main monitoring loop"""
        print("🔍 Training completion monitor started")
        print(f"Checking every {check_interval} seconds...")
        
        last_progress_time = time.time()
        
        while True:
            try:
                progress = self.get_current_progress()
                if not progress:
                    print("❌ No progress data available")
                    time.sleep(check_interval)
                    continue
                
                current_step = progress['step']
                completion_pct = (current_step / progress['max_steps']) * 100
                
                # Check for completion
                if completion_pct >= (self.completion_threshold * 100):
                    self.notify_completion(progress)
                    break
                
                # Check for progress
                if current_step > self.last_step:
                    self.last_step = current_step
                    last_progress_time = time.time()
                    print(f"📊 Progress: {current_step:,} steps ({completion_pct:.1f}%)")
                else:
                    # Check for stagnation
                    stagnant_seconds = time.time() - last_progress_time
                    if stagnant_seconds > self.stagnation_threshold:
                        minutes_stagnant = stagnant_seconds / 60
                        self.notify_stagnation(progress, minutes_stagnant)
                        
                        # Check if training process died
                        if not self.is_training_active():
                            print("❌ Training process not found - stopping monitor")
                            break
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("\n👋 Monitor stopped by user")
                break
            except Exception as e:
                print(f"❌ Monitor error: {e}")
                time.sleep(check_interval)

def main():
    notifier = TrainingCompletionNotifier()
    
    # Get initial progress
    initial_progress = notifier.get_current_progress()
    if initial_progress:
        print(f"🚀 Initial progress: {initial_progress['step']:,} / {initial_progress['max_steps']:,} steps")
        notifier.last_step = initial_progress['step']
    
    # Start monitoring
    notifier.monitor_loop(check_interval=60)  # Check every minute

if __name__ == "__main__":
    main()
