#!/usr/bin/env python3
"""Enhanced Training Callbacks mit Mini-Validation Probes und Rehearsal Replay"""

import os
import json
import torch
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from transformers.trainer_callback import TrainerCallback
from mini_validation_probes import MiniValidationProbes, setup_mini_validation_datasets
from rehearsal_replay import RehearsalReplayBuffer, WaveTransitionManager

class MiniValidationCallback(TrainerCallback):
    """Callback für Mini-Validation Probes alle 5k Steps"""
    
    def __init__(self, 
                 model_output_dir: str,
                 probe_interval: int = 5000,
                 async_probes: bool = True):
        self.model_output_dir = model_output_dir
        self.probe_interval = probe_interval
        self.async_probes = async_probes
        self.probe_results = []
        
        # Setup validation datasets
        self.validation_dir = setup_mini_validation_datasets()
        self.probes = MiniValidationProbes(
            model_path=model_output_dir,
            validation_data_dir=self.validation_dir
        )
        self.probes.load_validation_datasets()
        
        print(f"🔍 Mini-Validation Callback initialized")
        print(f"   Probe interval: {probe_interval} steps")
        print(f"   Async probes: {async_probes}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Überprüfe ob Mini-Probe ausgeführt werden soll"""
        current_step = state.global_step
        
        if current_step > 0 and current_step % self.probe_interval == 0:
            if self.async_probes:
                # Async Probe um Training nicht zu blockieren
                thread = threading.Thread(
                    target=self._run_async_probe,
                    args=(current_step,),
                    daemon=True
                )
                thread.start()
            else:
                # Sync Probe
                self._run_probe(current_step)
    
    def _run_async_probe(self, step: int):
        """Async Mini-Probe Ausführung"""
        try:
            result = self.probes.run_mini_probe(step)
            self.probe_results.append(result)
            
            # Speichere Results
            results_path = Path(self.model_output_dir) / f"mini_probe_results_step_{step}.json"
            with open(results_path, 'w') as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            print(f"❌ Async probe failed at step {step}: {e}")
    
    def _run_probe(self, step: int):
        """Sync Mini-Probe Ausführung"""
        result = self.probes.run_mini_probe(step)
        self.probe_results.append(result)
        
        # Speichere Results
        results_path = Path(self.model_output_dir) / f"mini_probe_results_step_{step}.json"
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Speichere alle Probe Results am Ende"""
        if self.probe_results:
            final_results_path = Path(self.model_output_dir) / "all_mini_probe_results.json"
            with open(final_results_path, 'w') as f:
                json.dump(self.probe_results, f, indent=2)
            
            print(f"💾 Saved {len(self.probe_results)} mini-probe results to {final_results_path}")

class RehearsalReplayCallback(TrainerCallback):
    """Callback für Rehearsal Replay bei Wave Transitions"""
    
    def __init__(self, 
                 buffer_size: int = 10000,
                 replay_ratio: float = 0.015,  # 1.5%
                 wave_transition_steps: int = 10000):
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.wave_transition_steps = wave_transition_steps
        
        # Initialize Rehearsal System
        self.replay_buffer = RehearsalReplayBuffer(
            buffer_size=buffer_size,
            replay_ratio=replay_ratio
        )
        self.transition_manager = WaveTransitionManager(self.replay_buffer)
        
        self.last_wave_step = 0
        self.current_wave_id = "initial_wave"
        
        print(f"🔄 Rehearsal Replay Callback initialized")
        print(f"   Buffer size: {buffer_size}")
        print(f"   Replay ratio: {replay_ratio*100}%")
        print(f"   Wave transition: every {wave_transition_steps} steps")
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Überprüfe Wave Transitions am Step Begin"""
        current_step = state.global_step
        
        # Check for wave transition
        if (current_step - self.last_wave_step) >= self.wave_transition_steps:
            self._trigger_wave_transition(current_step)
    
    def _trigger_wave_transition(self, step: int):
        """Triggere Wave Transition mit Rehearsal Replay"""
        new_wave_id = f"wave_{step // self.wave_transition_steps}"
        
        print(f"🌊 Wave transition at step {step}: {self.current_wave_id} → {new_wave_id}")
        
        # Simuliere Wave Data (in echter Implementation würde das aus dem aktuellen Dataset kommen)
        dummy_wave_data = [
            {"step": step, "wave_id": new_wave_id, "transition": True}
        ]
        
        self.transition_manager.start_new_wave(new_wave_id, dummy_wave_data)
        
        self.current_wave_id = new_wave_id
        self.last_wave_step = step
        
        # Logge Transition Stats
        stats = self.transition_manager.get_transition_stats()
        print(f"   Transition stats: {stats['replay_buffer']['total_samples']} buffer samples")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Speichere Rehearsal Stats am Ende"""
        stats = self.transition_manager.get_transition_stats()
        stats_path = Path(args.output_dir) / "rehearsal_replay_stats.json"
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"💾 Saved rehearsal replay stats to {stats_path}")

class CombinedEnhancedCallback(TrainerCallback):
    """Kombinierter Callback für alle Enhancements"""
    
    def __init__(self, 
                 model_output_dir: str,
                 mini_probe_interval: int = 5000,
                 rehearsal_buffer_size: int = 10000,
                 rehearsal_ratio: float = 0.015,
                 wave_transition_steps: int = 10000):
        
        # Initialize sub-callbacks
        self.mini_validation = MiniValidationCallback(
            model_output_dir=model_output_dir,
            probe_interval=mini_probe_interval,
            async_probes=True
        )
        
        self.rehearsal_replay = RehearsalReplayCallback(
            buffer_size=rehearsal_buffer_size,
            replay_ratio=rehearsal_ratio,
            wave_transition_steps=wave_transition_steps
        )
        
        self.model_output_dir = model_output_dir
        self.combined_stats = {
            "mini_probes_executed": 0,
            "wave_transitions": 0,
            "rehearsal_samples_used": 0
        }
        
        print(f"🚀 Combined Enhanced Callback initialized")
        print(f"   Mini-probes every {mini_probe_interval} steps")
        print(f"   Rehearsal replay: {rehearsal_ratio*100}% ratio")
        print(f"   Wave transitions: every {wave_transition_steps} steps")
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Step Begin - Rehearsal Check"""
        self.rehearsal_replay.on_step_begin(args, state, control, **kwargs)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Step End - Mini-Validation Check"""
        self.mini_validation.on_step_end(args, state, control, **kwargs)
        
        # Update combined stats
        if state.global_step % self.mini_validation.probe_interval == 0:
            self.combined_stats["mini_probes_executed"] += 1
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Enhanced Logging mit Additional Metrics"""
        if logs and state.global_step % 1000 == 0:  # Log every 1k steps
            # Füge Enhanced Stats zu Logs hinzu
            enhanced_logs = {
                "enhanced_step": state.global_step,
                "mini_probes_executed": self.combined_stats["mini_probes_executed"],
                "wave_transitions": self.combined_stats["wave_transitions"],
                "rehearsal_buffer_size": len(self.rehearsal_replay.replay_buffer.buffer),
                "current_wave": self.rehearsal_replay.current_wave_id
            }
            
            print(f"📊 Enhanced Training Stats (Step {state.global_step}):")
            for key, value in enhanced_logs.items():
                print(f"   {key}: {value}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Training End - Save All Stats"""
        self.mini_validation.on_train_end(args, state, control, **kwargs)
        self.rehearsal_replay.on_train_end(args, state, control, **kwargs)
        
        # Save combined stats
        combined_stats_path = Path(self.model_output_dir) / "combined_enhancement_stats.json"
        with open(combined_stats_path, 'w') as f:
            json.dump(self.combined_stats, f, indent=2)
        
        print(f"💾 Saved combined enhancement stats to {combined_stats_path}")

def create_enhanced_callbacks(model_output_dir: str) -> list:
    """Factory function für Enhanced Callbacks"""
    
    combined_callback = CombinedEnhancedCallback(
        model_output_dir=model_output_dir,
        mini_probe_interval=5000,     # Mini-probes alle 5k steps
        rehearsal_buffer_size=10000,  # 10k sample buffer
        rehearsal_ratio=0.015,        # 1.5% rehearsal ratio
        wave_transition_steps=10000   # Wave transition alle 10k steps
    )
    
    return [combined_callback]

if __name__ == "__main__":
    # Test Enhanced Callbacks
    callbacks = create_enhanced_callbacks("./test_output")
    
    print(f"✅ Enhanced Training Callbacks ready")
    print(f"   Callbacks created: {len(callbacks)}")
    print(f"   Features: Mini-Validation Probes + Rehearsal Replay")
