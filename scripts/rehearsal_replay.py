#!/usr/bin/env python3
"""Rehearsal Replay Mechanismus gegen Catastrophic Forgetting"""

import json
import random
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import Dataset, concatenate_datasets
from collections import deque

class RehearsalReplayBuffer:
    """Rehearsal Replay Buffer für Previous Wave Data"""
    
    def __init__(self, buffer_size: int = 10000, replay_ratio: float = 0.02):
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio  # 1-2% replay ratio
        self.buffer = deque(maxlen=buffer_size)
        self.wave_history = {}
        self.current_wave = 0
        
    def add_wave_samples(self, wave_id: str, samples: List[Dict[str, Any]]):
        """Füge Samples einer Wave zum Buffer hinzu"""
        # Sample wichtige Beispiele für Buffer
        sample_count = min(len(samples), self.buffer_size // 4)  # Max 25% pro Wave
        sampled = random.sample(samples, sample_count)
        
        # Markiere Samples mit Wave ID
        for sample in sampled:
            sample['wave_id'] = wave_id
            sample['replay_priority'] = self._calculate_priority(sample)
        
        # Füge zu Buffer hinzu
        self.buffer.extend(sampled)
        self.wave_history[wave_id] = len(sampled)
        
        print(f"🔄 Added {len(sampled)} samples from wave {wave_id} to replay buffer")
        print(f"   Buffer size: {len(self.buffer)}/{self.buffer_size}")
    
    def _calculate_priority(self, sample: Dict[str, Any]) -> float:
        """Berechne Replay Priority basierend auf Sample Eigenschaften"""
        priority = 1.0
        
        # Höhere Priority für mathematische/logische Probleme
        text = str(sample.get('problem', '')) + str(sample.get('solution', ''))
        if any(keyword in text.lower() for keyword in ['math', 'solve', 'calculate', 'logic']):
            priority *= 1.5
        
        # Höhere Priority für kürzere, prägnante Samples
        if len(text) < 200:
            priority *= 1.2
        
        # Random factor für Diversität
        priority *= random.uniform(0.8, 1.2)
        
        return priority
    
    def get_replay_samples(self, batch_size: int) -> List[Dict[str, Any]]:
        """Hole Replay Samples für aktuellen Batch"""
        if len(self.buffer) == 0:
            return []
        
        replay_count = max(1, int(batch_size * self.replay_ratio))
        replay_count = min(replay_count, len(self.buffer))
        
        # Prioritäts-basiertes Sampling
        buffer_list = list(self.buffer)
        priorities = [sample['replay_priority'] for sample in buffer_list]
        
        # Weighted random sampling
        total_priority = sum(priorities)
        if total_priority > 0:
            weights = [p / total_priority for p in priorities]
            selected_indices = random.choices(range(len(buffer_list)), weights=weights, k=replay_count)
            replay_samples = [buffer_list[i] for i in selected_indices]
        else:
            replay_samples = random.sample(buffer_list, replay_count)
        
        return replay_samples
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Statistiken zum Replay Buffer"""
        wave_counts = {}
        for sample in self.buffer:
            wave_id = sample.get('wave_id', 'unknown')
            wave_counts[wave_id] = wave_counts.get(wave_id, 0) + 1
        
        return {
            "total_samples": len(self.buffer),
            "buffer_utilization": len(self.buffer) / self.buffer_size,
            "wave_distribution": wave_counts,
            "replay_ratio": self.replay_ratio
        }

class WaveTransitionManager:
    """Manager für Wave Transitions mit Rehearsal Replay"""
    
    def __init__(self, replay_buffer: RehearsalReplayBuffer):
        self.replay_buffer = replay_buffer
        self.current_wave_data = []
        self.wave_transitions = []
        
    def start_new_wave(self, wave_id: str, wave_data: List[Dict[str, Any]]):
        """Starte neue Training Wave"""
        print(f"🌊 Starting new training wave: {wave_id}")
        
        # Speichere vorherige Wave im Replay Buffer
        if self.current_wave_data:
            previous_wave_id = f"wave_{len(self.wave_transitions)}"
            self.replay_buffer.add_wave_samples(previous_wave_id, self.current_wave_data)
        
        # Setze neue Wave
        self.current_wave_data = wave_data.copy()
        self.wave_transitions.append({
            "wave_id": wave_id,
            "start_time": torch.cuda.current_device() if torch.cuda.is_available() else 0,
            "sample_count": len(wave_data)
        })
        
        print(f"   Wave samples: {len(wave_data)}")
        print(f"   Replay buffer: {self.replay_buffer.get_buffer_stats()}")
    
    def create_mixed_batch(self, current_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Erstelle Mixed Batch mit Rehearsal Replay"""
        if not current_batch:
            return current_batch
        
        # Hole Replay Samples
        replay_samples = self.replay_buffer.get_replay_samples(len(current_batch))
        
        if replay_samples:
            # Mische Current + Replay Samples
            mixed_batch = current_batch + replay_samples
            random.shuffle(mixed_batch)
            
            print(f"🔄 Mixed batch: {len(current_batch)} current + {len(replay_samples)} replay = {len(mixed_batch)} total")
            return mixed_batch
        
        return current_batch
    
    def get_transition_stats(self) -> Dict[str, Any]:
        """Statistiken zu Wave Transitions"""
        buffer_stats = self.replay_buffer.get_buffer_stats()
        
        return {
            "total_waves": len(self.wave_transitions),
            "current_wave_samples": len(self.current_wave_data),
            "replay_buffer": buffer_stats,
            "transitions": self.wave_transitions
        }

def create_rehearsal_dataset(original_dataset: Dataset, transition_manager: WaveTransitionManager) -> Dataset:
    """Erstelle Dataset mit Rehearsal Replay Integration"""
    
    def add_rehearsal_samples(examples):
        """Callback um Rehearsal Samples zu aktuellen Batches hinzuzufügen"""
        current_samples = []
        
        # Konvertiere Examples zu Sample Format
        for i in range(len(examples['input_ids'])):
            sample = {
                'input_ids': examples['input_ids'][i],
                'attention_mask': examples['attention_mask'][i],
                'labels': examples['labels'][i]
            }
            current_samples.append(sample)
        
        # Füge Rehearsal Samples hinzu
        mixed_samples = transition_manager.create_mixed_batch(current_samples)
        
        # Konvertiere zurück zu Examples Format
        if len(mixed_samples) > len(current_samples):
            # Extend examples with replay samples
            extra_samples = mixed_samples[len(current_samples):]
            
            examples['input_ids'].extend([s['input_ids'] for s in extra_samples])
            examples['attention_mask'].extend([s['attention_mask'] for s in extra_samples])
            examples['labels'].extend([s['labels'] for s in extra_samples])
        
        return examples
    
    # Wende Rehearsal Integration an
    enhanced_dataset = original_dataset.map(
        add_rehearsal_samples,
        batched=True,
        batch_size=32,
        desc="Adding rehearsal replay samples"
    )
    
    return enhanced_dataset

if __name__ == "__main__":
    # Test Rehearsal Replay System
    replay_buffer = RehearsalReplayBuffer(buffer_size=1000, replay_ratio=0.02)
    transition_manager = WaveTransitionManager(replay_buffer)
    
    # Simuliere Wave Transitions
    dummy_samples = [
        {"problem": f"Test problem {i}", "solution": f"Solution {i}", "wave_data": True}
        for i in range(100)
    ]
    
    transition_manager.start_new_wave("wave_1", dummy_samples[:50])
    transition_manager.start_new_wave("wave_2", dummy_samples[50:])
    
    # Test Mixed Batch
    current_batch = dummy_samples[:10]
    mixed_batch = transition_manager.create_mixed_batch(current_batch)
    
    print(f"✅ Rehearsal Replay System ready")
    print(f"   Original batch: {len(current_batch)}")
    print(f"   Mixed batch: {len(mixed_batch)}")
    print(f"   Stats: {transition_manager.get_transition_stats()}")
