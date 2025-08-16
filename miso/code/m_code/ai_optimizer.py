#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE AI Optimizer

Dieses Modul implementiert eine KI-gestützte Optimierungsschicht für M-CODE.
Es analysiert Codemuster und Ausführungspfade, um automatisch optimale Ausführungsstrategien zu wählen.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import logging
import time
import random
import numpy as np
import threading
import json
import pickle
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.ai_optimizer")

# Import von internen Modulen
from .mlx_adapter import get_mlx_adapter, MLX_AVAILABLE
from .jit_compiler import get_jit_compiler
from .parallel_executor import get_echo_prime_integration
from .debug_profiler import get_profiler, profile


@dataclass
class OptimizerConfig:
    """Konfiguration für den AI-Optimizer"""
    enabled: bool = True
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    discount_factor: float = 0.9
    batch_size: int = 64
    memory_size: int = 10000
    update_frequency: int = 100
    min_samples: int = 1000
    model_path: Optional[str] = None
    use_reinforcement_learning: bool = True
    use_pattern_recognition: bool = True
    adaptive_optimization: bool = True
    telemetry_enabled: bool = True


@dataclass
class OptimizationStrategy:
    """Optimierungsstrategie für einen Code-Abschnitt"""
    strategy_id: str
    name: str
    parallelization_level: int = 0  # 0: keine, 1: Threads, 2: Prozesse
    jit_level: int = 0  # 0: keine, 1: Basic, 2: Aggressive
    device_target: str = "auto"  # auto, cpu, gpu, ane
    memory_optimization: str = "balanced"  # low, balanced, high
    batch_processing: bool = False
    tensor_fusion: bool = False
    operator_fusion: bool = False
    loop_unrolling: bool = False
    automatic_differentiation: bool = False
    confidence: float = 0.0  # Konfidenz des Modells in diese Strategie


@dataclass
class CodePattern:
    """Erkanntes Codemuster"""
    pattern_id: str
    name: str
    description: str
    features: Dict[str, Any]
    frequency: int = 0
    avg_execution_time: float = 0.0
    optimal_strategy: Optional[OptimizationStrategy] = None


@dataclass
class ExecutionContext:
    """Ausführungskontext für eine Optimierung"""
    code_hash: str
    input_shapes: List[Tuple[int, ...]]
    input_types: List[str]
    output_shape: Optional[Tuple[int, ...]] = None
    output_type: Optional[str] = None
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    strategy: Optional[OptimizationStrategy] = None
    success: bool = True
    error: Optional[str] = None


class PatternRecognizer:
    """Erkennt Muster in Code und Ausführungspfaden"""
    
    def __init__(self, min_frequency: int = 5):
        """
        Initialisiert einen neuen Mustererkenner.
        
        Args:
            min_frequency: Minimale Häufigkeit für ein erkanntes Muster
        """
        self.min_frequency = min_frequency
        self.patterns = {}
        self.code_to_pattern = {}
        self.execution_history = []
        self.pattern_counter = 0
    
    def analyze_code(self, code_str: str, execution_context: ExecutionContext) -> Optional[str]:
        """
        Analysiert Code und ordnet ihn einem Muster zu.
        
        Args:
            code_str: Code als String
            execution_context: Ausführungskontext
            
        Returns:
            Pattern-ID oder None
        """
        # Berechne Code-Hash als Identifikator
        code_hash = self._hash_code(code_str)
        
        # Prüfe, ob bereits zugeordnet
        if code_hash in self.code_to_pattern:
            pattern_id = self.code_to_pattern[code_hash]
            pattern = self.patterns[pattern_id]
            
            # Aktualisiere Statistiken
            pattern.frequency += 1
            pattern.avg_execution_time = (pattern.avg_execution_time * (pattern.frequency - 1) +
                                          execution_context.execution_time_ms) / pattern.frequency
            
            return pattern_id
        
        # Extrahiere Merkmale
        features = self._extract_features(code_str, execution_context)
        
        # Suche nach ähnlichem Muster
        best_match = self._find_similar_pattern(features)
        
        if best_match:
            # Ordne Code dem bestehenden Muster zu
            self.code_to_pattern[code_hash] = best_match
            self.patterns[best_match].frequency += 1
            return best_match
        
        # Erstelle neues Muster, wenn häufig genug gesehen
        temp_key = f"temp_{code_hash}"
        if temp_key not in self.patterns:
            self.patterns[temp_key] = CodePattern(
                pattern_id=temp_key,
                name=f"Unbekanntes Muster {self.pattern_counter}",
                description="Automatisch erkanntes Muster",
                features=features,
                frequency=1,
                avg_execution_time=execution_context.execution_time_ms
            )
        else:
            self.patterns[temp_key].frequency += 1
            self.patterns[temp_key].avg_execution_time = (
                (self.patterns[temp_key].avg_execution_time * (self.patterns[temp_key].frequency - 1) +
                 execution_context.execution_time_ms) / self.patterns[temp_key].frequency
            )
        
        # Wenn häufig genug, als richtiges Muster speichern
        if self.patterns[temp_key].frequency >= self.min_frequency:
            pattern_id = f"pattern_{self.pattern_counter}"
            self.pattern_counter += 1
            
            self.patterns[pattern_id] = self.patterns[temp_key]
            self.patterns[pattern_id].pattern_id = pattern_id
            self.code_to_pattern[code_hash] = pattern_id
            
            # Lösche temporäres Muster
            del self.patterns[temp_key]
            
            logger.info(f"Neues Codemuster erkannt: {pattern_id} (Häufigkeit: {self.patterns[pattern_id].frequency})")
            return pattern_id
        
        return None
    
    def _hash_code(self, code_str: str) -> str:
        """
        Berechnet einen Hash für Code.
        
        Args:
            code_str: Code als String
            
        Returns:
            Code-Hash
        """
        import hashlib
        return hashlib.md5(code_str.encode()).hexdigest()
    
    def _extract_features(self, code_str: str, context: ExecutionContext) -> Dict[str, Any]:
        """
        Extrahiert Merkmale aus Code.
        
        Args:
            code_str: Code als String
            context: Ausführungskontext
            
        Returns:
            Merkmale
        """
        import ast
        
        features = {
            "code_length": len(code_str),
            "has_loops": "for " in code_str or "while " in code_str,
            "has_conditionals": "if " in code_str,
            "has_functions": "def " in code_str or "lambda " in code_str,
            "has_tensor_ops": "tensor" in code_str.lower() or "array" in code_str.lower(),
            "input_dims": sum(len(shape) for shape in context.input_shapes) if context.input_shapes else 0,
            "input_elements": sum(np.prod(shape) for shape in context.input_shapes) if context.input_shapes else 0,
            "execution_time": context.execution_time_ms,
            "memory_usage": context.memory_usage_mb
        }
        
        # Versuche AST-Analyse für tiefere Inspektion
        try:
            tree = ast.parse(code_str)
            
            # Zähle verschiedene Knotentypen
            node_counts = defaultdict(int)
            for node in ast.walk(tree):
                node_counts[type(node).__name__] += 1
            
            # Füge ausgewählte Knotenzahlen zu Features hinzu
            important_nodes = ["For", "While", "If", "Call", "BinOp", "Compare", "List", "Dict", "Tuple"]
            for node_type in important_nodes:
                features[f"count_{node_type}"] = node_counts[node_type]
            
            # Analysiere Funktionsaufrufe
            features["function_calls"] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and hasattr(node.func, "id"):
                    features["function_calls"].append(node.func.id)
            
        except SyntaxError:
            # Fallback, wenn AST-Parsing fehlschlägt
            pass
        
        return features
    
    def _find_similar_pattern(self, features: Dict[str, Any]) -> Optional[str]:
        """
        Sucht nach einem ähnlichen Muster.
        
        Args:
            features: Merkmale
            
        Returns:
            Pattern-ID oder None
        """
        best_match = None
        best_similarity = 0.6  # Schwellenwert für Ähnlichkeit
        
        for pattern_id, pattern in self.patterns.items():
            if pattern_id.startswith("temp_"):
                continue
                
            similarity = self._calculate_similarity(features, pattern.features)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern_id
        
        return best_match
    
    def _calculate_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        Berechnet die Ähnlichkeit zwischen zwei Merkmalssätzen.
        
        Args:
            features1: Erster Merkmalssatz
            features2: Zweiter Merkmalssatz
            
        Returns:
            Ähnlichkeitswert zwischen 0 und 1
        """
        # Gemeinsame Schlüssel finden
        common_keys = set(features1.keys()) & set(features2.keys())
        
        if not common_keys:
            return 0.0
            
        similarity_sum = 0.0
        
        for key in common_keys:
            # Ignoriere Listen und komplexe Typen für einfache Ähnlichkeit
            if isinstance(features1[key], (list, dict, set)):
                continue
                
            # Berechne Ähnlichkeit basierend auf Typ
            if isinstance(features1[key], bool):
                similarity_sum += 1.0 if features1[key] == features2[key] else 0.0
            elif isinstance(features1[key], (int, float)):
                # Normalisiere numerische Werte
                max_val = max(abs(features1[key]), abs(features2[key]))
                if max_val == 0:
                    similarity_sum += 1.0  # Beide sind 0
                else:
                    diff = abs(features1[key] - features2[key]) / max_val
                    similarity_sum += max(0.0, 1.0 - diff)
            else:
                # String-Ähnlichkeit oder andere Typen
                similarity_sum += 1.0 if features1[key] == features2[key] else 0.0
        
        return similarity_sum / len(common_keys)
    
    def get_pattern(self, pattern_id: str) -> Optional[CodePattern]:
        """
        Gibt ein Muster zurück.
        
        Args:
            pattern_id: Pattern-ID
            
        Returns:
            Muster oder None
        """
        return self.patterns.get(pattern_id)
    
    def update_optimal_strategy(self, pattern_id: str, strategy: OptimizationStrategy) -> None:
        """
        Aktualisiert die optimale Strategie für ein Muster.
        
        Args:
            pattern_id: Pattern-ID
            strategy: Optimierungsstrategie
        """
        if pattern_id in self.patterns:
            self.patterns[pattern_id].optimal_strategy = strategy
            logger.info(f"Optimale Strategie für Muster {pattern_id} aktualisiert: {strategy.name}")


class ReinforcementLearner:
    """Reinforcement Learning für Optimierungsstrategien"""
    
    def __init__(self, config: OptimizerConfig):
        """
        Initialisiert einen neuen Reinforcement Learner.
        
        Args:
            config: Optimizer-Konfiguration
        """
        self.config = config
        self.strategies = {}
        self.q_table = {}
        self.memory = []
        self.exploration_rate = config.exploration_rate
        self.learning_rate = config.learning_rate
        self.discount_factor = config.discount_factor
        self.update_count = 0
    
    def register_strategy(self, strategy: OptimizationStrategy) -> None:
        """
        Registriert eine Optimierungsstrategie.
        
        Args:
            strategy: Optimierungsstrategie
        """
        self.strategies[strategy.strategy_id] = strategy
    
    def get_action(self, state: str) -> str:
        """
        Wählt eine Aktion (Strategie) für einen Zustand.
        
        Args:
            state: Zustand (Pattern-ID)
            
        Returns:
            Strategie-ID
        """
        # Initialisiere Q-Werte für neuen Zustand
        if state not in self.q_table:
            self.q_table[state] = {strategy_id: 0.0 for strategy_id in self.strategies}
        
        # Exploration (zufällige Strategie)
        if random.random() < self.exploration_rate:
            return random.choice(list(self.strategies.keys()))
        
        # Exploitation (beste Strategie)
        return max(self.q_table[state].items(), key=lambda x: x[1])[0]
    
    def remember(self, state: str, action: str, reward: float, next_state: str) -> None:
        """
        Speichert eine Erfahrung.
        
        Args:
            state: Aktueller Zustand
            action: Gewählte Aktion
            reward: Erhaltene Belohnung
            next_state: Nächster Zustand
        """
        self.memory.append((state, action, reward, next_state))
        
        # Beschränke Gedächtnisgröße
        if len(self.memory) > self.config.memory_size:
            self.memory.pop(0)
        
        # Aktualisiere Q-Tabelle nach bestimmter Anzahl von Erfahrungen
        self.update_count += 1
        if self.update_count % self.config.update_frequency == 0 and len(self.memory) >= self.config.min_samples:
            self._update_model()
    
    def _update_model(self) -> None:
        """Aktualisiert das Q-Learning-Modell"""
        # Zufällige Stichprobe aus Gedächtnis
        if len(self.memory) < self.config.batch_size:
            batch = self.memory
        else:
            batch = random.sample(self.memory, self.config.batch_size)
        
        for state, action, reward, next_state in batch:
            # Initialisiere Q-Werte für neuen Zustand
            if next_state not in self.q_table:
                self.q_table[next_state] = {strategy_id: 0.0 for strategy_id in self.strategies}
            
            # Q-Learning-Update
            if state not in self.q_table:
                self.q_table[state] = {strategy_id: 0.0 for strategy_id in self.strategies}
                
            # Finde maximalen Q-Wert für nächsten Zustand
            max_next_q = max(self.q_table[next_state].values())
            
            # Update-Formel: Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
            current_q = self.q_table[state][action]
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
            )
            
            self.q_table[state][action] = new_q
        
        # Reduziere Exploration mit der Zeit
        self.exploration_rate = max(0.01, self.exploration_rate * 0.995)
        
        logger.info(f"Modell aktualisiert. Neue Explorationsrate: {self.exploration_rate:.4f}")
    
    def get_best_strategy(self, state: str) -> OptimizationStrategy:
        """
        Gibt die beste Strategie für einen Zustand zurück.
        
        Args:
            state: Zustand (Pattern-ID)
            
        Returns:
            Optimierungsstrategie
        """
        if state not in self.q_table:
            # Wähle Standardstrategie für unbekannten Zustand
            strategy_id = next(iter(self.strategies.keys()))
            return self.strategies[strategy_id]
        
        # Wähle Strategie mit höchstem Q-Wert
        strategy_id = max(self.q_table[state].items(), key=lambda x: x[1])[0]
        strategy = self.strategies[strategy_id]
        
        # Kopiere Strategie und setze Konfidenz
        confidence = self.q_table[state][strategy_id]
        strategy_copy = OptimizationStrategy(
            strategy_id=strategy.strategy_id,
            name=strategy.name,
            parallelization_level=strategy.parallelization_level,
            jit_level=strategy.jit_level,
            device_target=strategy.device_target,
            memory_optimization=strategy.memory_optimization,
            batch_processing=strategy.batch_processing,
            tensor_fusion=strategy.tensor_fusion,
            operator_fusion=strategy.operator_fusion,
            loop_unrolling=strategy.loop_unrolling,
            automatic_differentiation=strategy.automatic_differentiation,
            confidence=confidence
        )
        
        return strategy_copy
    
    def save_model(self, filepath: str) -> None:
        """
        Speichert das Modell.
        
        Args:
            filepath: Dateipfad
        """
        model_data = {
            "q_table": self.q_table,
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "update_count": self.update_count
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Modell gespeichert in {filepath}")
    
    def load_model(self, filepath: str) -> bool:
        """
        Lädt ein Modell.
        
        Args:
            filepath: Dateipfad
            
        Returns:
            True, wenn das Modell erfolgreich geladen wurde, sonst False
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = model_data["q_table"]
            self.exploration_rate = model_data["exploration_rate"]
            self.learning_rate = model_data["learning_rate"]
            self.update_count = model_data["update_count"]
            
            logger.info(f"Modell geladen aus {filepath}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {e}")
            return False


class AIOptimizer:
    """KI-gestützter Optimizer für M-CODE"""
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        Initialisiert einen neuen AI-Optimizer.
        
        Args:
            config: Optimizer-Konfiguration
        """
        self.config = config or OptimizerConfig()
        self.pattern_recognizer = PatternRecognizer()
        self.reinforcement_learner = ReinforcementLearner(self.config)
        self.execution_stats = {}
        self.optimization_cache = {}
        self.warmup_complete = False
        self.lock = threading.RLock()
        
        # Initialisiere Standard-Optimierungsstrategien
        self._initialize_strategies()
        
        # Lade Modell, wenn Pfad angegeben
        if self.config.model_path and os.path.exists(self.config.model_path):
            self.reinforcement_learner.load_model(self.config.model_path)
            self.warmup_complete = True
        
        logger.info(f"AI-Optimizer initialisiert mit Konfiguration: {self.config}")
    
    def _initialize_strategies(self) -> None:
        """Initialisiert Standard-Optimierungsstrategien"""
        # Einfache Strategien für verschiedene Szenarien
        strategies = [
            OptimizationStrategy(
                strategy_id="default",
                name="Standard",
                parallelization_level=0,
                jit_level=0,
                device_target="auto",
                memory_optimization="balanced"
            ),
            OptimizationStrategy(
                strategy_id="cpu_optimized",
                name="CPU-Optimiert",
                parallelization_level=1,
                jit_level=1,
                device_target="cpu",
                memory_optimization="balanced",
                loop_unrolling=True
            ),
            OptimizationStrategy(
                strategy_id="gpu_optimized",
                name="GPU-Optimiert",
                parallelization_level=0,
                jit_level=2,
                device_target="gpu",
                memory_optimization="high",
                tensor_fusion=True,
                batch_processing=True
            ),
            OptimizationStrategy(
                strategy_id="ane_optimized",
                name="Neural-Engine-Optimiert",
                parallelization_level=0,
                jit_level=2,
                device_target="ane",
                memory_optimization="balanced",
                tensor_fusion=True,
                operator_fusion=True
            ),
            OptimizationStrategy(
                strategy_id="memory_efficient",
                name="Speichereffizient",
                parallelization_level=0,
                jit_level=1,
                device_target="cpu",
                memory_optimization="low"
            ),
            OptimizationStrategy(
                strategy_id="parallel_cpu",
                name="Parallele CPU",
                parallelization_level=2,
                jit_level=1,
                device_target="cpu",
                memory_optimization="balanced",
                loop_unrolling=True
            ),
            OptimizationStrategy(
                strategy_id="math_intensive",
                name="Mathematik-Intensiv",
                parallelization_level=1,
                jit_level=2,
                device_target="auto",
                memory_optimization="balanced",
                tensor_fusion=True,
                automatic_differentiation=True
            ),
            OptimizationStrategy(
                strategy_id="aggressive_optimization",
                name="Aggressive Optimierung",
                parallelization_level=2,
                jit_level=2,
                device_target="auto",
                memory_optimization="high",
                tensor_fusion=True,
                operator_fusion=True,
                batch_processing=True,
                loop_unrolling=True,
                automatic_differentiation=True
            )
        ]
        
        # Registriere Strategien beim Reinforcement Learner
        for strategy in strategies:
            self.reinforcement_learner.register_strategy(strategy)
    
    def optimize(self, code_str: str, context: ExecutionContext) -> OptimizationStrategy:
        """
        Optimiert Code mit KI-basierter Strategie.
        
        Args:
            code_str: Code als String
            context: Ausführungskontext
            
        Returns:
            Optimierungsstrategie
        """
        if not self.config.enabled:
            # Return default strategy if disabled
            return self.reinforcement_learner.get_best_strategy("default")
        
        with self.lock:
            # Prüfe Cache für bekannten Code
            code_hash = self._hash_code(code_str)
            if code_hash in self.optimization_cache:
                return self.optimization_cache[code_hash]
            
            # Analysiere Code und erkenne Muster
            pattern_id = self.pattern_recognizer.analyze_code(code_str, context)
            
            if not pattern_id:
                # Verwende Standardstrategie für unbekannte Muster
                strategy = self.reinforcement_learner.get_best_strategy("default")
                return strategy
            
            # Wähle Strategie basierend auf dem Muster
            strategy = self._select_strategy(pattern_id, context)
            
            # Speichere im Cache
            self.optimization_cache[code_hash] = strategy
            
            return strategy
    
    def _hash_code(self, code_str: str) -> str:
        """Berechnet einen Hash für den Code"""
        import hashlib
        return hashlib.md5(code_str.encode()).hexdigest()
    
    def _select_strategy(self, pattern_id: str, context: ExecutionContext) -> OptimizationStrategy:
        """
        Wählt eine Optimierungsstrategie basierend auf dem Muster und Kontext.
        
        Args:
            pattern_id: Muster-ID
            context: Ausführungskontext
            
        Returns:
            Optimierungsstrategie
        """
        # Check if pattern has an optimal strategy
        pattern = self.pattern_recognizer.get_pattern(pattern_id)
        if pattern and pattern.optimal_strategy and pattern.optimal_strategy.confidence > 0.7:
            return pattern.optimal_strategy
        
        # Use reinforcement learning to select strategy
        strategy_id = self.reinforcement_learner.get_action(pattern_id)
        strategy = self.reinforcement_learner.strategies[strategy_id]
        
        # Adaptive optimization based on hardware capabilities
        strategy = self._adapt_to_hardware(strategy)
        
        return strategy
    
    def _adapt_to_hardware(self, strategy: OptimizationStrategy) -> OptimizationStrategy:
        """
        Passt eine Strategie an die verfügbare Hardware an.
        
        Args:
            strategy: Ursprüngliche Strategie
            
        Returns:
            Angepasste Strategie
        """
        # Create a copy to avoid modifying the original
        adapted = OptimizationStrategy(
            strategy_id=strategy.strategy_id,
            name=strategy.name,
            parallelization_level=strategy.parallelization_level,
            jit_level=strategy.jit_level,
            device_target=strategy.device_target,
            memory_optimization=strategy.memory_optimization,
            batch_processing=strategy.batch_processing,
            tensor_fusion=strategy.tensor_fusion,
            operator_fusion=strategy.operator_fusion,
            loop_unrolling=strategy.loop_unrolling,
            automatic_differentiation=strategy.automatic_differentiation,
            confidence=strategy.confidence
        )
        
        # Check MLX availability
        if not MLX_AVAILABLE and adapted.device_target in ["gpu", "ane"]:
            adapted.device_target = "cpu"
            adapted.jit_level = min(adapted.jit_level, 1)  # Reduce JIT level
        
        # Check if Apple Neural Engine is available
        mlx_adapter = get_mlx_adapter()
        if adapted.device_target == "ane" and not mlx_adapter.supports_ane():
            adapted.device_target = "gpu" if mlx_adapter.is_available() else "cpu"
        
        # Adjust parallelization based on CPU count
        cpu_count = os.cpu_count() or 4
        if adapted.parallelization_level > 0 and cpu_count < 4:
            # Reduce parallelization on low-core systems
            adapted.parallelization_level = 1
        
        return adapted
    
    def feedback(self, code_str: str, strategy: OptimizationStrategy, execution_time: float, success: bool) -> None:
        """
        Gibt Feedback zur angewendeten Strategie.
        
        Args:
            code_str: Code als String
            strategy: Verwendete Strategie
            execution_time: Ausführungszeit in ms
            success: Ob die Ausführung erfolgreich war
        """
        if not self.config.enabled:
            return
        
        with self.lock:
            # Analysiere Code und erkenne Muster
            pattern_id = None
            for hash_code, cached_strategy in self.optimization_cache.items():
                if cached_strategy.strategy_id == strategy.strategy_id:
                    # Find the pattern that led to this strategy
                    context = ExecutionContext(
                        code_hash=hash_code,
                        input_shapes=[],
                        input_types=[],
                        execution_time_ms=execution_time,
                        success=success
                    )
                    pattern_id = self.pattern_recognizer.analyze_code(code_str, context)
                    break
            
            if not pattern_id:
                # No pattern found, can't provide good feedback
                return
            
            # Calculate reward based on execution time and success
            reward = self._calculate_reward(execution_time, success, strategy)
            
            # Provide feedback to reinforcement learner
            next_state = pattern_id  # In this simple case, state doesn't change
            self.reinforcement_learner.remember(
                state=pattern_id,
                action=strategy.strategy_id,
                reward=reward,
                next_state=next_state
            )
            
            # Update pattern's optimal strategy if this was particularly good
            if reward > 0.8:
                pattern = self.pattern_recognizer.get_pattern(pattern_id)
                if pattern:
                    if not pattern.optimal_strategy or reward > pattern.optimal_strategy.confidence:
                        strategy.confidence = reward
                        self.pattern_recognizer.update_optimal_strategy(pattern_id, strategy)
    
    def _calculate_reward(self, execution_time: float, success: bool, strategy: OptimizationStrategy) -> float:
        """
        Berechnet die Belohnung für eine Strategie.
        
        Args:
            execution_time: Ausführungszeit in ms
            success: Ob die Ausführung erfolgreich war
            strategy: Verwendete Strategie
            
        Returns:
            Belohnung zwischen -1 und 1
        """
        if not success:
            # Failed execution gets negative reward
            return -0.5
        
        # Normalize execution time (lower is better)
        # We use a reference time of 100ms as "average" performance
        normalized_time = min(1.0, 100.0 / max(1.0, execution_time))
        
        # Base reward on execution time
        reward = normalized_time
        
        # Adjust reward based on strategy complexity
        # More complex strategies should deliver better performance to be worth it
        complexity_factor = 1.0
        if strategy.jit_level > 0:
            complexity_factor += 0.1 * strategy.jit_level
        if strategy.parallelization_level > 0:
            complexity_factor += 0.1 * strategy.parallelization_level
        if strategy.tensor_fusion or strategy.operator_fusion:
            complexity_factor += 0.1
        
        # Higher complexity needs better performance to be rewarded
        reward = reward / complexity_factor
        
        # Ensure reward is in [-1, 1] range
        return max(-1.0, min(1.0, reward))
    
    def save_state(self, directory: str) -> None:
        """
        Speichert den Zustand des Optimizers.
        
        Args:
            directory: Verzeichnispfad
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save reinforcement learning model
        model_path = os.path.join(directory, "ai_optimizer_model.pkl")
        self.reinforcement_learner.save_model(model_path)
        
        # Save pattern database
        patterns_path = os.path.join(directory, "ai_optimizer_patterns.json")
        patterns_data = {}
        for pattern_id, pattern in self.pattern_recognizer.patterns.items():
            # Convert pattern to serializable format
            pattern_data = {
                "pattern_id": pattern.pattern_id,
                "name": pattern.name,
                "description": pattern.description,
                "frequency": pattern.frequency,
                "avg_execution_time": pattern.avg_execution_time
            }
            
            # Add optimal strategy if exists
            if pattern.optimal_strategy:
                pattern_data["optimal_strategy"] = {
                    "strategy_id": pattern.optimal_strategy.strategy_id,
                    "confidence": pattern.optimal_strategy.confidence
                }
                
            patterns_data[pattern_id] = pattern_data
        
        with open(patterns_path, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        logger.info(f"AI-Optimizer-Zustand gespeichert in {directory}")
    
    def load_state(self, directory: str) -> bool:
        """
        Lädt den Zustand des Optimizers.
        
        Args:
            directory: Verzeichnispfad
            
        Returns:
            True, wenn der Zustand erfolgreich geladen wurde, sonst False
        """
        try:
            # Load reinforcement learning model
            model_path = os.path.join(directory, "ai_optimizer_model.pkl")
            if os.path.exists(model_path):
                self.reinforcement_learner.load_model(model_path)
            
            # Load pattern database
            patterns_path = os.path.join(directory, "ai_optimizer_patterns.json")
            if os.path.exists(patterns_path):
                with open(patterns_path, 'r') as f:
                    patterns_data = json.load(f)
                
                # Recreate patterns
                for pattern_id, pattern_data in patterns_data.items():
                    # We need to reconstruct the pattern with minimal data
                    pattern = CodePattern(
                        pattern_id=pattern_data["pattern_id"],
                        name=pattern_data["name"],
                        description=pattern_data.get("description", ""),
                        features={},  # Empty features, will be populated by code analysis
                        frequency=pattern_data.get("frequency", 0),
                        avg_execution_time=pattern_data.get("avg_execution_time", 0.0)
                    )
                    
                    # Set optimal strategy if exists
                    if "optimal_strategy" in pattern_data:
                        strategy_id = pattern_data["optimal_strategy"]["strategy_id"]
                        if strategy_id in self.reinforcement_learner.strategies:
                            strategy = self.reinforcement_learner.strategies[strategy_id]
                            strategy.confidence = pattern_data["optimal_strategy"].get("confidence", 0.0)
                            pattern.optimal_strategy = strategy
                    
                    self.pattern_recognizer.patterns[pattern_id] = pattern
            
            self.warmup_complete = True
            logger.info(f"AI-Optimizer-Zustand geladen aus {directory}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Laden des AI-Optimizer-Zustands: {e}")
            return False
    
    @profile(name="ai_optimizer_apply")
    def apply_strategy(self, strategy: OptimizationStrategy, executor_function: Callable, *args, **kwargs) -> Any:
        """
        Wendet eine Optimierungsstrategie auf eine Funktion an.
        
        Args:
            strategy: Optimierungsstrategie
            executor_function: Auszuführende Funktion
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
            
        Returns:
            Rückgabewert der Funktion
        """
        start_time = time.time()
        result = None
        success = False
        
        try:
            # Get required components
            jit_compiler = get_jit_compiler()
            
            # Apply JIT optimization if needed
            if strategy.jit_level > 0:
                executor_function = jit_compiler.compile_function(executor_function, optimize=strategy.jit_level > 1)
            
            # Apply parallelization if needed
            if strategy.parallelization_level > 0:
                from .parallel_executor import parallel
                use_processes = strategy.parallelization_level > 1
                executor_function = parallel(executor_function, use_processes=use_processes)
            
            # Execute function with strategy applied
            result = executor_function(*args, **kwargs)
            success = True
            return result
        
        except Exception as e:
            logger.error(f"Fehler bei der Anwendung der Strategie {strategy.name}: {e}")
            # Propagate the exception after recording the failure
            raise
        
        finally:
            # Record execution stats
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            self.feedback(
                code_str=str(executor_function),
                strategy=strategy,
                execution_time=execution_time,
                success=success
            )


# Singleton-Instanz des AI-Optimizers
_ai_optimizer_instance = None

def get_ai_optimizer(config: Optional[OptimizerConfig] = None) -> AIOptimizer:
    """
    Gibt eine Singleton-Instanz des AI-Optimizers zurück.
    
    Args:
        config: Optimizer-Konfiguration
        
    Returns:
        AI-Optimizer
    """
    global _ai_optimizer_instance
    
    if _ai_optimizer_instance is None:
        _ai_optimizer_instance = AIOptimizer(config)
        
    return _ai_optimizer_instance


def optimize(func=None, *, strategy_id=None):
    """
    Dekorator für die KI-gestützte Optimierung von Funktionen.
    
    Args:
        func: Zu dekorierende Funktion
        strategy_id: Optionale Strategie-ID
        
    Returns:
        Dekorierte Funktion
    """
    def decorator(f):
        import functools
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Get AI optimizer
            optimizer = get_ai_optimizer()
            
            # Create execution context
            context = ExecutionContext(
                code_hash=optimizer._hash_code(inspect.getsource(f)),
                input_shapes=[],
                input_types=[]
            )
            
            # Select optimization strategy
            if strategy_id:
                # Use specified strategy if available
                strategy = optimizer.reinforcement_learner.strategies.get(
                    strategy_id,
                    optimizer.reinforcement_learner.get_best_strategy("default")
                )
            else:
                # Let the optimizer choose the best strategy
                strategy = optimizer.optimize(inspect.getsource(f), context)
            
            # Apply strategy and execute function
            return optimizer.apply_strategy(strategy, f, *args, **kwargs)
        
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)
