#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDD Tests for Real Benchmark Data Loaders
Test-Driven Development für echte Benchmark-Daten

Tests für:
- HellaSwag (Common Sense Reasoning)
- SWE-bench (Software Engineering)
- ARB (Advanced Reasoning)
- MEGAVERSE (Multilingual/Multimodal)
- AILuminate (AI Safety)

Copyright (c) 2025 VXOR AI. Alle Rechte vorbehalten.
"""

import unittest
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add vxor_clean to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.real_benchmark_data import RealBenchmarkLoader

class TestRealBenchmarkDataLoaders(unittest.TestCase):
    """
    TDD Tests für Real Benchmark Data Loaders
    
    RED-GREEN-REFACTOR Cycle:
    1. RED: Failing tests first
    2. GREEN: Minimum implementation
    3. REFACTOR: Clean up code
    """
    
    def setUp(self):
        """Setup für jeden Test"""
        self.loader = RealBenchmarkLoader()
        
    def test_load_hellaswag_data_structure(self):
        """
        RED PHASE: Test HellaSwag data structure
        
        Expected structure:
        {
            "ctx": str,           # Context/scenario
            "endings": List[str], # 4 possible endings
            "correct": int,       # Index of correct ending (0-3)
            "activity": str,      # Activity description
            "source": str         # Data source
        }
        """
        # This test will FAIL initially (RED phase)
        data = self.loader.load_hellaswag_data(num_samples=5)
        
        # Verify basic structure
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 5)
        
        # Verify each question structure
        for question in data:
            self.assertIsInstance(question, dict)
            
            # Required fields
            self.assertIn("ctx", question)
            self.assertIn("endings", question)
            self.assertIn("correct", question)
            self.assertIn("activity", question)
            self.assertIn("source", question)
            
            # Field types
            self.assertIsInstance(question["ctx"], str)
            self.assertIsInstance(question["endings"], list)
            self.assertIsInstance(question["correct"], int)
            self.assertIsInstance(question["activity"], str)
            self.assertIsInstance(question["source"], str)
            
            # Content validation
            self.assertEqual(len(question["endings"]), 4)  # Always 4 choices
            self.assertIn(question["correct"], [0, 1, 2, 3])  # Valid index
            self.assertGreater(len(question["ctx"]), 10)  # Meaningful context
            
    def test_load_swe_bench_data_structure(self):
        """
        RED PHASE: Test SWE-bench data structure
        
        Expected structure:
        {
            "issue_id": str,           # Unique issue identifier
            "repo": str,               # Repository name
            "problem_statement": str,  # Problem description
            "patch": str,              # Solution patch
            "test_patch": str,         # Test case
            "difficulty": str,         # easy/medium/hard
            "category": str            # bug_fix/enhancement/feature
        }
        """
        # This test will FAIL initially (RED phase)
        data = self.loader.load_swe_bench_data(num_samples=3)
        
        # Verify basic structure
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 3)
        
        # Verify each issue structure
        for issue in data:
            self.assertIsInstance(issue, dict)
            
            # Required fields
            required_fields = ["issue_id", "repo", "problem_statement", 
                             "patch", "test_patch", "difficulty", "category"]
            for field in required_fields:
                self.assertIn(field, issue)
                self.assertIsInstance(issue[field], str)
                
            # Content validation
            self.assertIn(issue["difficulty"], ["easy", "medium", "hard"])
            self.assertIn(issue["category"], ["bug_fix", "enhancement", "feature"])
            self.assertGreater(len(issue["problem_statement"]), 20)
            
    def test_load_arb_data_structure(self):
        """
        RED PHASE: Test ARB (Advanced Reasoning) data structure
        
        Expected structure:
        {
            "problem_id": str,         # Unique problem identifier
            "domain": str,             # mathematics/physics/biology/law
            "problem_text": str,       # Problem description
            "reasoning_steps": List[str], # Step-by-step reasoning
            "answer": str,             # Correct answer
            "difficulty": str,         # expert/advanced/intermediate
            "reasoning_type": str      # deductive/inductive/abductive
        }
        """
        # This test will FAIL initially (RED phase)
        data = self.loader.load_arb_data(num_samples=4)
        
        # Verify basic structure
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 4)
        
        # Verify each problem structure
        for problem in data:
            self.assertIsInstance(problem, dict)
            
            # Required fields
            required_fields = ["problem_id", "domain", "problem_text", 
                             "reasoning_steps", "answer", "difficulty", "reasoning_type"]
            for field in required_fields:
                self.assertIn(field, problem)
                
            # Field types
            self.assertIsInstance(problem["reasoning_steps"], list)
            self.assertGreater(len(problem["reasoning_steps"]), 0)
            
            # Content validation
            self.assertIn(problem["domain"], ["mathematics", "physics", "biology", "law"])
            self.assertIn(problem["difficulty"], ["expert", "advanced", "intermediate"])
            self.assertIn(problem["reasoning_type"], ["deductive", "inductive", "abductive"])
            
    def test_load_megaverse_data_structure(self):
        """
        RED PHASE: Test MEGAVERSE (Multilingual/Multimodal) data structure
        
        Expected structure:
        {
            "task_id": str,            # Unique task identifier
            "language": str,           # Language code (en, de, fr, etc.)
            "modality": str,           # text/image/audio/multimodal
            "task_type": str,          # translation/qa/classification
            "input_text": str,         # Input content
            "target_output": str,      # Expected output
            "difficulty": str,         # high_resource/low_resource
            "cultural_context": str    # Cultural context info
        }
        """
        # This test will FAIL initially (RED phase)
        data = self.loader.load_megaverse_data(num_samples=6)
        
        # Verify basic structure
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 6)
        
        # Verify each task structure
        for task in data:
            self.assertIsInstance(task, dict)
            
            # Required fields
            required_fields = ["task_id", "language", "modality", "task_type",
                             "input_text", "target_output", "difficulty", "cultural_context"]
            for field in required_fields:
                self.assertIn(field, task)
                self.assertIsInstance(task[field], str)
                
            # Content validation
            self.assertIn(task["modality"], ["text", "image", "audio", "multimodal"])
            self.assertIn(task["task_type"], ["translation", "qa", "classification", "generation"])
            self.assertIn(task["difficulty"], ["high_resource", "low_resource"])
            self.assertEqual(len(task["language"]), 2)  # Language code format
            
    def test_load_ai_luminate_data_structure(self):
        """
        RED PHASE: Test AILuminate (AI Safety) data structure
        
        Expected structure:
        {
            "prompt_id": str,          # Unique prompt identifier
            "prompt_text": str,        # Input prompt to test
            "safety_category": str,    # hate_speech/violence/self_harm/privacy
            "expected_response": str,  # Safe response type
            "risk_level": str,         # low/medium/high/critical
            "should_refuse": bool,     # Whether AI should refuse
            "evaluation_criteria": List[str] # Evaluation criteria
        }
        """
        # This test will FAIL initially (RED phase)
        data = self.loader.load_ai_luminate_data(num_samples=5)
        
        # Verify basic structure
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 5)
        
        # Verify each prompt structure
        for prompt in data:
            self.assertIsInstance(prompt, dict)
            
            # Required fields
            required_fields = ["prompt_id", "prompt_text", "safety_category",
                             "expected_response", "risk_level", "should_refuse", "evaluation_criteria"]
            for field in required_fields:
                self.assertIn(field, prompt)
                
            # Field types
            self.assertIsInstance(prompt["should_refuse"], bool)
            self.assertIsInstance(prompt["evaluation_criteria"], list)
            
            # Content validation
            safety_categories = ["hate_speech", "violence", "self_harm", "privacy", "misinformation"]
            self.assertIn(prompt["safety_category"], safety_categories)
            self.assertIn(prompt["risk_level"], ["low", "medium", "high", "critical"])
            self.assertGreater(len(prompt["evaluation_criteria"]), 0)
            
    def test_data_loading_performance(self):
        """
        RED PHASE: Test that data loading takes realistic time (not 0.00s)
        """
        # Test each loader for realistic execution time
        loaders = [
            ("HellaSwag", lambda: self.loader.load_hellaswag_data(10)),
            ("SWE-bench", lambda: self.loader.load_swe_bench_data(5)),
            ("ARB", lambda: self.loader.load_arb_data(5)),
            ("MEGAVERSE", lambda: self.loader.load_megaverse_data(8)),
            ("AILuminate", lambda: self.loader.load_ai_luminate_data(6))
        ]
        
        for name, loader_func in loaders:
            start_time = time.time()
            data = loader_func()
            execution_time = time.time() - start_time
            
            # Should take some measurable time (not instant simulation)
            self.assertGreater(execution_time, 0.0001, f"{name} should take measurable time")
            self.assertLess(execution_time, 1.0, f"{name} should complete within 1 second")
            
            # Should return valid data
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
            
    def test_data_content_quality(self):
        """
        RED PHASE: Test that loaded data contains realistic, high-quality content
        """
        # HellaSwag content quality
        hellaswag_data = self.loader.load_hellaswag_data(3)
        for question in hellaswag_data:
            # Context should be meaningful
            self.assertGreater(len(question["ctx"].split()), 5)
            # All endings should be different
            self.assertEqual(len(set(question["endings"])), 4)
            
        # SWE-bench content quality
        swe_data = self.loader.load_swe_bench_data(2)
        for issue in swe_data:
            # Should contain technical terms
            technical_terms = ["error", "bug", "fix", "implement", "test", "function", "class", "method"]
            problem_text = issue["problem_statement"].lower()
            has_technical_term = any(term in problem_text for term in technical_terms)
            self.assertTrue(has_technical_term, "SWE-bench should contain technical content")
            
        # ARB content quality
        arb_data = self.loader.load_arb_data(2)
        for problem in arb_data:
            # Should have multiple reasoning steps
            self.assertGreaterEqual(len(problem["reasoning_steps"]), 2)
            # Problem should be substantial
            self.assertGreater(len(problem["problem_text"]), 50)

if __name__ == "__main__":
    # Run TDD tests
    unittest.main(verbosity=2)
