#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR AI Tests - TDD Implementation
Enterprise-Grade Neural Engine Tests

Copyright (c) 2025 VXOR AI. Alle Rechte vorbehalten.
"""

import unittest
import sys
import time
import numpy as np
from pathlib import Path

# Add vxor_clean to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai import (
    AIEngine, AIConfig, AIBackend, ModelType,
    AdaptiveTransformer, get_ai_engine, reset_ai_engine,
    create_model, get_model, benchmark_models, get_system_info
)

class TestAIConfig(unittest.TestCase):
    """Test AI Configuration"""
    
    def test_default_config(self):
        """Test default AI configuration"""
        config = AIConfig()
        
        self.assertEqual(config.backend, AIBackend.AUTO)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.precision, "float32")
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.optimization_level, 2)
        self.assertTrue(config.enable_quantization)
        self.assertTrue(config.enable_pruning)
        self.assertTrue(config.enable_nas)
        self.assertTrue(config.cache_models)
        
    def test_custom_config(self):
        """Test custom AI configuration"""
        config = AIConfig(
            backend=AIBackend.NUMPY,
            device="cpu",
            batch_size=64,
            learning_rate=0.01,
            optimization_level=3,
            enable_quantization=False
        )
        
        self.assertEqual(config.backend, AIBackend.NUMPY)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.optimization_level, 3)
        self.assertFalse(config.enable_quantization)

class TestAdaptiveTransformer(unittest.TestCase):
    """Test Adaptive Transformer Model"""
    
    def setUp(self):
        """Setup test environment"""
        self.config = AIConfig(backend=AIBackend.NUMPY, optimization_level=0)
        self.model = AdaptiveTransformer(self.config)
        
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.backend, "numpy")
        self.assertEqual(self.model.device, "cpu")
        self.assertFalse(self.model.is_trained)
        self.assertIsNone(self.model.model)
        
    def test_backend_selection(self):
        """Test backend selection logic"""
        # Test AUTO backend selection
        auto_config = AIConfig(backend=AIBackend.AUTO)
        auto_model = AdaptiveTransformer(auto_config)
        self.assertIn(auto_model.backend, ["numpy", "torch", "mlx"])
        
        # Test specific backend
        numpy_config = AIConfig(backend=AIBackend.NUMPY)
        numpy_model = AdaptiveTransformer(numpy_config)
        self.assertEqual(numpy_model.backend, "numpy")
        
    def test_device_selection(self):
        """Test device selection logic"""
        model = AdaptiveTransformer(self.config)
        self.assertIn(model.device, ["cpu", "cuda", "mps"])
        
    def test_model_building(self):
        """Test model building"""
        input_shape = (224, 224, 3)
        output_shape = (1000,)
        
        model_obj = self.model.build_model(input_shape, output_shape)
        self.assertIsNotNone(model_obj)
        self.assertIsNotNone(self.model.model)
        
    def test_forward_pass(self):
        """Test forward pass"""
        input_shape = (224, 224, 3)
        output_shape = (1000,)
        
        self.model.build_model(input_shape, output_shape)
        
        # Test with numpy array
        test_input = np.random.randn(1, *input_shape)
        output = self.model.forward(test_input)
        self.assertIsNotNone(output)
        
    def test_training_step(self):
        """Test training step"""
        input_shape = (224, 224, 3)
        output_shape = (1000,)
        
        self.model.build_model(input_shape, output_shape)
        
        # Test training step
        test_input = np.random.randn(1, *input_shape)
        test_target = np.random.randn(1, *output_shape)
        
        train_results = self.model.train_step(test_input, test_target)
        
        self.assertIn("loss", train_results)
        self.assertIn("learning_rate", train_results)
        self.assertIsInstance(train_results["loss"], float)
        
    def test_model_optimization(self):
        """Test model optimization"""
        input_shape = (224, 224, 3)
        output_shape = (1000,)
        
        self.model.build_model(input_shape, output_shape)
        
        # Test optimization
        optimization_config = AIConfig(
            backend=AIBackend.NUMPY,
            enable_quantization=True,
            enable_pruning=True,
            enable_nas=True
        )
        
        optimized_model = AdaptiveTransformer(optimization_config)
        optimized_model.build_model(input_shape, output_shape)
        
        optimization_results = optimized_model.optimize_model()
        
        self.assertIn("original_size", optimization_results)
        self.assertIn("optimized_size", optimization_results)
        self.assertIn("optimizations_applied", optimization_results)
        self.assertIn("compression_ratio", optimization_results)
        
    def test_model_benchmarking(self):
        """Test model benchmarking"""
        input_shape = (224, 224, 3)
        output_shape = (1000,)
        
        self.model.build_model(input_shape, output_shape)
        
        # Benchmark model
        benchmark_results = self.model.benchmark(input_shape, num_iterations=10)
        
        self.assertIn("avg_inference_time_ms", benchmark_results)
        self.assertIn("throughput_fps", benchmark_results)
        self.assertIn("model_size", benchmark_results)
        self.assertIn("backend", benchmark_results)
        self.assertIn("device", benchmark_results)
        
        # Verify reasonable performance
        self.assertGreater(benchmark_results["throughput_fps"], 0)
        self.assertGreater(benchmark_results["avg_inference_time_ms"], 0)

class TestAIEngine(unittest.TestCase):
    """Test AI Engine"""
    
    def setUp(self):
        """Setup test environment"""
        reset_ai_engine()
        self.config = AIConfig(backend=AIBackend.NUMPY, optimization_level=1)
        self.engine = AIEngine(self.config)
        
    def tearDown(self):
        """Cleanup after test"""
        reset_ai_engine()
        
    def test_engine_initialization(self):
        """Test AI engine initialization"""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.config.backend, AIBackend.NUMPY)
        self.assertEqual(len(self.engine.models), 0)
        self.assertEqual(len(self.engine.performance_history), 0)
        
    def test_model_creation(self):
        """Test model creation"""
        model = self.engine.create_model(
            name="test_transformer",
            model_type=ModelType.TRANSFORMER,
            input_shape=(224, 224, 3),
            output_shape=(1000,)
        )
        
        self.assertIsNotNone(model)
        self.assertIsInstance(model, AdaptiveTransformer)
        self.assertIn("test_transformer", self.engine.models)
        
    def test_model_retrieval(self):
        """Test model retrieval"""
        # Create model first
        self.engine.create_model(
            name="test_model",
            model_type=ModelType.ADAPTIVE,
            input_shape=(128, 128, 3),
            output_shape=(10,)
        )
        
        # Retrieve model
        retrieved_model = self.engine.get_model("test_model")
        self.assertIsNotNone(retrieved_model)
        self.assertIsInstance(retrieved_model, AdaptiveTransformer)
        
        # Test non-existent model
        non_existent = self.engine.get_model("non_existent")
        self.assertIsNone(non_existent)
        
    def test_model_benchmarking(self):
        """Test model benchmarking"""
        # Create test models
        self.engine.create_model(
            name="model1",
            model_type=ModelType.TRANSFORMER,
            input_shape=(224, 224, 3),
            output_shape=(1000,)
        )
        
        self.engine.create_model(
            name="model2",
            model_type=ModelType.ADAPTIVE,
            input_shape=(224, 224, 3),
            output_shape=(100,)
        )
        
        # Benchmark all models
        benchmark_results = self.engine.benchmark_all_models()
        
        self.assertIn("model1", benchmark_results)
        self.assertIn("model2", benchmark_results)
        
        for model_name, results in benchmark_results.items():
            if "error" not in results:
                self.assertIn("throughput_fps", results)
                self.assertIn("avg_inference_time_ms", results)
                
    def test_system_info(self):
        """Test system info retrieval"""
        system_info = self.engine.get_system_info()
        
        self.assertIn("backend_support", system_info)
        self.assertIn("config", system_info)
        self.assertIn("models", system_info)
        self.assertIn("performance_history_size", system_info)
        
        # Check backend support
        backend_support = system_info["backend_support"]
        self.assertIn("torch", backend_support)
        self.assertIn("mlx", backend_support)
        self.assertIn("numpy", backend_support)
        self.assertTrue(backend_support["numpy"])  # NumPy should always be available

class TestGlobalAIEngine(unittest.TestCase):
    """Test Global AI Engine Functions"""
    
    def setUp(self):
        """Setup test environment"""
        reset_ai_engine()
        
    def tearDown(self):
        """Cleanup after test"""
        reset_ai_engine()
        
    def test_global_engine_singleton(self):
        """Test global engine singleton behavior"""
        engine1 = get_ai_engine()
        engine2 = get_ai_engine()
        
        self.assertIs(engine1, engine2)
        
    def test_global_engine_reset(self):
        """Test global engine reset"""
        engine1 = get_ai_engine()
        reset_ai_engine()
        engine2 = get_ai_engine()
        
        self.assertIsNot(engine1, engine2)
        
    def test_convenience_functions(self):
        """Test convenience functions"""
        # Test model creation
        model = create_model(
            name="convenience_test",
            model_type=ModelType.TRANSFORMER,
            input_shape=(64, 64, 3),
            output_shape=(10,)
        )
        
        self.assertIsNotNone(model)
        
        # Test model retrieval
        retrieved_model = get_model("convenience_test")
        self.assertIs(model, retrieved_model)
        
        # Test system info
        system_info = get_system_info()
        self.assertIn("backend_support", system_info)
        
        # Test benchmarking
        benchmark_results = benchmark_models()
        self.assertIn("convenience_test", benchmark_results)

class TestAIPerformance(unittest.TestCase):
    """Test AI Performance"""
    
    def test_model_creation_performance(self):
        """Test model creation performance"""
        config = AIConfig(backend=AIBackend.NUMPY, optimization_level=0)
        engine = AIEngine(config)
        
        start_time = time.time()
        
        model = engine.create_model(
            name="perf_test",
            model_type=ModelType.TRANSFORMER,
            input_shape=(224, 224, 3),
            output_shape=(1000,)
        )
        
        creation_time = time.time() - start_time
        
        self.assertLess(creation_time, 5.0)  # Should create model in under 5 seconds
        self.assertIsNotNone(model)
        
    def test_inference_performance(self):
        """Test inference performance"""
        config = AIConfig(backend=AIBackend.NUMPY)
        engine = AIEngine(config)
        
        model = engine.create_model(
            name="inference_test",
            model_type=ModelType.TRANSFORMER,
            input_shape=(224, 224, 3),
            output_shape=(1000,)
        )
        
        # Benchmark inference
        benchmark_results = model.benchmark((224, 224, 3), num_iterations=50)
        
        # Should achieve reasonable performance
        self.assertGreater(benchmark_results["throughput_fps"], 1.0)  # At least 1 FPS
        self.assertLess(benchmark_results["avg_inference_time_ms"], 1000.0)  # Under 1 second

if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during tests
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)
