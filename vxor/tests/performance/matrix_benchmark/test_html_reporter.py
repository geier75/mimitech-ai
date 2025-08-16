#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test for the HTMLReporter functionality.

This test module validates all aspects of the HTMLReporter including:
1. Data processing and chart generation
2. Template rendering
3. Historical data handling
4. System information gathering
5. Recommendation generation
"""

import os
import sys
import json
import unittest
import tempfile
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the HTMLReporter class
from matrix_benchmark.reporters.html_reporter import HTMLReporter

# Mock classes for testing
@dataclass
class MockOperation:
    name: str
    
@dataclass
class MockBackend:
    name: str
    
@dataclass
class MockPrecision:
    name: str
    
@dataclass
class MockBenchmarkResult:
    operation: MockOperation
    backend: MockBackend
    dimension: int
    precision: MockPrecision
    mean_time: float
    std_dev: float = 0.001
    min_time: float = field(default=0.0)
    max_time: float = field(default=0.0)
    success_rate: float = 100.0
    memory_change: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.min_time == 0.0:
            self.min_time = self.mean_time * 0.9
        if self.max_time == 0.0:
            self.max_time = self.mean_time * 1.1


class TestHTMLReporter(unittest.TestCase):
    """Test cases for the HTMLReporter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reporter = HTMLReporter(use_plotly=True, include_historical=True, theme="light")
        self.output_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.output_dir, "test_report.html")
        
        # Create historical data directory
        self.history_dir = os.path.join(self.output_dir, "benchmark_history")
        os.makedirs(self.history_dir, exist_ok=True)
        
        # Generate test benchmark results
        self.results = self._generate_test_results()
        
        # Generate historical data
        self._generate_historical_data()
        
        logger.info(f"Test setup complete. Output file: {self.output_file}")
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up files in the test directory
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
            
        # Clean up historical data files
        for file in os.listdir(self.history_dir):
            os.remove(os.path.join(self.history_dir, file))
            
        # Remove directories
        os.rmdir(self.history_dir)
        os.rmdir(self.output_dir)
        
        logger.info("Test cleanup complete")
    
    def _generate_test_results(self) -> List[MockBenchmarkResult]:
        """Generate mock benchmark results for testing."""
        operations = [MockOperation(name) for name in ["matmul", "add", "transpose", "inverse"]]
        backends = [MockBackend(name) for name in ["PyTorch", "NumPy", "MLX"]]
        precisions = [MockPrecision(name) for name in ["FLOAT32", "FLOAT64", "BFLOAT16"]]
        dimensions = [16, 32, 64, 128, 256, 512]
        
        results = []
        # Generate results for all combinations
        for op in operations:
            for backend in backends:
                for dim in dimensions:
                    for precision in precisions:
                        # Skip some combinations to make it more realistic
                        if backend.name == "MLX" and precision.name == "FLOAT64":
                            continue
                            
                        # Base time depends on operation and dimension
                        base_time = 0
                        if op.name == "matmul":
                            # O(n^3) complexity
                            base_time = (dim / 100) ** 3 * 0.001
                        elif op.name == "inverse":
                            # O(n^3) complexity but more expensive
                            base_time = (dim / 100) ** 3 * 0.002
                        elif op.name == "transpose":
                            # O(n^2) complexity
                            base_time = (dim / 100) ** 2 * 0.0005
                        else:  # add
                            # O(n^2) complexity
                            base_time = (dim / 100) ** 2 * 0.0002
                        
                        # Backend-specific speedups
                        if backend.name == "PyTorch":
                            base_time *= 0.8  # 20% faster than baseline
                        elif backend.name == "MLX":
                            base_time *= 0.5  # 50% faster than baseline
                        
                        # Precision-specific adjustments
                        if precision.name == "FLOAT64":
                            base_time *= 1.2  # 20% slower for higher precision
                        elif precision.name == "BFLOAT16":
                            base_time *= 0.9  # 10% faster for lower precision
                        
                        # Memory change (in bytes)
                        memory_change = dim * dim * 4  # 4 bytes per float32
                        if precision.name == "FLOAT64":
                            memory_change *= 2
                        elif precision.name == "BFLOAT16":
                            memory_change //= 2
                            
                        # Double memory for matmul (stores result)
                        if op.name == "matmul" or op.name == "inverse":
                            memory_change *= 2
                        
                        # Create result
                        result = MockBenchmarkResult(
                            operation=op,
                            backend=backend,
                            dimension=dim,
                            precision=precision,
                            mean_time=base_time,
                            std_dev=base_time * 0.1,  # 10% std deviation
                            memory_change=memory_change
                        )
                        results.append(result)
        
        logger.info(f"Generated {len(results)} mock benchmark results")
        return results
    
    def _generate_historical_data(self):
        """Generate historical benchmark data."""
        # Generate 3 historical datasets with timestamps from past days
        for i in range(3):
            # Create a copy of results with slightly different times
            historical_results = []
            for result in self.results:
                # Vary the times based on "age" of the dataset
                time_factor = 1.0 + (i * 0.2)  # Older results are slower
                
                # MLX improvement over time (newer = faster)
                if result.backend.name == "MLX":
                    time_factor -= 0.1 * (3 - i)
                
                # Create a modified result
                historical_result = {
                    'operation': result.operation.name,
                    'backend': result.backend.name,
                    'dimension': result.dimension,
                    'precision': result.precision.name,
                    'mean_time': result.mean_time * time_factor,
                    'std_dev': result.std_dev * time_factor,
                    'min_time': result.min_time * time_factor,
                    'max_time': result.max_time * time_factor,
                    'success_rate': result.success_rate,
                    'memory_change': result.memory_change
                }
                historical_results.append(historical_result)
            
            # Create a timestamp for this historical dataset
            days_ago = (i + 1) * 7  # Weekly intervals
            timestamp = (datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) - 
                        timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
            
            # Create the historical dataset
            historical_data = {
                'timestamp': timestamp,
                'results': historical_results,
                'metadata': {
                    'system_info': {
                        'Python Version': '3.9.7',
                        'OS': 'macOS',
                        'is_apple_silicon': True,
                        't_math_version': f'0.{4-i}.0'  # Simulate version changes
                    }
                }
            }
            
            # Write to a file in the history directory
            filename = f"benchmark_history_{timestamp.replace(' ', '_').replace(':', '-')}.json"
            with open(os.path.join(self.history_dir, filename), 'w') as f:
                json.dump(historical_data, f)
                
            logger.info(f"Generated historical dataset {i+1} at {timestamp}")
    
    def test_generate_report(self):
        """Test the generation of an HTML report."""
        try:
            # Generate the report
            self.reporter.generate_report(self.results, self.output_file)
            
            # Check if the file was created
            self.assertTrue(os.path.exists(self.output_file), "Output file was not created")
            
            # Check file size (should be non-trivial)
            file_size = os.path.getsize(self.output_file)
            self.assertGreater(file_size, 10000, "HTML report is too small, might be incomplete")
            
            # Check if the file contains expected content
            with open(self.output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check for critical HTML structure elements
                self.assertIn("<!DOCTYPE html>", content, "DOCTYPE declaration missing")
                self.assertIn("<html", content, "HTML tag missing")
                self.assertIn("<head>", content, "HEAD tag missing")
                self.assertIn("<body", content, "BODY tag missing")
                
                # Check for key sections
                self.assertIn("Matrix-Benchmark-Ergebnisse", content, "Title missing")
                self.assertIn("Schnellübersicht", content, "Overview section missing")
                self.assertIn("Detaillierte Ergebnisse", content, "Details section missing")
                self.assertIn("Leistungsdiagramme", content, "Charts section missing")
                self.assertIn("Empfehlungen", content, "Recommendations section missing")
                
                # Check for dynamic content
                self.assertIn("PyTorch", content, "Backend name missing")
                self.assertIn("MLX", content, "Backend name missing")
                self.assertIn("NumPy", content, "Backend name missing")
                self.assertIn("matmul", content, "Operation name missing")
                
                # Check for JavaScript integration
                self.assertIn("Plotly", content, "Plotly.js integration missing")
                self.assertIn("function", content, "JavaScript functions missing")
                self.assertIn("createBackendPerformanceChart", content, "Chart function missing")
                
                # Check for filtering functionality
                self.assertIn("filter-select", content, "Filter elements missing")
                self.assertIn("applyFilters", content, "Filter function missing")
                
                # Check for historical data
                if self.reporter.include_historical:
                    self.assertIn("Historischer Vergleich", content, "Historical section missing")
                    self.assertIn("historicalPerformanceChart", content, "Historical chart missing")
                
                logger.info(f"Generated HTML report verified, size: {file_size} bytes")
        except Exception as e:
            logger.error(f"Error during test_generate_report: {str(e)}")
            raise
    
    def test_component_functions(self):
        """Test individual component functions of the HTMLReporter."""
        try:
            # Test chart data preparation
            chart_data = self.reporter._prepare_chart_data(self.results)
            self.assertIsNotNone(chart_data, "Chart data preparation failed")
            self.assertIn("operations", chart_data, "Operations missing in chart data")
            self.assertIn("backends", chart_data, "Backends missing in chart data")
            
            # Test fastest backend detection
            fastest_backend, speedup = self.reporter._get_fastest_backend(self.results)
            self.assertIsNotNone(fastest_backend, "Failed to identify fastest backend")
            self.assertGreater(speedup, 1.0, "Speedup should be greater than 1.0")
            self.assertEqual(fastest_backend, "MLX", "MLX should be the fastest backend in test data")
            
            # Test recommendation generation
            recommendations = self.reporter._generate_recommendations(self.results)
            self.assertIsNotNone(recommendations, "Failed to generate recommendations")
            self.assertGreater(len(recommendations), 0, "No recommendations generated")
            
            # Test historical data preparation if enabled
            if self.reporter.include_historical:
                self.reporter.load_historical_data(self.history_dir)
                self.assertGreater(len(self.reporter.historical_datasets), 0, 
                                "Failed to load historical datasets")
                
                historical_comparison = self.reporter._prepare_historical_comparison(
                    self.results, self.reporter.historical_datasets)
                self.assertIsNotNone(historical_comparison, "Historical comparison data is None")
                self.assertIn("timestamps", historical_comparison, "Timestamps missing in historical data")
                self.assertIn("datasets", historical_comparison, "Datasets missing in historical data")
            
            logger.info("Component function tests passed")
        except Exception as e:
            logger.error(f"Error during test_component_functions: {str(e)}")
            raise
    
    def test_system_info(self):
        """Test system information gathering."""
        try:
            system_info = self.reporter._get_system_info()
            self.assertIsNotNone(system_info, "System info is None")
            self.assertIn("Python Version", system_info, "Python version missing in system info")
            self.assertIn("Betriebssystem", system_info, "OS info missing in system info")
            
            logger.info("System info test passed")
        except Exception as e:
            logger.error(f"Error during test_system_info: {str(e)}")
            raise
    
    def test_different_themes(self):
        """Test the theme functionality."""
        try:
            # Test dark theme
            dark_reporter = HTMLReporter(theme="dark")
            dark_output = os.path.join(self.output_dir, "dark_theme_report.html")
            dark_reporter.generate_report(self.results, dark_output)
            
            with open(dark_output, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn('class="dark"', content, "Dark theme class missing")
            
            logger.info("Theme test passed")
        except Exception as e:
            logger.error(f"Error during test_different_themes: {str(e)}")
            raise
        finally:
            if os.path.exists(dark_output):
                os.remove(dark_output)


if __name__ == "__main__":
    unittest.main()
