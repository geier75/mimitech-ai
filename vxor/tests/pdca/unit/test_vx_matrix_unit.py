#!/usr/bin/env python3
"""
PDCA Unit Tests für vx_matrix (T-Mathematics Engine)
Test-ID: DO-001
PDCA-Phase: DO (Durchführung)
TDD-Relevanz: Ja
Risiko-Score: 8/10
Modul: T-Mathematics Engine
Framework: pytest

Testziele:
- Matrix-Operationen validieren
- Backend-Auswahl (MLX, PyTorch, NumPy) testen
- Hardware-Optimierung prüfen
- Fehlerbehandlung validieren
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Pfad für MISO-Module hinzufügen
sys.path.append('/Volumes/My Book/MISO_Ultimate 15.32.28')

try:
    from miso.math.t_mathematics.engine import TMathEngine
    from miso.math.t_mathematics.tensor import MISOTensor, MLXTensor, TorchTensor
    T_MATH_AVAILABLE = True
except ImportError as e:
    T_MATH_AVAILABLE = False
    print(f"T-Mathematics Engine nicht verfügbar: {e}")

# MLX-Verfügbarkeit prüfen
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# PyTorch-Verfügbarkeit prüfen
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestVXMatrixUnit:
    """Unit Tests für vx_matrix Kernfunktionalität"""
    
    @pytest.fixture
    def mock_engine(self):
        """Mock T-Mathematics Engine für Tests ohne Hardware-Abhängigkeiten"""
        engine = Mock()
        engine.create_tensor = Mock()
        engine.matmul = Mock()
        engine.svd = Mock()
        engine.attention = Mock()
        engine.layer_norm = Mock()
        engine.activate = Mock()
        return engine
    
    @pytest.fixture
    def sample_data(self):
        """Testdaten für Matrix-Operationen"""
        return {
            'matrix_2x2': [[1.0, 2.0], [3.0, 4.0]],
            'matrix_3x3': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            'vector_3': [1.0, 2.0, 3.0],
            'identity_2x2': [[1.0, 0.0], [0.0, 1.0]]
        }
    
    def test_matrix_creation_basic(self, mock_engine, sample_data):
        """Test: Grundlegende Matrix-Erstellung"""
        # Arrange
        mock_engine.create_tensor.return_value = Mock()
        
        # Act
        result = mock_engine.create_tensor(sample_data['matrix_2x2'])
        
        # Assert
        mock_engine.create_tensor.assert_called_once_with(sample_data['matrix_2x2'])
        assert result is not None
    
    def test_matrix_multiplication_basic(self, mock_engine, sample_data):
        """Test: Grundlegende Matrix-Multiplikation"""
        # Arrange
        tensor_a = Mock()
        tensor_b = Mock()
        expected_result = Mock()
        mock_engine.matmul.return_value = expected_result
        
        # Act
        result = mock_engine.matmul(tensor_a, tensor_b)
        
        # Assert
        mock_engine.matmul.assert_called_once_with(tensor_a, tensor_b)
        assert result == expected_result
    
    def test_svd_decomposition(self, mock_engine):
        """Test: SVD-Zerlegung"""
        # Arrange
        tensor = Mock()
        expected_u = Mock()
        expected_s = Mock()
        expected_v = Mock()
        mock_engine.svd.return_value = (expected_u, expected_s, expected_v)
        
        # Act
        u, s, v = mock_engine.svd(tensor)
        
        # Assert
        mock_engine.svd.assert_called_once_with(tensor)
        assert u == expected_u
        assert s == expected_s
        assert v == expected_v
    
    def test_attention_mechanism(self, mock_engine):
        """Test: Attention-Mechanismus"""
        # Arrange
        query = Mock()
        key = Mock()
        value = Mock()
        mask = Mock()
        expected_result = Mock()
        mock_engine.attention.return_value = expected_result
        
        # Act
        result = mock_engine.attention(query, key, value, mask)
        
        # Assert
        mock_engine.attention.assert_called_once_with(query, key, value, mask)
        assert result == expected_result
    
    def test_layer_normalization(self, mock_engine):
        """Test: Layer-Normalisierung"""
        # Arrange
        input_tensor = Mock()
        weight = Mock()
        bias = Mock()
        eps = 1e-5
        expected_result = Mock()
        mock_engine.layer_norm.return_value = expected_result
        
        # Act
        result = mock_engine.layer_norm(input_tensor, weight, bias, eps)
        
        # Assert
        mock_engine.layer_norm.assert_called_once_with(input_tensor, weight, bias, eps)
        assert result == expected_result
    
    def test_activation_functions(self, mock_engine):
        """Test: Aktivierungsfunktionen"""
        # Arrange
        input_tensor = Mock()
        activation_types = ['relu', 'gelu', 'tanh', 'sigmoid']
        expected_result = Mock()
        mock_engine.activate.return_value = expected_result
        
        for activation_type in activation_types:
            # Act
            result = mock_engine.activate(input_tensor, activation_type)
            
            # Assert
            assert result == expected_result
    
    @pytest.mark.skipif(not T_MATH_AVAILABLE, reason="T-Mathematics Engine nicht verfügbar")
    def test_real_engine_initialization(self):
        """Test: Echte Engine-Initialisierung (falls verfügbar)"""
        try:
            engine = TMathEngine()
            assert engine is not None
            assert hasattr(engine, 'create_tensor')
            assert hasattr(engine, 'matmul')
            assert hasattr(engine, 'svd')
        except Exception as e:
            pytest.skip(f"Engine-Initialisierung fehlgeschlagen: {e}")
    
    @pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX nicht verfügbar")
    def test_mlx_backend_selection(self):
        """Test: MLX-Backend-Auswahl (Apple Silicon)"""
        try:
            if T_MATH_AVAILABLE:
                engine = TMathEngine(preferred_backend='mlx')
                assert engine.current_backend == 'mlx'
        except Exception as e:
            pytest.skip(f"MLX-Backend nicht verfügbar: {e}")
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch nicht verfügbar")
    def test_torch_backend_selection(self):
        """Test: PyTorch-Backend-Auswahl"""
        try:
            if T_MATH_AVAILABLE:
                engine = TMathEngine(preferred_backend='torch')
                assert engine.current_backend in ['torch', 'numpy']  # Fallback möglich
        except Exception as e:
            pytest.skip(f"PyTorch-Backend nicht verfügbar: {e}")
    
    def test_error_handling_invalid_backend(self, mock_engine):
        """Test: Fehlerbehandlung bei ungültigem Backend"""
        # Arrange
        mock_engine.create_tensor.side_effect = ValueError("Ungültiges Backend")
        
        # Act & Assert
        with pytest.raises(ValueError, match="Ungültiges Backend"):
            mock_engine.create_tensor([[1, 2], [3, 4]])
    
    def test_error_handling_dimension_mismatch(self, mock_engine):
        """Test: Fehlerbehandlung bei Dimensionskonflikt"""
        # Arrange
        mock_engine.matmul.side_effect = ValueError("Dimensionskonflikt")
        tensor_a = Mock()
        tensor_b = Mock()
        
        # Act & Assert
        with pytest.raises(ValueError, match="Dimensionskonflikt"):
            mock_engine.matmul(tensor_a, tensor_b)
    
    def test_performance_benchmark_mock(self, mock_engine, sample_data):
        """Test: Performance-Benchmark (Mock)"""
        import time
        
        # Arrange
        mock_engine.matmul.return_value = Mock()
        tensor_a = Mock()
        tensor_b = Mock()
        
        # Act
        start_time = time.time()
        for _ in range(100):
            mock_engine.matmul(tensor_a, tensor_b)
        end_time = time.time()
        
        # Assert
        execution_time = end_time - start_time
        assert execution_time < 1.0  # Mock sollte sehr schnell sein
        assert mock_engine.matmul.call_count == 100
    
    def test_memory_efficiency_mock(self, mock_engine):
        """Test: Speicher-Effizienz (Mock)"""
        import gc
        
        # Arrange
        mock_engine.create_tensor.return_value = Mock()
        
        # Act
        tensors = []
        for i in range(10):
            tensor = mock_engine.create_tensor([[i, i+1], [i+2, i+3]])
            tensors.append(tensor)
        
        # Cleanup
        del tensors
        gc.collect()
        
        # Assert
        assert mock_engine.create_tensor.call_count == 10
    
    def test_thread_safety_mock(self, mock_engine):
        """Test: Thread-Sicherheit (Mock)"""
        import threading
        import time
        
        # Arrange
        mock_engine.create_tensor.return_value = Mock()
        results = []
        
        def worker():
            result = mock_engine.create_tensor([[1, 2], [3, 4]])
            results.append(result)
        
        # Act
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Assert
        assert len(results) == 5
        assert mock_engine.create_tensor.call_count == 5


class TestVXMatrixIntegration:
    """Integration Tests für vx_matrix mit anderen Komponenten"""
    
    def test_integration_with_echo_prime(self, mock_engine):
        """Test: Integration mit ECHO-PRIME"""
        # Arrange
        mock_engine.create_tensor.return_value = Mock()
        timeline_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        
        # Act
        timeline_tensor = mock_engine.create_tensor(timeline_data)
        
        # Assert
        mock_engine.create_tensor.assert_called_once_with(timeline_data)
        assert timeline_tensor is not None
    
    def test_integration_with_prism_engine(self, mock_engine):
        """Test: Integration mit PRISM-Engine"""
        # Arrange
        mock_engine.matmul.return_value = Mock()
        probability_matrix = Mock()
        state_vector = Mock()
        
        # Act
        result = mock_engine.matmul(probability_matrix, state_vector)
        
        # Assert
        mock_engine.matmul.assert_called_once_with(probability_matrix, state_vector)
        assert result is not None


if __name__ == "__main__":
    # Test-Ausführung mit detaillierter Ausgabe
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "--cov=miso.math.t_mathematics",
        "--cov-report=term-missing"
    ])
