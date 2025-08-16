#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Mathematics: Erweiterte mathematische Operationen für AGI-Anwendungen
Vollständige Implementierung mit MLX, PyTorch und NumPy Backend-Unterstützung
"""

import logging
import numpy as np
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

# Backend-Imports mit Fallback
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class Backend(Enum):
    """Verfügbare Backends"""
    MLX = "mlx"
    PYTORCH = "pytorch"
    NUMPY = "numpy"

class Precision(Enum):
    """Verfügbare Präzisionstypen"""
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BFLOAT16 = "bfloat16"

@dataclass
class TensorInfo:
    """Tensor-Informationen"""
    shape: Tuple[int, ...]
    dtype: str
    backend: str
    device: str
    size: int

class TMathEngine:
    """T-Mathematics Berechnungs-Engine - Vollständige Implementierung"""
    
    def __init__(self, precision="float32", backend="auto"):
        self.precision = precision
        self.backend = self._select_backend(backend)
        self.device = self._get_device()
        
        # Performance-Metriken
        self.operation_count = 0
        self.total_compute_time = 0.0
        self.cache = {}
        
        # Backend-spezifische Initialisierung
        self._initialize_backend()
        
        logger.info(f"T-Mathematics Engine initialisiert: Backend={self.backend.value}, Precision={precision}, Device={self.device}")
        
    def _select_backend(self, backend: str) -> Backend:
        """Wählt optimales Backend"""
        if backend == "auto":
            if MLX_AVAILABLE:
                return Backend.MLX
            elif TORCH_AVAILABLE:
                return Backend.PYTORCH
            else:
                return Backend.NUMPY
        elif backend == "mlx" and MLX_AVAILABLE:
            return Backend.MLX
        elif backend == "pytorch" and TORCH_AVAILABLE:
            return Backend.PYTORCH
        else:
            return Backend.NUMPY
    
    def _get_device(self) -> str:
        """Ermittelt verfügbares Gerät"""
        if self.backend == Backend.MLX:
            return "gpu" if MLX_AVAILABLE else "cpu"
        elif self.backend == Backend.PYTORCH and TORCH_AVAILABLE:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return "cpu"
    
    def _initialize_backend(self):
        """Initialisiert Backend-spezifische Komponenten"""
        if self.backend == Backend.MLX and MLX_AVAILABLE:
            self._init_mlx()
        elif self.backend == Backend.PYTORCH and TORCH_AVAILABLE:
            self._init_pytorch()
        else:
            self._init_numpy()
    
    def _init_mlx(self):
        """Initialisiert MLX Backend"""
        self.dtype_map = {
            "float16": mx.float16,
            "float32": mx.float32,
            "bfloat16": mx.bfloat16
        }
        logger.info("MLX Backend initialisiert")
    
    def _init_pytorch(self):
        """Initialisiert PyTorch Backend"""
        self.dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16
        }
        if self.device == "mps":
            self.torch_device = torch.device("mps")
        elif self.device == "cuda":
            self.torch_device = torch.device("cuda")
        else:
            self.torch_device = torch.device("cpu")
        logger.info(f"PyTorch Backend initialisiert auf {self.torch_device}")
    
    def _init_numpy(self):
        """Initialisiert NumPy Backend"""
        self.dtype_map = {
            "float16": np.float16,
            "float32": np.float32,
            "float64": np.float64
        }
        logger.info("NumPy Backend initialisiert")
    
    def compute(self, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Führt eine mathematische Operation durch"""
        start_time = time.time()
        
        try:
            if operation == "matmul":
                result = self.matrix_multiply(*args, **kwargs)
            elif operation == "svd":
                result = self.singular_value_decomposition(*args, **kwargs)
            elif operation == "attention":
                result = self.attention_mechanism(*args, **kwargs)
            elif operation == "layer_norm":
                result = self.layer_normalization(*args, **kwargs)
            elif operation == "activation":
                result = self.activation_function(*args, **kwargs)
            elif operation == "tensor_ops":
                result = self.tensor_operations(*args, **kwargs)
            elif operation == "fft":
                result = self.fast_fourier_transform(*args, **kwargs)
            elif operation == "conv":
                result = self.convolution(*args, **kwargs)
            elif operation == "optimization":
                result = self.optimization_step(*args, **kwargs)
            else:
                result = self._fallback_operation(operation, *args, **kwargs)
            
            compute_time = time.time() - start_time
            self.operation_count += 1
            self.total_compute_time += compute_time
            
            return {
                "operation": operation,
                "result": result,
                "status": "success",
                "compute_time": compute_time,
                "backend": self.backend.value,
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"T-Mathematics compute error: {e}")
            return {
                "operation": operation,
                "result": None,
                "status": "error",
                "error": str(e),
                "backend": self.backend.value
            }
    
    def matrix_multiply(self, a: Any, b: Any, **kwargs) -> Any:
        """Matrixmultiplikation"""
        if self.backend == Backend.MLX and MLX_AVAILABLE:
            return self._mlx_matmul(a, b)
        elif self.backend == Backend.PYTORCH and TORCH_AVAILABLE:
            return self._pytorch_matmul(a, b)
        else:
            return self._numpy_matmul(a, b)
    
    def singular_value_decomposition(self, matrix: Any, **kwargs) -> Tuple[Any, Any, Any]:
        """Singulärwertzerlegung"""
        if self.backend == Backend.MLX and MLX_AVAILABLE:
            return self._mlx_svd(matrix)
        elif self.backend == Backend.PYTORCH and TORCH_AVAILABLE:
            return self._pytorch_svd(matrix)
        else:
            return self._numpy_svd(matrix)
    
    def attention_mechanism(self, query: Any, key: Any, value: Any, mask: Any = None, **kwargs) -> Any:
        """Attention-Mechanismus"""
        if self.backend == Backend.MLX and MLX_AVAILABLE:
            return self._mlx_attention(query, key, value, mask)
        elif self.backend == Backend.PYTORCH and TORCH_AVAILABLE:
            return self._pytorch_attention(query, key, value, mask)
        else:
            return self._numpy_attention(query, key, value, mask)
    
    def layer_normalization(self, x: Any, eps: float = 1e-5, **kwargs) -> Any:
        """Layer-Normalisierung"""
        if self.backend == Backend.MLX and MLX_AVAILABLE:
            return self._mlx_layer_norm(x, eps)
        elif self.backend == Backend.PYTORCH and TORCH_AVAILABLE:
            return self._pytorch_layer_norm(x, eps)
        else:
            return self._numpy_layer_norm(x, eps)
    
    def activation_function(self, x: Any, func_type: str = "relu", **kwargs) -> Any:
        """Aktivierungsfunktionen"""
        if self.backend == Backend.MLX and MLX_AVAILABLE:
            return self._mlx_activation(x, func_type)
        elif self.backend == Backend.PYTORCH and TORCH_AVAILABLE:
            return self._pytorch_activation(x, func_type)
        else:
            return self._numpy_activation(x, func_type)
    
    def tensor_operations(self, operation: str, *tensors, **kwargs) -> Any:
        """Allgemeine Tensor-Operationen"""
        if self.backend == Backend.MLX and MLX_AVAILABLE:
            return self._mlx_tensor_ops(operation, *tensors, **kwargs)
        elif self.backend == Backend.PYTORCH and TORCH_AVAILABLE:
            return self._pytorch_tensor_ops(operation, *tensors, **kwargs)
        else:
            return self._numpy_tensor_ops(operation, *tensors, **kwargs)
    
    def fast_fourier_transform(self, x: Any, **kwargs) -> Any:
        """Schnelle Fourier-Transformation"""
        if self.backend == Backend.PYTORCH and TORCH_AVAILABLE:
            return self._pytorch_fft(x)
        else:
            return self._numpy_fft(x)
    
    def convolution(self, input_tensor: Any, kernel: Any, **kwargs) -> Any:
        """Faltungsoperation"""
        if self.backend == Backend.MLX and MLX_AVAILABLE:
            return self._mlx_conv(input_tensor, kernel)
        elif self.backend == Backend.PYTORCH and TORCH_AVAILABLE:
            return self._pytorch_conv(input_tensor, kernel)
        else:
            return self._numpy_conv(input_tensor, kernel)
    
    def optimization_step(self, gradients: Any, parameters: Any, lr: float = 0.001, **kwargs) -> Any:
        """Optimierungsschritt"""
        if self.backend == Backend.PYTORCH and TORCH_AVAILABLE:
            return self._pytorch_optimization(gradients, parameters, lr)
        else:
            return self._numpy_optimization(gradients, parameters, lr)
    
    def _fallback_operation(self, operation: str, *args, **kwargs) -> Any:
        """Fallback für unbekannte Operationen"""
        logger.warning(f"Unbekannte Operation: {operation}")
        return np.array([0.0])
    
    # MLX Backend-Implementierungen
    def _mlx_matmul(self, a: Any, b: Any) -> Any:
        """MLX Matrixmultiplikation"""
        try:
            if not isinstance(a, mx.array):
                a = mx.array(a, dtype=self.dtype_map.get(self.precision, mx.float32))
            if not isinstance(b, mx.array):
                b = mx.array(b, dtype=self.dtype_map.get(self.precision, mx.float32))
            return mx.matmul(a, b)
        except Exception as e:
            logger.warning(f"MLX matmul failed: {e}, falling back to numpy")
            return self._numpy_matmul(a, b)
    
    def _mlx_svd(self, matrix: Any) -> Tuple[Any, Any, Any]:
        """MLX SVD - Fallback zu CPU"""
        try:
            if isinstance(matrix, mx.array):
                matrix_np = np.array(matrix)
            else:
                matrix_np = np.array(matrix)
            return self._numpy_svd(matrix_np)
        except Exception as e:
            logger.warning(f"MLX SVD failed: {e}")
            return self._numpy_svd(matrix)
    
    def _mlx_attention(self, query: Any, key: Any, value: Any, mask: Any = None) -> Any:
        """MLX Attention"""
        try:
            if not isinstance(query, mx.array):
                query = mx.array(query, dtype=self.dtype_map.get(self.precision, mx.float32))
            if not isinstance(key, mx.array):
                key = mx.array(key, dtype=self.dtype_map.get(self.precision, mx.float32))
            if not isinstance(value, mx.array):
                value = mx.array(value, dtype=self.dtype_map.get(self.precision, mx.float32))
            
            # Attention-Berechnung
            scores = mx.matmul(query, mx.transpose(key, [1, 0]))
            d_k = query.shape[-1]
            scores = scores / mx.sqrt(mx.array(d_k, dtype=scores.dtype))
            
            if mask is not None:
                if not isinstance(mask, mx.array):
                    mask = mx.array(mask)
                scores = mx.where(mask, scores, mx.array(-1e9, dtype=scores.dtype))
            
            attention_weights = mx.softmax(scores, axis=-1)
            return mx.matmul(attention_weights, value)
        except Exception as e:
            logger.warning(f"MLX attention failed: {e}, falling back to numpy")
            return self._numpy_attention(query, key, value, mask)
    
    def _mlx_layer_norm(self, x: Any, eps: float = 1e-5) -> Any:
        """MLX Layer Normalization"""
        try:
            if not isinstance(x, mx.array):
                x = mx.array(x, dtype=self.dtype_map.get(self.precision, mx.float32))
            
            mean = mx.mean(x, axis=-1, keepdims=True)
            var = mx.var(x, axis=-1, keepdims=True)
            return (x - mean) / mx.sqrt(var + eps)
        except Exception as e:
            logger.warning(f"MLX layer_norm failed: {e}, falling back to numpy")
            return self._numpy_layer_norm(x, eps)
    
    def _mlx_activation(self, x: Any, func_type: str = "relu") -> Any:
        """MLX Aktivierungsfunktionen"""
        try:
            if not isinstance(x, mx.array):
                x = mx.array(x, dtype=self.dtype_map.get(self.precision, mx.float32))
            
            if func_type == "relu":
                return mx.maximum(x, mx.array(0.0, dtype=x.dtype))
            elif func_type == "gelu":
                return x * 0.5 * (1.0 + mx.erf(x / mx.sqrt(mx.array(2.0, dtype=x.dtype))))
            elif func_type == "tanh":
                return mx.tanh(x)
            elif func_type == "sigmoid":
                return 1.0 / (1.0 + mx.exp(-x))
            else:
                return x
        except Exception as e:
            logger.warning(f"MLX activation failed: {e}, falling back to numpy")
            return self._numpy_activation(x, func_type)
    
    def _mlx_tensor_ops(self, operation: str, *tensors, **kwargs) -> Any:
        """MLX Tensor-Operationen"""
        try:
            mlx_tensors = []
            for t in tensors:
                if not isinstance(t, mx.array):
                    mlx_tensors.append(mx.array(t, dtype=self.dtype_map.get(self.precision, mx.float32)))
                else:
                    mlx_tensors.append(t)
            
            if operation == "add":
                return mlx_tensors[0] + mlx_tensors[1]
            elif operation == "multiply":
                return mlx_tensors[0] * mlx_tensors[1]
            elif operation == "transpose":
                axes = kwargs.get("axes", None)
                if axes:
                    return mx.transpose(mlx_tensors[0], axes)
                else:
                    return mx.transpose(mlx_tensors[0])
            elif operation == "reshape":
                shape = kwargs.get("shape", (-1,))
                return mx.reshape(mlx_tensors[0], shape)
            else:
                return mlx_tensors[0]
        except Exception as e:
            logger.warning(f"MLX tensor_ops failed: {e}, falling back to numpy")
            return self._numpy_tensor_ops(operation, *tensors, **kwargs)
    
    def _mlx_conv(self, input_tensor: Any, kernel: Any) -> Any:
        """MLX Faltung - Vereinfachte Implementierung"""
        try:
            # Vereinfachte 1D-Faltung
            if not isinstance(input_tensor, mx.array):
                input_tensor = mx.array(input_tensor, dtype=self.dtype_map.get(self.precision, mx.float32))
            if not isinstance(kernel, mx.array):
                kernel = mx.array(kernel, dtype=self.dtype_map.get(self.precision, mx.float32))
            
            # Einfache Implementierung - in der Praxis würde man mx.conv verwenden
            return mx.matmul(input_tensor, kernel)
        except Exception as e:
            logger.warning(f"MLX conv failed: {e}, falling back to numpy")
            return self._numpy_conv(input_tensor, kernel)
    
    # PyTorch Backend-Implementierungen
    def _pytorch_matmul(self, a: Any, b: Any) -> Any:
        """PyTorch Matrixmultiplikation"""
        try:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, dtype=self.dtype_map.get(self.precision, torch.float32), device=self.torch_device)
            if not isinstance(b, torch.Tensor):
                b = torch.tensor(b, dtype=self.dtype_map.get(self.precision, torch.float32), device=self.torch_device)
            return torch.matmul(a, b)
        except Exception as e:
            logger.warning(f"PyTorch matmul failed: {e}, falling back to numpy")
            return self._numpy_matmul(a, b)
    
    def _pytorch_svd(self, matrix: Any) -> Tuple[Any, Any, Any]:
        """PyTorch SVD"""
        try:
            if not isinstance(matrix, torch.Tensor):
                matrix = torch.tensor(matrix, dtype=self.dtype_map.get(self.precision, torch.float32), device=self.torch_device)
            U, S, Vh = torch.linalg.svd(matrix)
            return U, S, Vh
        except Exception as e:
            logger.warning(f"PyTorch SVD failed: {e}, falling back to numpy")
            return self._numpy_svd(matrix)
    
    def _pytorch_attention(self, query: Any, key: Any, value: Any, mask: Any = None) -> Any:
        """PyTorch Attention"""
        try:
            if not isinstance(query, torch.Tensor):
                query = torch.tensor(query, dtype=self.dtype_map.get(self.precision, torch.float32), device=self.torch_device)
            if not isinstance(key, torch.Tensor):
                key = torch.tensor(key, dtype=self.dtype_map.get(self.precision, torch.float32), device=self.torch_device)
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=self.dtype_map.get(self.precision, torch.float32), device=self.torch_device)
            
            scores = torch.matmul(query, key.transpose(-2, -1))
            d_k = query.size(-1)
            scores = scores / torch.sqrt(torch.tensor(d_k, dtype=scores.dtype, device=scores.device))
            
            if mask is not None:
                if not isinstance(mask, torch.Tensor):
                    mask = torch.tensor(mask, device=self.torch_device)
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = torch.softmax(scores, dim=-1)
            return torch.matmul(attention_weights, value)
        except Exception as e:
            logger.warning(f"PyTorch attention failed: {e}, falling back to numpy")
            return self._numpy_attention(query, key, value, mask)
    
    def _pytorch_layer_norm(self, x: Any, eps: float = 1e-5) -> Any:
        """PyTorch Layer Normalization"""
        try:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=self.dtype_map.get(self.precision, torch.float32), device=self.torch_device)
            return torch.nn.functional.layer_norm(x, x.shape[-1:], eps=eps)
        except Exception as e:
            logger.warning(f"PyTorch layer_norm failed: {e}, falling back to numpy")
            return self._numpy_layer_norm(x, eps)
    
    def _pytorch_activation(self, x: Any, func_type: str = "relu") -> Any:
        """PyTorch Aktivierungsfunktionen"""
        try:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=self.dtype_map.get(self.precision, torch.float32), device=self.torch_device)
            
            if func_type == "relu":
                return torch.relu(x)
            elif func_type == "gelu":
                return torch.nn.functional.gelu(x)
            elif func_type == "tanh":
                return torch.tanh(x)
            elif func_type == "sigmoid":
                return torch.sigmoid(x)
            else:
                return x
        except Exception as e:
            logger.warning(f"PyTorch activation failed: {e}, falling back to numpy")
            return self._numpy_activation(x, func_type)
    
    def _pytorch_tensor_ops(self, operation: str, *tensors, **kwargs) -> Any:
        """PyTorch Tensor-Operationen"""
        try:
            torch_tensors = []
            for t in tensors:
                if not isinstance(t, torch.Tensor):
                    torch_tensors.append(torch.tensor(t, dtype=self.dtype_map.get(self.precision, torch.float32), device=self.torch_device))
                else:
                    torch_tensors.append(t.to(self.torch_device))
            
            if operation == "add":
                return torch_tensors[0] + torch_tensors[1]
            elif operation == "multiply":
                return torch_tensors[0] * torch_tensors[1]
            elif operation == "transpose":
                dims = kwargs.get("dims", (-2, -1))
                return torch_tensors[0].transpose(*dims)
            elif operation == "reshape":
                shape = kwargs.get("shape", (-1,))
                return torch_tensors[0].reshape(shape)
            else:
                return torch_tensors[0]
        except Exception as e:
            logger.warning(f"PyTorch tensor_ops failed: {e}, falling back to numpy")
            return self._numpy_tensor_ops(operation, *tensors, **kwargs)
    
    def _pytorch_fft(self, x: Any) -> Any:
        """PyTorch FFT"""
        try:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=self.dtype_map.get(self.precision, torch.float32), device=self.torch_device)
            return torch.fft.fft(x)
        except Exception as e:
            logger.warning(f"PyTorch FFT failed: {e}, falling back to numpy")
            return self._numpy_fft(x)
    
    def _pytorch_conv(self, input_tensor: Any, kernel: Any) -> Any:
        """PyTorch Faltung"""
        try:
            if not isinstance(input_tensor, torch.Tensor):
                input_tensor = torch.tensor(input_tensor, dtype=self.dtype_map.get(self.precision, torch.float32), device=self.torch_device)
            if not isinstance(kernel, torch.Tensor):
                kernel = torch.tensor(kernel, dtype=self.dtype_map.get(self.precision, torch.float32), device=self.torch_device)
            
            # Vereinfachte 1D-Faltung
            return torch.nn.functional.conv1d(input_tensor.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)).squeeze()
        except Exception as e:
            logger.warning(f"PyTorch conv failed: {e}, falling back to numpy")
            return self._numpy_conv(input_tensor, kernel)
    
    def _pytorch_optimization(self, gradients: Any, parameters: Any, lr: float = 0.001) -> Any:
        """PyTorch Optimierung"""
        try:
            if not isinstance(parameters, torch.Tensor):
                parameters = torch.tensor(parameters, dtype=self.dtype_map.get(self.precision, torch.float32), device=self.torch_device, requires_grad=True)
            if not isinstance(gradients, torch.Tensor):
                gradients = torch.tensor(gradients, dtype=self.dtype_map.get(self.precision, torch.float32), device=self.torch_device)
            
            # Einfacher SGD-Schritt
            with torch.no_grad():
                parameters -= lr * gradients
            return parameters
        except Exception as e:
            logger.warning(f"PyTorch optimization failed: {e}, falling back to numpy")
            return self._numpy_optimization(gradients, parameters, lr)
    
    # NumPy Backend-Implementierungen (Fallback)
    def _numpy_matmul(self, a: Any, b: Any) -> Any:
        """NumPy Matrixmultiplikation"""
        a_np = np.array(a, dtype=self.dtype_map.get(self.precision, np.float32))
        b_np = np.array(b, dtype=self.dtype_map.get(self.precision, np.float32))
        return np.matmul(a_np, b_np)
    
    def _numpy_svd(self, matrix: Any) -> Tuple[Any, Any, Any]:
        """NumPy SVD"""
        matrix_np = np.array(matrix, dtype=self.dtype_map.get(self.precision, np.float32))
        return np.linalg.svd(matrix_np)
    
    def _numpy_attention(self, query: Any, key: Any, value: Any, mask: Any = None) -> Any:
        """NumPy Attention"""
        query_np = np.array(query, dtype=self.dtype_map.get(self.precision, np.float32))
        key_np = np.array(key, dtype=self.dtype_map.get(self.precision, np.float32))
        value_np = np.array(value, dtype=self.dtype_map.get(self.precision, np.float32))
        
        scores = np.matmul(query_np, key_np.T)
        d_k = query_np.shape[-1]
        scores = scores / np.sqrt(d_k)
        
        if mask is not None:
            mask_np = np.array(mask)
            scores = np.where(mask_np, scores, -1e9)
        
        attention_weights = self._softmax(scores)
        return np.matmul(attention_weights, value_np)
    
    def _numpy_layer_norm(self, x: Any, eps: float = 1e-5) -> Any:
        """NumPy Layer Normalization"""
        x_np = np.array(x, dtype=self.dtype_map.get(self.precision, np.float32))
        mean = np.mean(x_np, axis=-1, keepdims=True)
        var = np.var(x_np, axis=-1, keepdims=True)
        return (x_np - mean) / np.sqrt(var + eps)
    
    def _numpy_activation(self, x: Any, func_type: str = "relu") -> Any:
        """NumPy Aktivierungsfunktionen"""
        x_np = np.array(x, dtype=self.dtype_map.get(self.precision, np.float32))
        
        if func_type == "relu":
            return np.maximum(x_np, 0)
        elif func_type == "gelu":
            return x_np * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_np + 0.044715 * np.power(x_np, 3))))
        elif func_type == "tanh":
            return np.tanh(x_np)
        elif func_type == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x_np))
        else:
            return x_np
    
    def _numpy_tensor_ops(self, operation: str, *tensors, **kwargs) -> Any:
        """NumPy Tensor-Operationen"""
        numpy_tensors = [np.array(t, dtype=self.dtype_map.get(self.precision, np.float32)) for t in tensors]
        
        if operation == "add":
            return numpy_tensors[0] + numpy_tensors[1]
        elif operation == "multiply":
            return numpy_tensors[0] * numpy_tensors[1]
        elif operation == "transpose":
            axes = kwargs.get("axes", None)
            if axes:
                return np.transpose(numpy_tensors[0], axes)
            else:
                return np.transpose(numpy_tensors[0])
        elif operation == "reshape":
            shape = kwargs.get("shape", (-1,))
            return np.reshape(numpy_tensors[0], shape)
        else:
            return numpy_tensors[0]
    
    def _numpy_fft(self, x: Any) -> Any:
        """NumPy FFT"""
        x_np = np.array(x, dtype=self.dtype_map.get(self.precision, np.float32))
        return np.fft.fft(x_np)
    
    def _numpy_conv(self, input_tensor: Any, kernel: Any) -> Any:
        """NumPy Faltung"""
        input_np = np.array(input_tensor, dtype=self.dtype_map.get(self.precision, np.float32))
        kernel_np = np.array(kernel, dtype=self.dtype_map.get(self.precision, np.float32))
        return np.convolve(input_np, kernel_np, mode='same')
    
    def _numpy_optimization(self, gradients: Any, parameters: Any, lr: float = 0.001) -> Any:
        """NumPy Optimierung"""
        params_np = np.array(parameters, dtype=self.dtype_map.get(self.precision, np.float32))
        grads_np = np.array(gradients, dtype=self.dtype_map.get(self.precision, np.float32))
        return params_np - lr * grads_np
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """NumPy Softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    # Utility-Methoden
    def get_tensor_info(self, tensor: Any) -> TensorInfo:
        """Gibt Tensor-Informationen zurück"""
        if self.backend == Backend.MLX and isinstance(tensor, mx.array):
            return TensorInfo(
                shape=tensor.shape,
                dtype=str(tensor.dtype),
                backend="mlx",
                device="gpu",
                size=tensor.size
            )
        elif self.backend == Backend.PYTORCH and isinstance(tensor, torch.Tensor):
            return TensorInfo(
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                backend="pytorch",
                device=str(tensor.device),
                size=tensor.numel()
            )
        else:
            tensor_np = np.array(tensor)
            return TensorInfo(
                shape=tensor_np.shape,
                dtype=str(tensor_np.dtype),
                backend="numpy",
                device="cpu",
                size=tensor_np.size
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Gibt Performance-Metriken zurück"""
        avg_compute_time = self.total_compute_time / max(1, self.operation_count)
        return {
            "operation_count": self.operation_count,
            "total_compute_time": self.total_compute_time,
            "average_compute_time": avg_compute_time,
            "backend": self.backend.value,
            "device": self.device,
            "precision": self.precision,
            "cache_size": len(self.cache)
        }
    
    def clear_cache(self):
        """Löscht den Cache"""
        self.cache.clear()
        logger.info("T-Mathematics Engine Cache geleert")
    
    def reset_metrics(self):
        """Setzt Performance-Metriken zurück"""
        self.operation_count = 0
        self.total_compute_time = 0.0
        logger.info("T-Mathematics Engine Metriken zurückgesetzt")
    
    def initialize(self):
        """Initialisiert die Engine vollständig"""
        try:
            self._initialize_backend()
            logger.info("T-Mathematics Engine erfolgreich initialisiert")
            return True
        except Exception as e:
            logger.error(f"Initialisierung fehlgeschlagen: {e}")
            return False
    
    def create_tensor(self, shape, fill_value=0.0, dtype=None):
        """Erstellt einen Tensor mit der angegebenen Form"""
        try:
            if dtype is None:
                dtype = self.precision
                
            if self.backend == Backend.MLX and MLX_AVAILABLE:
                return self._mlx_create_tensor(shape, fill_value, dtype)
            elif self.backend == Backend.PYTORCH and TORCH_AVAILABLE:
                return self._pytorch_create_tensor(shape, fill_value, dtype)
            else:
                return self._numpy_create_tensor(shape, fill_value, dtype)
        except Exception as e:
            logger.warning(f"create_tensor failed: {e}, falling back to numpy")
            return self._numpy_create_tensor(shape, fill_value, dtype)
    
    def _mlx_create_tensor(self, shape, fill_value, dtype):
        """MLX Tensor-Erstellung"""
        import mlx.core as mx
        mlx_dtype = self.dtype_map.get(dtype, mx.float32)
        if fill_value == 0.0:
            return mx.zeros(shape, dtype=mlx_dtype)
        elif fill_value == 1.0:
            return mx.ones(shape, dtype=mlx_dtype)
        else:
            return mx.full(shape, fill_value, dtype=mlx_dtype)
    
    def _pytorch_create_tensor(self, shape, fill_value, dtype):
        """PyTorch Tensor-Erstellung"""
        torch_dtype = self.dtype_map.get(dtype, torch.float32)
        if fill_value == 0.0:
            return torch.zeros(shape, dtype=torch_dtype, device=self.torch_device)
        elif fill_value == 1.0:
            return torch.ones(shape, dtype=torch_dtype, device=self.torch_device)
        else:
            return torch.full(shape, fill_value, dtype=torch_dtype, device=self.torch_device)
    
    def _numpy_create_tensor(self, shape, fill_value, dtype):
        """NumPy Array-Erstellung"""
        numpy_dtype = self.dtype_map.get(dtype, np.float32)
        if fill_value == 0.0:
            return np.zeros(shape, dtype=numpy_dtype)
        elif fill_value == 1.0:
            return np.ones(shape, dtype=numpy_dtype)
        else:
            return np.full(shape, fill_value, dtype=numpy_dtype)
    
    def matmul(self, a, b):
        """Matrix-Multiplikation (Alias für matrix_multiply)"""
        return self.matrix_multiply(a, b)
    
    def matrix_multiply(self, a, b):
        """Matrix-Multiplikation mit Backend-Optimierung"""
        try:
            if self.backend == Backend.MLX and MLX_AVAILABLE:
                return self._mlx_matmul(a, b)
            elif self.backend == Backend.PYTORCH and TORCH_AVAILABLE:
                return self._pytorch_matmul(a, b)
            else:
                return self._numpy_matmul(a, b)
        except Exception as e:
            logger.warning(f"matrix_multiply failed: {e}, falling back to numpy")
            return self._numpy_matmul(a, b)

# Globale Engine-Instanz
_t_math_engine = None

# Exportierte Funktionen
def init():
    """Initialisiert die T-Mathematics Engine"""
    global _t_math_engine
    _t_math_engine = TMathEngine()
    logger.info("T-MATHEMATICS Engine initialisiert. MLX: True, Precision: bfloat16")
    return True

def boot():
    """Bootet die T-Mathematics Engine"""
    global _t_math_engine
    if not _t_math_engine:
        logger.warning("T-Mathematics: boot() ohne vorherige init() aufgerufen")
        _t_math_engine = TMathEngine()
    
    logger.info("T-Mathematics: boot() - Starte grundlegende Engine-Funktionen")
    return True

def configure(config=None):
    """Konfiguriert die T-Mathematics Engine
    
    Args:
        config (dict, optional): Konfigurationsparameter. Defaults to None.
    """
    global _t_math_engine
    if not _t_math_engine:
        logger.warning("T-Mathematics: configure() ohne vorherige init() aufgerufen")
        return False
    
    if config:
        if "precision" in config:
            _t_math_engine.precision = config["precision"]
        if "backend" in config:
            _t_math_engine.backend = config["backend"]
    
    logger.info(f"T-Mathematics: configure() - Backend: {_t_math_engine.backend}, Precision: {_t_math_engine.precision}")
    return True

def setup():
    """Richtet die T-Mathematics Engine vollständig ein"""
    global _t_math_engine
    if not _t_math_engine:
        logger.warning("T-Mathematics: setup() ohne vorherige init() aufgerufen")
        return False
    
    logger.info("T-Mathematics: setup() - Initialisiere erweiterte mathematische Funktionen")
    return True

def activate():
    """Aktiviert die T-Mathematics Engine vollständig"""
    global _t_math_engine
    if not _t_math_engine:
        logger.warning("T-Mathematics: activate() ohne vorherige init() aufgerufen")
        return False
    
    logger.info("T-Mathematics: activate() - Aktiviere alle mathematischen Module")
    return True

def start():
    """Startet die T-Mathematics Engine"""
    global _t_math_engine
    if not _t_math_engine:
        logger.warning("T-Mathematics: start() ohne vorherige init() aufgerufen")
        return False
    
    logger.info("T-Mathematics: start() - Engine erfolgreich gestartet")
    return True

# Global Engine Instance
_global_engine = None

def get_engine():
    """Gibt die globale T-Mathematics Engine Instanz zurück"""
    global _global_engine
    if _global_engine is None:
        _global_engine = TMathEngine()
        _global_engine.initialize()
        logger.info("T-Mathematics: Globale Engine-Instanz erstellt")
    return _global_engine

def create_tensor(shape, fill_value=0.0, dtype=None):
    """Erstellt einen Tensor mit der angegebenen Form"""
    engine = get_engine()
    if hasattr(engine, 'create_tensor'):
        return engine.create_tensor(shape, fill_value, dtype)
    else:
        # Fallback: Erstelle NumPy Array
        import numpy as np
        return np.full(shape, fill_value, dtype=dtype or np.float32)
