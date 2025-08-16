#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MATRIX: T-Mathematics Adapter

Diese Datei implementiert die Adapter-Schnittstelle zwischen VX-MATRIX und der
MISO T-Mathematics Engine für optimierte Tensor-Operationen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
ZTM-Level: STRICT
"""

import os
import sys
import enum
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path

# Füge MISO-Root zum Pythonpfad hinzu
VXOR_AI_DIR = Path(__file__).parent.parent.parent.absolute()
MISO_ROOT = VXOR_AI_DIR.parent / "miso"
sys.path.insert(0, str(MISO_ROOT))

# ZTM-Protokoll-Initialisierung
ZTM_ACTIVE = os.environ.get('MISO_ZTM_MODE', '1') == '1'
ZTM_LOG_LEVEL = os.environ.get('MISO_ZTM_LOG_LEVEL', 'INFO')

# Logger konfigurieren
logger = logging.getLogger("VXOR.VX-MATRIX.t_math_adapter")
logger.setLevel(getattr(logging, ZTM_LOG_LEVEL))

def ztm_log(message: str, level: str = 'INFO', module: str = 'T_MATH_ADAPTER'):
    """ZTM-konforme Logging-Funktion"""
    if not ZTM_ACTIVE:
        return
    
    log_func = getattr(logger, level.lower())
    log_func(f"[ZTM:{module}] {message}")

# Versuche, die MISO T-Mathematics-Engine zu importieren
try:
    from miso.math.t_mathematics.engine import T_Mathematics_Engine
    from miso.math.t_mathematics.mlx_support import MLXBackend, tensor_to_mlx
    from miso.math.t_mathematics.tensor_wrappers import MLXTensor, PTTensor
    TMATH_AVAILABLE = True
    ztm_log("T-Mathematics Engine erfolgreich importiert", level="INFO")
except ImportError as e:
    TMATH_AVAILABLE = False
    ztm_log(f"T-Mathematics Engine nicht verfügbar: {e}", level="ERROR")

# Versuche, VX-MATRIX-Core-Module zu importieren
try:
    from ..core.matrix_core import TensorType, MatrixCore
    from ..core.tensor_bridge import TensorBridge, ConversionMode
    MATRIX_CORE_AVAILABLE = True
except ImportError:
    MATRIX_CORE_AVAILABLE = False
    ztm_log("VX-MATRIX Core-Module nicht verfügbar", level="WARNING")
    
    # Fallback-Definitionen
    class TensorType(enum.Enum):
        """Definiert die unterstützten Tensor-Typen"""
        NUMPY = "numpy"
        TORCH = "torch"
        MLX = "mlx"
        JAX = "jax"
        UNKNOWN = "unknown"

class TMathAdapter:
    """
    Adapter für die MISO T-Mathematics Engine
    
    Diese Klasse stellt eine Schnittstelle zwischen der VX-MATRIX und der 
    MISO T-Mathematics Engine her, um optimierte Tensor-Operationen zu ermöglichen.
    """
    
    def __init__(self):
        """Initialisiert den T-Mathematics Adapter"""
        self.engine = None
        self.mlx_backend = None
        self.initialized = False
        self.matrix_core = None
        self.tensor_bridge = None
        
        # Operation Counter für Profiling
        self.op_counter = {
            "matrix_ops": 0,
            "tensor_transfers": 0,
            "optimized_ops": 0,
            "fallback_ops": 0
        }
        
        # Initialisieren der Komponenten
        self._initialize_components()
        
        if self.initialized:
            ztm_log("T-Mathematics Adapter erfolgreich initialisiert", level="INFO")
        else:
            ztm_log("T-Mathematics Adapter konnte nicht initialisiert werden", level="WARNING")
    
    def _initialize_components(self):
        """Initialisiert die erforderlichen Komponenten für den Adapter"""
        # Initialisiere T-Mathematics Engine
        if TMATH_AVAILABLE:
            try:
                self.engine = T_Mathematics_Engine()
                
                # Versuche, das MLX-Backend zu erhalten
                if 'mlx' in self.engine.available_backends:
                    self.mlx_backend = self.engine.get_backend('mlx')
                
                # Überprüfe MLX-Backend
                if self.mlx_backend is None:
                    ztm_log("MLX-Backend nicht verfügbar in T-Mathematics Engine", level="WARNING")
                
                # Initialisierung erfolgreich
                self.initialized = True
                ztm_log("T-Mathematics Engine initialisiert", level="INFO")
            except Exception as e:
                ztm_log(f"Fehler bei der Initialisierung der T-Mathematics Engine: {e}", level="ERROR")
        
        # Initialisiere VX-MATRIX Komponenten
        if MATRIX_CORE_AVAILABLE:
            try:
                from ..core.matrix_core import get_matrix_core
                from ..core.tensor_bridge import get_tensor_bridge
                
                self.matrix_core = get_matrix_core()
                self.tensor_bridge = get_tensor_bridge()
                
                ztm_log("VX-MATRIX Core-Komponenten initialisiert", level="INFO")
            except Exception as e:
                ztm_log(f"Fehler bei der Initialisierung der VX-MATRIX Core-Komponenten: {e}", level="ERROR")
    
    def is_mlx_optimized(self) -> bool:
        """
        Überprüft, ob die T-Mathematics Engine mit MLX-Optimierung läuft
        
        Returns:
            True, wenn MLX-optimiert, sonst False
        """
        if not self.initialized or not TMATH_AVAILABLE:
            return False
            
        if self.mlx_backend is None:
            return False
            
        return self.mlx_backend.is_available()
    
    def to_t_math_tensor(self, tensor: Any, backend_type: str = None) -> Any:
        """
        Konvertiert einen Tensor in einen T-Mathematics-Tensor
        
        Args:
            tensor: Eingabe-Tensor (NumPy, PyTorch, MLX, JAX)
            backend_type: 'mlx', 'pytorch' oder None für automatische Auswahl
            
        Returns:
            T-Mathematics-Tensor
        """
        if not self.initialized or not TMATH_AVAILABLE:
            ztm_log("T-Mathematics Engine nicht verfügbar für Konvertierung", level="ERROR")
            return tensor
        
        self.op_counter["tensor_transfers"] += 1
        
        # Ermittle Eingangstyp
        tensor_type = TensorType.detect(tensor) if hasattr(TensorType, 'detect') else None
        
        # Automatische Backend-Auswahl
        if backend_type is None:
            if self.mlx_backend is not None and self.mlx_backend.is_available():
                backend_type = 'mlx'
            elif 'pytorch' in self.engine.available_backends:
                backend_type = 'pytorch'
            else:
                backend_type = next(iter(self.engine.available_backends), None)
        
        if backend_type is None:
            ztm_log("Kein verfügbares Backend für T-Mathematics-Konvertierung", level="ERROR")
            return tensor
        
        try:
            if backend_type == 'mlx':
                # Zu MLX-Tensor konvertieren
                if tensor_type == TensorType.MLX:
                    # Bereits MLX, nichts zu tun
                    return MLXTensor(tensor)
                
                # Konvertiere zu MLX
                if self.tensor_bridge is not None:
                    # Verwende VX-MATRIX TensorBridge
                    mlx_array = self.tensor_bridge.convert(tensor, 'mlx')
                    return MLXTensor(mlx_array)
                elif self.mlx_backend is not None:
                    # Verwende T-Mathematics MLXBackend
                    if tensor_type == TensorType.NUMPY:
                        return self.mlx_backend.from_numpy(tensor)
                    elif tensor_type == TensorType.TORCH:
                        # Verwende tensor_to_mlx, falls verfügbar
                        if 'tensor_to_mlx' in globals():
                            mlx_array = tensor_to_mlx(tensor)
                            return MLXTensor(mlx_array)
                        else:
                            # Fallback über NumPy
                            np_array = tensor.detach().cpu().numpy()
                            return self.mlx_backend.from_numpy(np_array)
                    elif tensor_type == TensorType.JAX:
                        # JAX -> NumPy -> MLX
                        np_array = np.array(tensor)
                        return self.mlx_backend.from_numpy(np_array)
                
                # Letzter Ausweg: über NumPy konvertieren
                if tensor_type == TensorType.NUMPY:
                    return self.mlx_backend.from_numpy(tensor)
                elif tensor_type == TensorType.TORCH:
                    return self.mlx_backend.from_numpy(tensor.detach().cpu().numpy())
                elif tensor_type == TensorType.JAX:
                    return self.mlx_backend.from_numpy(np.array(tensor))
                
            elif backend_type == 'pytorch':
                # Zu PyTorch-Tensor konvertieren
                import torch
                
                if tensor_type == TensorType.TORCH:
                    # Bereits PyTorch, nichts zu tun
                    return PTTensor(tensor)
                
                # Konvertiere zu PyTorch
                if self.tensor_bridge is not None:
                    # Verwende VX-MATRIX TensorBridge
                    torch_tensor = self.tensor_bridge.convert(tensor, 'torch')
                    return PTTensor(torch_tensor)
                else:
                    # Eigene Konvertierung
                    if tensor_type == TensorType.NUMPY:
                        return PTTensor(torch.tensor(tensor))
                    elif tensor_type == TensorType.MLX:
                        # MLX -> NumPy -> PyTorch
                        np_array = np.array(tensor.tolist())
                        return PTTensor(torch.tensor(np_array))
                    elif tensor_type == TensorType.JAX:
                        # JAX -> NumPy -> PyTorch
                        np_array = np.array(tensor)
                        return PTTensor(torch.tensor(np_array))
            
            # Generischer Fallback
            ztm_log(f"Unbekanntes Backend-Typ {backend_type} für T-Mathematics-Konvertierung", level="WARNING")
            return tensor
                
        except Exception as e:
            ztm_log(f"Fehler bei der Konvertierung zu T-Mathematics-Tensor: {e}", level="ERROR")
            return tensor
    
    def from_t_math_tensor(self, t_math_tensor: Any, target_type: str = None) -> Any:
        """
        Konvertiert einen T-Mathematics-Tensor in den angegebenen Zieltyp
        
        Args:
            t_math_tensor: T-Mathematics-Tensor (MLXTensor oder PTTensor)
            target_type: 'numpy', 'torch', 'mlx', 'jax' oder None für bevorzugten Typ
            
        Returns:
            Konvertierter Tensor
        """
        if not self.initialized or not TMATH_AVAILABLE:
            ztm_log("T-Mathematics Engine nicht verfügbar für Konvertierung", level="ERROR")
            return t_math_tensor
        
        self.op_counter["tensor_transfers"] += 1
        
        # Bevorzugten Typ bestimmen, falls nicht angegeben
        if target_type is None:
            if self.tensor_bridge is not None:
                target_type = self.tensor_bridge.preferred_backend
            elif self.matrix_core is not None:
                target_type = self.matrix_core.preferred_backend
            else:
                target_type = 'numpy'  # Sicherer Fallback
        
        try:
            # MLXTensor Konvertierung
            if isinstance(t_math_tensor, MLXTensor):
                mlx_array = t_math_tensor.tensor
                
                if target_type == 'mlx':
                    return mlx_array
                elif self.tensor_bridge is not None:
                    # Verwende VX-MATRIX TensorBridge
                    return self.tensor_bridge.convert(mlx_array, target_type)
                else:
                    # Eigene Konvertierung
                    if target_type == 'numpy':
                        return np.array(mlx_array.tolist())
                    elif target_type == 'torch':
                        import torch
                        np_array = np.array(mlx_array.tolist())
                        return torch.tensor(np_array)
                    elif target_type == 'jax':
                        import jax.numpy as jnp
                        np_array = np.array(mlx_array.tolist())
                        return jnp.array(np_array)
            
            # PTTensor Konvertierung
            elif isinstance(t_math_tensor, PTTensor):
                torch_tensor = t_math_tensor.tensor
                
                if target_type == 'torch':
                    return torch_tensor
                elif self.tensor_bridge is not None:
                    # Verwende VX-MATRIX TensorBridge
                    return self.tensor_bridge.convert(torch_tensor, target_type)
                else:
                    # Eigene Konvertierung
                    if target_type == 'numpy':
                        return torch_tensor.detach().cpu().numpy()
                    elif target_type == 'mlx':
                        import mlx.core as mx
                        np_array = torch_tensor.detach().cpu().numpy()
                        return mx.array(np_array)
                    elif target_type == 'jax':
                        import jax.numpy as jnp
                        np_array = torch_tensor.detach().cpu().numpy()
                        return jnp.array(np_array)
            
            # Unbekannter T-Mathematics-Tensor-Typ
            else:
                ztm_log(f"Unbekannter T-Mathematics-Tensor-Typ: {type(t_math_tensor)}", level="WARNING")
                return t_math_tensor
                
        except Exception as e:
            ztm_log(f"Fehler bei der Konvertierung von T-Mathematics-Tensor: {e}", level="ERROR")
            return t_math_tensor
    
    def perform_matrix_operation(self, op_name: str, *args, **kwargs) -> Any:
        """
        Führt eine Matrixoperation mit der T-Mathematics Engine durch
        
        Args:
            op_name: Name der Operation ('matmul', 'svd', 'inverse', etc.)
            *args, **kwargs: Argumente für die Operation
            
        Returns:
            Ergebnis der Operation
        """
        if not self.initialized or not TMATH_AVAILABLE:
            ztm_log("T-Mathematics Engine nicht verfügbar für Operation", level="ERROR")
            if self.matrix_core is not None:
                ztm_log("Verwende VX-MATRIX-Core als Fallback", level="WARNING")
                return self._perform_matrix_core_operation(op_name, *args, **kwargs)
            return None
        
        self.op_counter["matrix_ops"] += 1
        
        try:
            # Konvertiere Eingaben zu T-Mathematics-Tensoren
            t_math_args = []
            for arg in args:
                if arg is not None:
                    t_math_arg = self.to_t_math_tensor(arg)
                    t_math_args.append(t_math_arg)
                else:
                    t_math_args.append(None)
            
            t_math_kwargs = {}
            for key, value in kwargs.items():
                if value is not None:
                    if isinstance(value, (np.ndarray, list)) or hasattr(value, 'shape'):
                        # Konvertiere nur Tensoren/Arrays
                        t_math_value = self.to_t_math_tensor(value)
                        t_math_kwargs[key] = t_math_value
                    else:
                        t_math_kwargs[key] = value
                else:
                    t_math_kwargs[key] = None
            
            # Führe die Operation mit der T-Mathematics Engine durch
            if op_name == 'matmul':
                self.op_counter["optimized_ops"] += 1
                result = self.engine.matmul(t_math_args[0], t_math_args[1])
            elif op_name == 'svd':
                self.op_counter["optimized_ops"] += 1
                full_matrices = t_math_kwargs.get('full_matrices', True)
                compute_uv = t_math_kwargs.get('compute_uv', True)
                result = self.engine.svd(t_math_args[0], full_matrices=full_matrices, compute_uv=compute_uv)
            elif op_name == 'inverse':
                self.op_counter["optimized_ops"] += 1
                result = self.engine.inverse(t_math_args[0])
            elif op_name == 'transpose':
                self.op_counter["optimized_ops"] += 1
                result = self.engine.transpose(t_math_args[0])
            elif op_name == 'einsum':
                self.op_counter["optimized_ops"] += 1
                equation = t_math_kwargs.get('equation', None)
                if equation is None and len(t_math_args) >= 1:
                    equation = t_math_args[0]
                    t_math_args = t_math_args[1:]
                result = self.engine.einsum(equation, *t_math_args)
            elif op_name == 'concat' or op_name == 'concatenate':
                self.op_counter["optimized_ops"] += 1
                axis = t_math_kwargs.get('axis', 0)
                result = self.engine.concatenate(t_math_args, axis=axis)
            else:
                self.op_counter["fallback_ops"] += 1
                ztm_log(f"Unbekannte Operation: {op_name}, verwende MatrixCore als Fallback", level="WARNING")
                return self._perform_matrix_core_operation(op_name, *args, **kwargs)
            
            # Konvertiere Ergebnis zurück zum gewünschten Zieltyp
            target_type = kwargs.get('target_type', None)
            if target_type is not None:
                return self.from_t_math_tensor(result, target_type)
            else:
                return result
                
        except Exception as e:
            self.op_counter["fallback_ops"] += 1
            ztm_log(f"Fehler bei der T-Mathematics-Operation {op_name}: {e}", level="ERROR")
            
            # Fallback zu MatrixCore, wenn verfügbar
            if self.matrix_core is not None:
                ztm_log("Verwende VX-MATRIX-Core als Fallback", level="WARNING")
                return self._perform_matrix_core_operation(op_name, *args, **kwargs)
            
            # Letzter Ausweg: Ausnahme neu auslösen
            raise
    
    def _perform_matrix_core_operation(self, op_name: str, *args, **kwargs) -> Any:
        """
        Führt eine Matrixoperation mit der MatrixCore durch (Fallback)
        
        Args:
            op_name: Name der Operation
            *args, **kwargs: Argumente für die Operation
            
        Returns:
            Ergebnis der Operation
        """
        if self.matrix_core is None:
            ztm_log("MatrixCore nicht verfügbar für Fallback-Operation", level="ERROR")
            return None
        
        try:
            if op_name == 'matmul':
                return self.matrix_core.matrix_multiply(args[0], args[1])
            elif op_name == 'svd':
                full_matrices = kwargs.get('full_matrices', True)
                return self.matrix_core.svd(args[0], full_matrices=full_matrices)
            elif op_name == 'inverse':
                return self.matrix_core.matrix_inverse(args[0])
            elif op_name == 'transpose':
                # Muss im MatrixCore implementiert werden
                raise NotImplementedError("Transpose nicht implementiert in MatrixCore")
            elif op_name == 'einsum':
                # Muss im MatrixCore implementiert werden
                raise NotImplementedError("Einsum nicht implementiert in MatrixCore")
            elif op_name == 'concat' or op_name == 'concatenate':
                # Muss im MatrixCore implementiert werden
                raise NotImplementedError("Concatenate nicht implementiert in MatrixCore")
            else:
                raise ValueError(f"Unbekannte Operation: {op_name}")
                
        except Exception as e:
            ztm_log(f"Fehler bei der MatrixCore-Fallback-Operation {op_name}: {e}", level="ERROR")
            raise
    
    def get_operation_stats(self) -> Dict[str, int]:
        """
        Gibt Statistiken über die durchgeführten Operationen zurück
        
        Returns:
            Dictionary mit Operation-Zählern
        """
        return self.op_counter.copy()

# Wenn direkt ausgeführt, führe einen kleinen Test durch
if __name__ == "__main__":
    print("T-Mathematics Adapter Test")
    
    # Erstelle T-Mathematics Adapter
    adapter = TMathAdapter()
    print(f"Adapter initialisiert: {adapter.initialized}")
    print(f"MLX-optimiert: {adapter.is_mlx_optimized()}")
    
    if adapter.initialized:
        import numpy as np
        
        # Testmatrix erstellen
        a = np.random.rand(3, 3)
        b = np.random.rand(3, 3)
        
        print("\nMatrix A:")
        print(a)
        print("\nMatrix B:")
        print(b)
        
        # Matrix-Multiplikation über T-Mathematics
        try:
            c = adapter.perform_matrix_operation('matmul', a, b, target_type='numpy')
            print("\nA * B (via T-Mathematics):")
            print(c)
            
            # Konvertierungstest
            t_math_a = adapter.to_t_math_tensor(a)
            print(f"\nT-Mathematics Tensor Typ: {type(t_math_a)}")
            
            # Rückkonvertierung
            a_np = adapter.from_t_math_tensor(t_math_a, 'numpy')
            print("\nZurück zu NumPy:")
            print(a_np)
            
            # Operation Stats
            print("\nOperation Stats:")
            stats = adapter.get_operation_stats()
            for op, count in stats.items():
                print(f"{op}: {count}")
                
        except Exception as e:
            print(f"Fehler bei Testoperationen: {e}")
    else:
        print("Adapter nicht initialisiert, Tests übersprungen")
