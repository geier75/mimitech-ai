#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-VISION Kernel-Modul

Dieses Modul enthält optimierte Bildverarbeitungs-Kernels für verschiedene Hardware-Backends:
- MLX für Apple Silicon (Neural Engine)
- PyTorch für CUDA/MPS/ROCm
- NumPy als Fallback für CPU
"""

# Importiere zuerst die gemeinsamen Komponenten
from vxor.vision.kernels.common import (
    KernelOperation, KernelType, kernel_registry,
    register_kernel, benchmark_kernel, get_best_kernel,
    list_available_operations, list_available_backends
)

# Importiere alle Backend-Implementierungen
# Dies führt dazu, dass die Kernel-Funktionen automatisch registriert werden
from vxor.vision.kernels import mlx_kernels
from vxor.vision.kernels import torch_kernels
from vxor.vision.kernels import numpy_kernels

# Führe eine Initialisierungsfunktion aus, um sicherzustellen, dass alle Kernels geladen sind
def _initialize_kernels():
    """Initialisiert alle verfügbaren Kernels."""
    available_ops = list_available_operations()
    print(f"VX-VISION: {len(available_ops)} Kernel-Operationen geladen: {available_ops}")
    
# Führe die Initialisierung beim Import aus
_initialize_kernels()
