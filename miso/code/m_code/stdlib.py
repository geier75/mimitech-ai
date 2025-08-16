#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Standard Library

Dieses Modul implementiert die Standardbibliothek für die M-CODE Programmiersprache.
Die Standardbibliothek stellt grundlegende Funktionen und Klassen zur Verfügung.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import logging
import sys
import time
import math
import random
import datetime
import json
import re
import uuid
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from dataclasses import dataclass, field

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.stdlib")

# Import von internen Modulen
from .runtime import MCodeFunction, MCodeClass, MCodeModule, MCodeObject


# Registry für Standardbibliotheksfunktionen
_STDLIB_FUNCTIONS = {}
_STDLIB_CLASSES = {}
_STDLIB_MODULES = {}


def register_function(name: str, func: Callable, module: str = "stdlib") -> None:
    """
    Registriert eine Funktion in der Standardbibliothek.
    
    Args:
        name: Name der Funktion
        func: Funktion
        module: Name des Moduls
    """
    if module not in _STDLIB_MODULES:
        _STDLIB_MODULES[module] = {}
    
    _STDLIB_MODULES[module][name] = func
    _STDLIB_FUNCTIONS[name] = func
    
    logger.debug(f"Funktion '{name}' in Modul '{module}' registriert")


def register_class(name: str, cls: type, module: str = "stdlib") -> None:
    """
    Registriert eine Klasse in der Standardbibliothek.
    
    Args:
        name: Name der Klasse
        cls: Klasse
        module: Name des Moduls
    """
    if module not in _STDLIB_MODULES:
        _STDLIB_MODULES[module] = {}
    
    _STDLIB_MODULES[module][name] = cls
    _STDLIB_CLASSES[name] = cls
    
    logger.debug(f"Klasse '{name}' in Modul '{module}' registriert")


def get_stdlib_functions() -> Dict[str, Callable]:
    """
    Gibt alle registrierten Standardbibliotheksfunktionen zurück.
    
    Returns:
        Wörterbuch mit Funktionsnamen und Funktionen
    """
    return _STDLIB_FUNCTIONS


def get_stdlib_classes() -> Dict[str, type]:
    """
    Gibt alle registrierten Standardbibliotheksklassen zurück.
    
    Returns:
        Wörterbuch mit Klassennamen und Klassen
    """
    return _STDLIB_CLASSES


def get_stdlib_module(name: str) -> Dict[str, Any]:
    """
    Gibt ein Standardbibliotheksmodul zurück.
    
    Args:
        name: Name des Moduls
        
    Returns:
        Wörterbuch mit Modulinhalten
        
    Raises:
        KeyError: Wenn das Modul nicht existiert
    """
    if name in _STDLIB_MODULES:
        return _STDLIB_MODULES[name]
    
    raise KeyError(f"Modul '{name}' nicht gefunden")


def get_all_stdlib_modules() -> Dict[str, Dict[str, Any]]:
    """
    Gibt alle Standardbibliotheksmodule zurück.
    
    Returns:
        Wörterbuch mit Modulnamen und Modulinhalten
    """
    return _STDLIB_MODULES


# Registriere Standardfunktionen
register_function("print", print)
register_function("len", len)
register_function("range", range)
register_function("str", str)
register_function("int", int)
register_function("float", float)
register_function("bool", bool)
register_function("list", list)
register_function("dict", dict)
register_function("set", set)
register_function("tuple", tuple)
register_function("sum", sum)
register_function("min", min)
register_function("max", max)
register_function("abs", abs)
register_function("round", round)


# Math-Modul
def _register_math_module():
    """Registriert das Math-Modul"""
    math_functions = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "asinh": math.asinh,
        "acosh": math.acosh,
        "atanh": math.atanh,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "pow": math.pow,
        "sqrt": math.sqrt,
        "ceil": math.ceil,
        "floor": math.floor,
        "trunc": math.trunc,
        "isnan": math.isnan,
        "isinf": math.isinf,
        "isfinite": math.isfinite,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "degrees": math.degrees,
        "radians": math.radians
    }
    
    math_constants = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
        "nan": math.nan
    }
    
    for name, func in math_functions.items():
        register_function(name, func, "math")
    
    for name, const in math_constants.items():
        register_function(name, lambda c=const: c, "math")


# Random-Modul
def _register_random_module():
    """Registriert das Random-Modul"""
    random_functions = {
        "random": random.random,
        "randint": random.randint,
        "uniform": random.uniform,
        "choice": random.choice,
        "choices": random.choices,
        "sample": random.sample,
        "shuffle": random.shuffle,
        "seed": random.seed,
        "getstate": random.getstate,
        "setstate": random.setstate,
        "randrange": random.randrange,
        "normalvariate": random.normalvariate,
        "lognormvariate": random.lognormvariate,
        "expovariate": random.expovariate,
        "vonmisesvariate": random.vonmisesvariate,
        "gammavariate": random.gammavariate,
        "gauss": random.gauss,
        "betavariate": random.betavariate,
        "paretovariate": random.paretovariate,
        "weibullvariate": random.weibullvariate
    }
    
    for name, func in random_functions.items():
        register_function(name, func, "random")


# Time-Modul
def _register_time_module():
    """Registriert das Time-Modul"""
    time_functions = {
        "time": time.time,
        "sleep": time.sleep,
        "ctime": time.ctime,
        "gmtime": time.gmtime,
        "localtime": time.localtime,
        "mktime": time.mktime,
        "strftime": time.strftime,
        "strptime": time.strptime,
        "monotonic": time.monotonic,
        "perf_counter": time.perf_counter,
        "process_time": time.process_time
    }
    
    for name, func in time_functions.items():
        register_function(name, func, "time")


# JSON-Modul
def _register_json_module():
    """Registriert das JSON-Modul"""
    json_functions = {
        "loads": json.loads,
        "dumps": json.dumps,
        "load": json.load,
        "dump": json.dump
    }
    
    for name, func in json_functions.items():
        register_function(name, func, "json")


# Regex-Modul
def _register_regex_module():
    """Registriert das Regex-Modul"""
    regex_functions = {
        "match": re.match,
        "search": re.search,
        "findall": re.findall,
        "finditer": re.finditer,
        "split": re.split,
        "sub": re.sub,
        "subn": re.subn,
        "escape": re.escape,
        "compile": re.compile
    }
    
    for name, func in regex_functions.items():
        register_function(name, func, "regex")


# UUID-Modul
def _register_uuid_module():
    """Registriert das UUID-Modul"""
    uuid_functions = {
        "uuid1": uuid.uuid1,
        "uuid3": uuid.uuid3,
        "uuid4": uuid.uuid4,
        "uuid5": uuid.uuid5
    }
    
    for name, func in uuid_functions.items():
        register_function(name, func, "uuid")


# NumPy-Modul
def _register_numpy_module():
    """Registriert das NumPy-Modul"""
    numpy_functions = {
        "array": np.array,
        "zeros": np.zeros,
        "ones": np.ones,
        "empty": np.empty,
        "eye": np.eye,
        "identity": np.identity,
        "arange": np.arange,
        "linspace": np.linspace,
        "logspace": np.logspace,
        "random": np.random.random,
        "randn": np.random.randn,
        "randint": np.random.randint,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "sum": np.sum,
        "mean": np.mean,
        "std": np.std,
        "var": np.var,
        "min": np.min,
        "max": np.max,
        "argmin": np.argmin,
        "argmax": np.argmax,
        "dot": np.dot,
        "matmul": np.matmul,
        "transpose": np.transpose,
        "reshape": lambda arr, shape: arr.reshape(shape),
        "concatenate": np.concatenate,
        "stack": np.stack,
        "vstack": np.vstack,
        "hstack": np.hstack,
        "split": np.split,
        "vsplit": np.vsplit,
        "hsplit": np.hsplit
    }
    
    for name, func in numpy_functions.items():
        register_function(name, func, "numpy")


# PyTorch-Modul
def _register_torch_module():
    """Registriert das PyTorch-Modul"""
    torch_functions = {
        "tensor": torch.tensor,
        "zeros": torch.zeros,
        "ones": torch.ones,
        "empty": torch.empty,
        "eye": torch.eye,
        "arange": torch.arange,
        "linspace": torch.linspace,
        "logspace": torch.logspace,
        "rand": torch.rand,
        "randn": torch.randn,
        "randint": torch.randint,
        "sin": torch.sin,
        "cos": torch.cos,
        "tan": torch.tan,
        "exp": torch.exp,
        "log": torch.log,
        "sqrt": torch.sqrt,
        "abs": torch.abs,
        "sum": torch.sum,
        "mean": torch.mean,
        "std": torch.std,
        "var": torch.var,
        "min": torch.min,
        "max": torch.max,
        "argmin": torch.argmin,
        "argmax": torch.argmax,
        "matmul": torch.matmul,
        "mm": torch.mm,
        "bmm": torch.bmm,
        "transpose": lambda t, dim0, dim1: t.transpose(dim0, dim1),
        "reshape": lambda t, shape: t.reshape(shape),
        "view": lambda t, shape: t.view(shape),
        "cat": torch.cat,
        "stack": torch.stack,
        "chunk": torch.chunk,
        "split": torch.split,
        "cuda": lambda t: t.cuda() if torch.cuda.is_available() else t,
        "cpu": lambda t: t.cpu(),
        "to": lambda t, device: t.to(device),
        "device": lambda name: torch.device(name),
        "save": torch.save,
        "load": torch.load
    }
    
    for name, func in torch_functions.items():
        register_function(name, func, "torch")


# Tensor-Modul (M-CODE spezifisch)
# Die verbesserte MTensor-Klasse wird aus dem tensor-Modul importiert
from .tensor import MTensor, tensor, zeros, ones, eye, random, jit


def _register_tensor_module():
    """Registriert das Tensor-Modul"""
    # Registriere Tensor-Klasse
    register_class("Tensor", MTensor, module="tensor")
    
    # Registriere Tensor-Funktionen
    register_function("tensor", tensor, module="tensor")
    register_function("zeros", zeros, module="tensor")
    register_function("ones", ones, module="tensor")
    register_function("eye", eye, module="tensor")
    register_function("random", random, module="tensor")
    register_function("jit", jit, module="tensor")
    
    # Registriere MLX-spezifische Funktionen, wenn MLX verfügbar ist
    from .mlx_adapter import MLX_AVAILABLE
    if MLX_AVAILABLE:
        logger.info("MLX-unterstützte Tensor-Operationen registriert")
    else:
        logger.warning("MLX nicht verfügbar, verwende NumPy-Fallback für Tensor-Operationen")
    
    logger.info("Tensor-Modul initialisiert")


# Quantum-Modul (M-CODE spezifisch)
class MQuantumState:
    """M-CODE Quantenzustand-Klasse"""
    
    def __init__(self, amplitudes=None):
        """
        Initialisiert einen neuen M-CODE Quantenzustand.
        
        Args:
            amplitudes: Wörterbuch mit Zuständen und komplexen Amplituden
        """
        self.amplitudes = amplitudes or {}
        self._normalize()
    
    def __repr__(self) -> str:
        return f"MQuantumState({self.amplitudes})"
    
    def add_state(self, state, amplitude):
        """Fügt einen Zustand hinzu"""
        self.amplitudes[state] = amplitude
        self._normalize()
        return self
    
    def measure(self):
        """Führt eine Messung durch"""
        states = list(self.amplitudes.keys())
        probs = [abs(amp) ** 2 for amp in self.amplitudes.values()]
        
        if not states:
            return None
        
        result = random.choices(states, weights=probs, k=1)[0]
        
        # Kollabiere Zustand
        self.amplitudes = {result: 1.0}
        
        return result
    
    def _normalize(self):
        """Normalisiert die Amplituden"""
        if not self.amplitudes:
            return
        
        norm = sum(abs(amp) ** 2 for amp in self.amplitudes.values())
        
        if norm > 0:
            norm = math.sqrt(norm)
            for state in self.amplitudes:
                self.amplitudes[state] /= norm


def _register_quantum_module():
    """Registriert das Quantum-Modul"""
    quantum_functions = {
        "quantum_state": MQuantumState,
        "measure": lambda state: state.measure() if isinstance(state, MQuantumState) else None,
        "hadamard": lambda state, target: _hadamard(state, target),
        "phase": lambda state, target, angle: _phase(state, target, angle),
        "cnot": lambda state, control, target: _cnot(state, control, target),
        "swap": lambda state, a, b: _swap(state, a, b)
    }
    
    for name, func in quantum_functions.items():
        register_function(name, func, "quantum")
    
    register_class("MQuantumState", MQuantumState, "quantum")


def _hadamard(state, target):
    """Wendet das Hadamard-Gatter an"""
    if not isinstance(state, MQuantumState):
        return state
    
    new_state = MQuantumState()
    
    for s, amp in state.amplitudes.items():
        # Annahme: Zustand ist eine Binärzeichenkette
        if len(s) <= target:
            continue
        
        bit = s[target]
        
        # Erstelle neue Zustände
        if bit == '0':
            s0 = s[:target] + '0' + s[target+1:]
            s1 = s[:target] + '1' + s[target+1:]
            new_state.amplitudes[s0] = new_state.amplitudes.get(s0, 0) + amp / math.sqrt(2)
            new_state.amplitudes[s1] = new_state.amplitudes.get(s1, 0) + amp / math.sqrt(2)
        else:
            s0 = s[:target] + '0' + s[target+1:]
            s1 = s[:target] + '1' + s[target+1:]
            new_state.amplitudes[s0] = new_state.amplitudes.get(s0, 0) + amp / math.sqrt(2)
            new_state.amplitudes[s1] = new_state.amplitudes.get(s1, 0) - amp / math.sqrt(2)
    
    new_state._normalize()
    return new_state


def _phase(state, target, angle):
    """Wendet das Phasen-Gatter an"""
    if not isinstance(state, MQuantumState):
        return state
    
    new_state = MQuantumState()
    
    for s, amp in state.amplitudes.items():
        # Annahme: Zustand ist eine Binärzeichenkette
        if len(s) <= target:
            continue
        
        bit = s[target]
        
        if bit == '1':
            new_state.amplitudes[s] = amp * complex(math.cos(angle), math.sin(angle))
        else:
            new_state.amplitudes[s] = amp
    
    new_state._normalize()
    return new_state


def _cnot(state, control, target):
    """Wendet das CNOT-Gatter an"""
    if not isinstance(state, MQuantumState):
        return state
    
    new_state = MQuantumState()
    
    for s, amp in state.amplitudes.items():
        # Annahme: Zustand ist eine Binärzeichenkette
        if len(s) <= max(control, target):
            continue
        
        control_bit = s[control]
        
        if control_bit == '1':
            # Invertiere Target-Bit
            target_bit = s[target]
            new_target_bit = '1' if target_bit == '0' else '0'
            new_s = s[:target] + new_target_bit + s[target+1:]
            new_state.amplitudes[new_s] = amp
        else:
            new_state.amplitudes[s] = amp
    
    new_state._normalize()
    return new_state


def _swap(state, a, b):
    """Wendet das SWAP-Gatter an"""
    if not isinstance(state, MQuantumState):
        return state
    
    new_state = MQuantumState()
    
    for s, amp in state.amplitudes.items():
        # Annahme: Zustand ist eine Binärzeichenkette
        if len(s) <= max(a, b):
            continue
        
        bit_a = s[a]
        bit_b = s[b]
        
        if bit_a != bit_b:
            # Tausche Bits
            new_s = list(s)
            new_s[a], new_s[b] = new_s[b], new_s[a]
            new_s = ''.join(new_s)
            new_state.amplitudes[new_s] = amp
        else:
            new_state.amplitudes[s] = amp
    
    new_state._normalize()
    return new_state


# Initialisiere alle Module
def initialize_stdlib():
    """Initialisiert die Standardbibliothek"""
    _register_math_module()
    _register_random_module()
    _register_time_module()
    _register_json_module()
    _register_regex_module()
    _register_uuid_module()
    _register_numpy_module()
    _register_torch_module()
    _register_tensor_module()
    _register_quantum_module()
    
    logger.info("M-CODE Standardbibliothek initialisiert")


# Initialisiere Standardbibliothek beim Import
initialize_stdlib()
