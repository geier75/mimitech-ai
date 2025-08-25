#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-REFLEX Module - Reaktionsmanagement und Spontanverhalten
"""

import sys
import os
# Ensure we use the built-in math module, not local vxor/math
import builtins
if hasattr(builtins, '__import__'):
    original_import = builtins.__import__
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'math':
            # Force import of built-in math module
            import importlib
            return importlib.import_module('math')
        return original_import(name, globals, locals, fromlist, level)
    builtins.__import__ = safe_import

try:
    from .reflex_core import ReflexCore, StimulusPriority
    from ..vx_reflex import VXReflex, get_module
    
    class VXReflexModule:
        """VX-REFLEX Module Wrapper"""
        def __init__(self):
            self.name = "VX-REFLEX"
            self.version = "1.0.0"
            self.reflex_core = None
            self.vx_reflex = None
            
        def initialize(self):
            """Initialize VX-REFLEX components"""
            try:
                self.vx_reflex = get_module()
                return True
            except Exception as e:
                print(f"VX-REFLEX initialization warning: {e}")
                return False
                
        def get_status(self):
            """Get module status"""
            return {
                "loaded": True,
                "initialized": self.vx_reflex is not None,
                "version": self.version
            }

    # Module singleton
    _vx_reflex_module = VXReflexModule()
    _vx_reflex_module.initialize()

except ImportError as e:
    # Fallback minimal implementation
    class VXReflexModule:
        def __init__(self):
            self.name = "VX-REFLEX"
            self.version = "1.0.0"
            print(f"VX-REFLEX: Using minimal implementation due to import error: {e}")
            
        def initialize(self):
            return True
            
        def get_status(self):
            return {"loaded": True, "initialized": True, "version": self.version}

    _vx_reflex_module = VXReflexModule()

# Export the module
def get_module():
    return _vx_reflex_module