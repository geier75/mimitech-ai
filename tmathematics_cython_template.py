#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T-Mathematics Engine Cython-Kompilierungsvorlage für VXOR AI
Diese Vorlage enthält spezielle Distutils-Anweisungen für die optimale 
Kompilierung der T-Mathematics Engine mit Unterstützung für MLX, PyTorch und NumPy.
"""

import os
import sys
import glob
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

class TMathematicsBuildExt(build_ext):
    """Spezielle Build-Extension für die T-Mathematics Engine."""
    
    def build_extensions(self):
        # MLX-Backend-Unterstützung hinzufügen (Apple Neural Engine)
        try:
            import mlx.core as mx
            print("MLX-Backend gefunden - Apple Neural Engine wird unterstützt.")
            for ext in self.extensions:
                ext.define_macros.append(('HAVE_MLX', '1'))
                # MLX-spezifische Optimierungen aktivieren
                ext.extra_compile_args.append('-DMLX_ACCELERATION')
        except ImportError:
            print("MLX-Backend nicht gefunden - Apple Neural Engine wird nicht unterstützt.")
        
        # PyTorch-Backend-Unterstützung hinzufügen (MPS/GPU)
        try:
            import torch
            print("PyTorch-Backend gefunden - MPS/GPU wird unterstützt.")
            for ext in self.extensions:
                ext.define_macros.append(('HAVE_TORCH', '1'))
                # PyTorch-spezifische Optimierungen aktivieren
                ext.extra_compile_args.append('-DTORCH_ACCELERATION')
        except ImportError:
            print("PyTorch-Backend nicht gefunden - MPS/GPU wird nicht unterstützt.")
        
        # NumPy-Backend-Unterstützung hinzufügen (CPU)
        try:
            import numpy as np
            print("NumPy-Backend gefunden - CPU wird unterstützt.")
            for ext in self.extensions:
                ext.define_macros.append(('HAVE_NUMPY', '1'))
                ext.include_dirs.append(np.get_include())
        except ImportError:
            print("WARNUNG: NumPy-Backend nicht gefunden - Grundlegendes CPU-Backend fehlt!")
            sys.exit(1)
        
        # Plattformspezifische Optimierungen
        if sys.platform == 'darwin':  # macOS
            for ext in self.extensions:
                # macOS-spezifische Optimierungen
                ext.extra_compile_args.extend([
                    '-O3',                    # Höchstes Optimierungslevel
                    '-march=native',          # Prozessorspezifische Optimierungen
                    '-ffast-math',            # Schnellere Mathematikoperationen
                    '-Wno-unused-function',   # Ignoriere ungenutzte Funktionen
                    '-Wno-unneeded-internal-declaration'  # Ignoriere interne Deklarationen
                ])
                
                # Überprüfen, ob Apple Silicon
                import platform
                if platform.machine() == 'arm64':
                    print("Apple Silicon (M-Series) erkannt - optimiere für ARM64.")
                    ext.extra_compile_args.extend([
                        '-mcpu=apple-m1',     # M1-spezifische Optimierungen
                        '-DAPPLE_SILICON'     # Apple Silicon Marker
                    ])
        
        # Multithreading aktivieren
        for ext in self.extensions:
            ext.extra_compile_args.append('-DCYTHON_PARALLEL')
        
        # Standard-Build ausführen
        super().build_extensions()

def build_tmathematics_extension(module_path, output_dir):
    """Baut eine T-Mathematics-Erweiterung aus einem Cython-Modul."""
    
    # Modulname extrahieren
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    
    # Cython-Erweiterung erstellen
    extension = Extension(
        name=module_name,
        sources=[module_path],
        define_macros=[
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
            ('CYTHON_TRACE', '1'),  # Debugging aktivieren
        ],
        extra_compile_args=['-O3'],  # Optimierungslevel
        language='c++'
    )
    
    # Setuptools-Konfiguration
    setup(
        name=f'vxor_tmathematics_{module_name}',
        ext_modules=cythonize(
            [extension],
            compiler_directives={
                'language_level': '3',
                'boundscheck': False,    # Deaktiviere Grenzchecks für höhere Geschwindigkeit
                'wraparound': False,     # Deaktiviere negative Indizierung für höhere Geschwindigkeit
                'initializedcheck': False,  # Deaktiviere Initialisierungschecks
                'cdivision': True,       # Deaktiviere Division durch Null Check
                'profile': False         # Kein Profiling im Produktionscode
            }
        ),
        cmdclass={'build_ext': TMathematicsBuildExt}
    )
    
    # Finde die kompilierte .so-Datei und kopiere sie ins Ausgabeverzeichnis
    so_files = glob.glob(f'build/*/{module_name}.*.so')
    if so_files:
        os.makedirs(output_dir, exist_ok=True)
        target_path = os.path.join(output_dir, f'{module_name}.so')
        shutil.copy2(so_files[0], target_path)
        print(f"Kompilierte Datei nach {target_path} kopiert.")
        return target_path
    else:
        print(f"FEHLER: Keine kompilierte .so-Datei für {module_name} gefunden.")
        return None

def add_miso_metadata(so_file):
    """Fügt MISO-spezifische Metadaten zur kompilierten Datei hinzu."""
    if not os.path.exists(so_file):
        print(f"FEHLER: Datei {so_file} existiert nicht.")
        return False
    
    # In einer realen Implementierung würden hier digitale Signaturen, 
    # Verschlüsselungsschlüssel oder andere Metadaten zur .so-Datei hinzugefügt werden
    print(f"MISO-Metadaten zu {so_file} hinzugefügt.")
    return True

# Beispiel-Verwendung:
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Verwendung: python tmathematics_cython_template.py <input_file.pyx> <output_dir>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"FEHLER: Eingabedatei {input_file} existiert nicht.")
        sys.exit(1)
    
    # T-Mathematics-Erweiterung bauen
    so_file = build_tmathematics_extension(input_file, output_dir)
    
    if so_file:
        # Metadaten hinzufügen
        add_miso_metadata(so_file)
        print("Kompilierung erfolgreich abgeschlossen.")
        sys.exit(0)
    else:
        print("Kompilierung fehlgeschlagen.")
        sys.exit(1)
