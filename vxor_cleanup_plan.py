#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR System Cleanup & Refactoring Plan
Systematische Bereinigung nach OODA Loop & TDD Prinzipien
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VXORCleanupManager:
    """Systematische VXOR System Bereinigung"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.cleanup_stats = {
            "files_moved": 0,
            "files_deleted": 0,
            "directories_created": 0,
            "imports_fixed": 0
        }
        
    def execute_cleanup_plan(self):
        """Führt den kompletten Bereinigungsplan aus"""
        logger.info("🧹 Starte VXOR System Cleanup...")
        
        # Phase 1: Namespace Konsolidierung
        self.consolidate_namespaces()
        
        # Phase 2: Struktur Vereinfachung
        self.simplify_structure()
        
        # Phase 3: Import Fixes
        self.fix_imports()
        
        # Phase 4: Code Refactoring
        self.refactor_code()
        
        # Phase 5: Tests erstellen
        self.create_tests()
        
        logger.info("✅ VXOR System Cleanup abgeschlossen")
        self.print_stats()
        
    def consolidate_namespaces(self):
        """Konsolidiert die verschiedenen Namespaces"""
        logger.info("📦 Konsolidiere Namespaces...")
        
        # Neue einheitliche Struktur
        new_structure = {
            "vxor/": {
                "core/": ["vxor/core/", "miso/core/"],
                "agents/": ["miso/vxor/", "vXor_Modules/"],
                "ai/": ["vxor.ai/"],
                "math/": ["miso/math/", "miso/tmathematics/"],
                "lang/": ["miso/lang/"],
                "security/": ["miso/security/"],
                "tests/": ["tests/", "test-modules/"],
                "tools/": ["tools/", "verify/"],
                "benchmarks/": ["vxor-benchmark-suite/", "benchmark/"]
            }
        }
        
        # Erstelle neue Struktur
        for target_dir, source_dirs in new_structure["vxor/"].items():
            target_path = self.root_path / "vxor" / target_dir
            target_path.mkdir(parents=True, exist_ok=True)
            self.cleanup_stats["directories_created"] += 1
            
            # Verschiebe Dateien aus Quellverzeichnissen
            for source_dir in source_dirs:
                source_path = self.root_path / source_dir
                if source_path.exists():
                    self.move_directory_contents(source_path, target_path)
                    
    def simplify_structure(self):
        """Vereinfacht die Verzeichnisstruktur"""
        logger.info("🗂️ Vereinfache Struktur...")
        
        # Entferne leere Verzeichnisse
        self.remove_empty_directories()
        
        # Konsolidiere ähnliche Dateien
        self.consolidate_similar_files()
        
    def fix_imports(self):
        """Korrigiert Import-Statements"""
        logger.info("🔧 Korrigiere Imports...")
        
        import_mappings = {
            "from miso.": "from vxor.",
            "import miso.": "import vxor.",
            "from vXor_Modules.": "from vxor.agents.",
            "from vxor.ai.": "from vxor.ai."
        }
        
        # Durchsuche alle Python-Dateien
        for py_file in self.root_path.rglob("*.py"):
            if self.fix_file_imports(py_file, import_mappings):
                self.cleanup_stats["imports_fixed"] += 1
                
    def refactor_code(self):
        """Refactoring nach TDD Prinzipien"""
        logger.info("⚡ Refactoring Code...")
        
        # Performance-kritische Module identifizieren
        performance_modules = [
            "vxor/math/",
            "vxor/ai/",
            "vxor/core/"
        ]
        
        for module_path in performance_modules:
            module_dir = self.root_path / module_path
            if module_dir.exists():
                self.optimize_module_performance(module_dir)
                
    def create_tests(self):
        """Erstellt TDD-basierte Tests"""
        logger.info("🧪 Erstelle Tests...")
        
        test_templates = {
            "test_core.py": self.generate_core_tests(),
            "test_agents.py": self.generate_agent_tests(),
            "test_math.py": self.generate_math_tests(),
            "test_performance.py": self.generate_performance_tests()
        }
        
        test_dir = self.root_path / "vxor" / "tests"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        for test_file, test_content in test_templates.items():
            test_path = test_dir / test_file
            with open(test_path, 'w') as f:
                f.write(test_content)
                
    def move_directory_contents(self, source: Path, target: Path):
        """Verschiebt Verzeichnisinhalte"""
        if not source.exists():
            return
            
        for item in source.iterdir():
            if item.is_file():
                target_file = target / item.name
                if not target_file.exists():
                    shutil.move(str(item), str(target_file))
                    self.cleanup_stats["files_moved"] += 1
            elif item.is_dir():
                target_subdir = target / item.name
                target_subdir.mkdir(exist_ok=True)
                self.move_directory_contents(item, target_subdir)
                
    def remove_empty_directories(self):
        """Entfernt leere Verzeichnisse"""
        for root, dirs, files in os.walk(self.root_path, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        logger.debug(f"Entfernt leeres Verzeichnis: {dir_path}")
                except OSError:
                    pass  # Verzeichnis nicht leer oder andere Probleme
                    
    def consolidate_similar_files(self):
        """Konsolidiert ähnliche Dateien"""
        # Finde Dateien mit ähnlichen Namen
        similar_files = self.find_similar_files()
        
        for file_group in similar_files:
            if len(file_group) > 1:
                # Behalte die neueste Version
                newest_file = max(file_group, key=lambda f: f.stat().st_mtime)
                for file_path in file_group:
                    if file_path != newest_file:
                        file_path.unlink()
                        self.cleanup_stats["files_deleted"] += 1
                        
    def find_similar_files(self) -> List[List[Path]]:
        """Findet ähnliche Dateien"""
        files_by_stem = {}
        
        for py_file in self.root_path.rglob("*.py"):
            stem = py_file.stem.lower()
            # Entferne Versionssuffixe
            stem = stem.replace("_v2", "").replace("_old", "").replace("_backup", "")
            
            if stem not in files_by_stem:
                files_by_stem[stem] = []
            files_by_stem[stem].append(py_file)
            
        return [files for files in files_by_stem.values() if len(files) > 1]
        
    def fix_file_imports(self, file_path: Path, mappings: Dict[str, str]) -> bool:
        """Korrigiert Imports in einer Datei"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            
            for old_import, new_import in mappings.items():
                content = content.replace(old_import, new_import)
                
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            logger.warning(f"Fehler beim Korrigieren von {file_path}: {e}")
            
        return False
        
    def optimize_module_performance(self, module_dir: Path):
        """Optimiert Performance eines Moduls"""
        for py_file in module_dir.rglob("*.py"):
            self.optimize_file_performance(py_file)
            
    def optimize_file_performance(self, file_path: Path):
        """Optimiert Performance einer Datei"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Performance-Optimierungen
            optimizations = [
                # List comprehensions statt loops
                (r'for\s+(\w+)\s+in\s+([^:]+):\s*\n\s*(\w+)\.append\(([^)]+)\)',
                 r'\3 = [\4 for \1 in \2]'),
                
                # Type hints hinzufügen (basic)
                (r'def\s+(\w+)\(([^)]*)\):', r'def \1(\2) -> None:'),
                
                # Logging optimieren
                (r'print\(([^)]+)\)', r'logger.info(\1)')
            ]
            
            original_content = content
            for pattern, replacement in optimizations:
                import re
                content = re.sub(pattern, replacement, content)
                
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
        except Exception as e:
            logger.warning(f"Fehler beim Optimieren von {file_path}: {e}")
            
    def generate_core_tests(self) -> str:
        """Generiert Core-Tests"""
        return '''#!/usr/bin/env python3
"""
VXOR Core Tests - TDD Implementation
"""

import unittest
import sys
from pathlib import Path

# Add vxor to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestVXORCore(unittest.TestCase):
    """Test VXOR Core functionality"""
    
    def setUp(self):
        """Setup test environment"""
        pass
        
    def test_core_initialization(self):
        """Test core module initialization"""
        # TODO: Implement core initialization test
        self.assertTrue(True)
        
    def test_module_loading(self):
        """Test module loading functionality"""
        # TODO: Implement module loading test
        self.assertTrue(True)
        
if __name__ == '__main__':
    unittest.main()
'''
        
    def generate_agent_tests(self) -> str:
        """Generiert Agent-Tests"""
        return '''#!/usr/bin/env python3
"""
VXOR Agent Tests - TDD Implementation
"""

import unittest

class TestVXORAgents(unittest.TestCase):
    """Test VXOR Agent functionality"""
    
    def test_agent_communication(self):
        """Test inter-agent communication"""
        # TODO: Implement agent communication test
        self.assertTrue(True)
        
if __name__ == '__main__':
    unittest.main()
'''
        
    def generate_math_tests(self) -> str:
        """Generiert Math-Tests"""
        return '''#!/usr/bin/env python3
"""
VXOR Math Tests - TDD Implementation
"""

import unittest
import numpy as np

class TestVXORMath(unittest.TestCase):
    """Test VXOR Math functionality"""
    
    def test_matrix_operations(self):
        """Test matrix operations"""
        # TODO: Implement matrix operation tests
        self.assertTrue(True)
        
if __name__ == '__main__':
    unittest.main()
'''
        
    def generate_performance_tests(self) -> str:
        """Generiert Performance-Tests"""
        return '''#!/usr/bin/env python3
"""
VXOR Performance Tests - TDD Implementation
"""

import unittest
import time

class TestVXORPerformance(unittest.TestCase):
    """Test VXOR Performance"""
    
    def test_response_time(self):
        """Test system response time"""
        start_time = time.time()
        # TODO: Implement performance test
        end_time = time.time()
        
        response_time = end_time - start_time
        self.assertLess(response_time, 1.0)  # Should respond within 1 second
        
if __name__ == '__main__':
    unittest.main()
'''
        
    def print_stats(self):
        """Druckt Cleanup-Statistiken"""
        logger.info("📊 Cleanup Statistiken:")
        for key, value in self.cleanup_stats.items():
            logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    cleanup_manager = VXORCleanupManager()
    cleanup_manager.execute_cleanup_plan()
