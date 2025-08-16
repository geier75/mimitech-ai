#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vXor AGI Test-Hilfsprogramme

Dieses Modul stellt Hilfsfunktionen für Tests bereit, insbesondere für
Dependency Injection und das Patchen von Modulen während der Tests.
"""

import sys
import importlib
import logging
from typing import Dict, Any, Callable, Optional, List, Tuple, Type
from contextlib import contextmanager

logger = logging.getLogger("vxor.test_helpers")

class DependencyInjector:
    """
    Verwaltet die Dependency Injection für Tests.
    Erlaubt das temporäre Ersetzen von Modulen und Objekten während der Tests.
    """
    
    def __init__(self):
        self.original_modules = {}
        self.patched_modules = {}
        self.patched_objects = {}
    
    @contextmanager
    def patch_module(self, module_name: str, mock_module: Any):
        """
        Patcht ein Modul temporär mit einem Mock-Modul.
        
        Args:
            module_name: Name des zu patchenden Moduls
            mock_module: Das Mock-Modul, das das Original ersetzen soll
            
        Yields:
            Das Mock-Modul
        """
        if module_name in sys.modules:
            self.original_modules[module_name] = sys.modules[module_name]
        
        sys.modules[module_name] = mock_module
        self.patched_modules[module_name] = mock_module
        
        try:
            yield mock_module
        finally:
            # Stelle das Originalmodul wieder her
            if module_name in self.original_modules:
                sys.modules[module_name] = self.original_modules[module_name]
            else:
                sys.modules.pop(module_name, None)
            
            self.patched_modules.pop(module_name, None)
    
    @contextmanager
    def patch_object(self, target_object: Any, attribute_name: str, mock_object: Any):
        """
        Patcht ein Attribut eines Objekts temporär mit einem Mock-Objekt.
        
        Args:
            target_object: Das Objekt, dessen Attribut gepatcht werden soll
            attribute_name: Name des zu patchenden Attributs
            mock_object: Das Mock-Objekt, das das Original ersetzen soll
            
        Yields:
            Das Mock-Objekt
        """
        original = None
        has_original = hasattr(target_object, attribute_name)
        
        if has_original:
            original = getattr(target_object, attribute_name)
        
        setattr(target_object, attribute_name, mock_object)
        self.patched_objects[(id(target_object), attribute_name)] = (target_object, mock_object, original, has_original)
        
        try:
            yield mock_object
        finally:
            # Stelle das Originalattribut wieder her
            if has_original:
                setattr(target_object, attribute_name, original)
            else:
                delattr(target_object, attribute_name)
            
            self.patched_objects.pop((id(target_object), attribute_name), None)
    
    def restore_all(self):
        """Stellt alle gepatchten Module und Objekte wieder her."""
        # Stelle gepatchte Module wieder her
        for module_name, original in self.original_modules.items():
            sys.modules[module_name] = original
        self.original_modules.clear()
        self.patched_modules.clear()
        
        # Stelle gepatchte Objekte wieder her
        for key, (target, _, original, has_original) in self.patched_objects.items():
            if has_original:
                setattr(target, key[1], original)
            else:
                delattr(target, key[1])
        self.patched_objects.clear()


class ModuleTestHelper:
    """
    Hilfsklasse für Tests von vXor-Modulen mit standardisierten Entry-Points.
    """
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.module = None
        self.injector = DependencyInjector()
        self.dependencies = {}
    
    def setup(self, mock_dependencies: Dict[str, Any] = None):
        """
        Richtet die Testumgebung für ein Modul ein.
        
        Args:
            mock_dependencies: Dictionary mit Mock-Modulen, die Abhängigkeiten ersetzen sollen
        """
        # Patche Abhängigkeiten, falls angegeben
        if mock_dependencies:
            for dep_name, mock_dep in mock_dependencies.items():
                self.dependencies[dep_name] = self.injector.patch_module(dep_name, mock_dep)
        
        # Importiere das zu testende Modul
        try:
            self.module = importlib.import_module(self.module_name)
            logger.info(f"Modul {self.module_name} erfolgreich für Tests geladen")
            return True
        except ImportError as e:
            logger.error(f"Fehler beim Importieren des Moduls {self.module_name}: {e}")
            return False
    
    def teardown(self):
        """Räumt die Testumgebung auf."""
        self.injector.restore_all()
        
        # Entferne das Modul aus dem Cache, damit es beim nächsten Mal neu geladen wird
        if self.module_name in sys.modules:
            del sys.modules[self.module_name]
        
        self.module = None
    
    def test_entry_points(self) -> Dict[str, bool]:
        """
        Testet die standardisierten Entry-Points des Moduls.
        
        Returns:
            Dictionary mit Testergebnissen für jeden Entry-Point
        """
        if not self.module:
            logger.error("Modul ist nicht geladen, führe setup() zuerst aus")
            return {}
        
        results = {}
        
        # Liste der zu testenden Entry-Points
        entry_points = ['init', 'boot', 'configure', 'start', 'setup', 'activate']
        
        for ep in entry_points:
            if hasattr(self.module, ep) and callable(getattr(self.module, ep)):
                try:
                    # Rufe den Entry-Point auf
                    result = getattr(self.module, ep)()
                    
                    # Prüfe, ob der Entry-Point ein gültiges Ergebnis zurückgibt
                    results[ep] = result is not None
                    
                    logger.info(f"Entry-Point {ep} erfolgreich getestet")
                except Exception as e:
                    logger.error(f"Fehler beim Testen des Entry-Points {ep}: {e}")
                    results[ep] = False
            else:
                results[ep] = False
                logger.warning(f"Entry-Point {ep} nicht im Modul {self.module_name} gefunden")
        
        return results


# Globale Hilfsfunktionen

def create_module_test(module_name: str, mock_dependencies: Dict[str, Any] = None) -> ModuleTestHelper:
    """
    Erstellt einen ModuleTestHelper für ein bestimmtes Modul.
    
    Args:
        module_name: Name des zu testenden Moduls
        mock_dependencies: Dictionary mit Mock-Modulen, die Abhängigkeiten ersetzen sollen
        
    Returns:
        ModuleTestHelper-Instanz
    """
    helper = ModuleTestHelper(module_name)
    helper.setup(mock_dependencies)
    return helper


@contextmanager
def module_test_context(module_name: str, mock_dependencies: Dict[str, Any] = None):
    """
    Context-Manager für Modultests mit automatischem Setup und Teardown.
    
    Args:
        module_name: Name des zu testenden Moduls
        mock_dependencies: Dictionary mit Mock-Modulen, die Abhängigkeiten ersetzen sollen
        
    Yields:
        ModuleTestHelper-Instanz und das geladene Modul
    """
    helper = ModuleTestHelper(module_name)
    helper.setup(mock_dependencies)
    
    try:
        yield helper, helper.module
    finally:
        helper.teardown()
