#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests für die Ressourcenüberwachung und -limitierung der M-CODE Security Sandbox.
"""

import unittest
import time
import sys
import os
import psutil
import threading
import numpy as np
from unittest.mock import patch, MagicMock

# Pfad zum Hauptverzeichnis hinzufügen, um die Module zu importieren
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from security.m_code_security_sandbox import (
    MCodeSecuritySandbox, ResourceMonitor, SecurityLevel, 
    ResourceLimitExceededError, SecurityViolationError
)


class TestResourceMonitor(unittest.TestCase):
    """Tests für die ResourceMonitor-Klasse."""
    
    def setUp(self):
        """Testumgebung vorbereiten."""
        self.process = psutil.Process()
        
    def test_init(self):
        """Test der Initialisierung."""
        monitor = ResourceMonitor(
            max_cpu_percent=50.0,
            max_memory_mb=100.0,
            max_file_size_mb=10.0,
            max_open_files=5,
            timeout_seconds=5
        )
        
        self.assertEqual(monitor.max_cpu_percent, 50.0)
        self.assertEqual(monitor.max_memory_mb, 100.0)
        self.assertEqual(monitor.max_file_size_mb, 10.0)
        self.assertEqual(monitor.max_open_files, 5)
        self.assertEqual(monitor.timeout_seconds, 5)
        self.assertFalse(monitor.violation)
        self.assertFalse(monitor._monitoring)

    def test_start_stop_monitoring(self):
        """Test des Starts und Stopps der Überwachung."""
        monitor = ResourceMonitor(
            max_cpu_percent=50.0,
            max_memory_mb=100.0,
            max_file_size_mb=10.0,
            max_open_files=5,
            timeout_seconds=5
        )
        
        # Überwachung starten
        monitor.start_monitoring()
        self.assertTrue(monitor._monitoring)
        self.assertIsNotNone(monitor._monitor_thread)
        self.assertTrue(monitor._monitor_thread.is_alive())
        
        # Überwachung stoppen
        monitor.stop_monitoring()
        self.assertFalse(monitor._monitoring)
        time.sleep(0.1)  # Kurze Pause für Thread-Beendigung
        self.assertFalse(monitor._monitor_thread.is_alive())

    @patch('psutil.Process')
    def test_check_resource_usage_cpu_limit_exceeded(self, mock_process):
        """Test der CPU-Limitüberwachung."""
        # Mock-Prozess mit hoher CPU-Auslastung konfigurieren
        mock_process_instance = MagicMock()
        mock_process_instance.cpu_percent.return_value = 90.0
        mock_process.return_value = mock_process_instance
        
        monitor = ResourceMonitor(
            max_cpu_percent=50.0,
            max_memory_mb=100.0,
            max_file_size_mb=10.0,
            max_open_files=5,
            timeout_seconds=5
        )
        monitor.process = mock_process_instance
        
        # Ressourcenprüfung durchführen
        monitor._check_resource_usage()
        
        self.assertEqual(monitor.violation, "CPU_LIMIT_EXCEEDED")

    @patch('psutil.Process')
    def test_check_resource_usage_memory_limit_exceeded(self, mock_process):
        """Test der Speicher-Limitüberwachung."""
        # Mock-Prozess mit hohem Speicherverbrauch konfigurieren
        mock_process_instance = MagicMock()
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 200 * 1024 * 1024  # 200 MB
        mock_process_instance.memory_info.return_value = mock_memory_info
        mock_process_instance.cpu_percent.return_value = 10.0  # Niedrige CPU-Auslastung
        mock_process.return_value = mock_process_instance
        
        monitor = ResourceMonitor(
            max_cpu_percent=50.0,
            max_memory_mb=100.0,
            max_file_size_mb=10.0,
            max_open_files=5,
            timeout_seconds=5
        )
        monitor.process = mock_process_instance
        
        # Ressourcenprüfung durchführen
        monitor._check_resource_usage()
        
        self.assertEqual(monitor.violation, "MEMORY_LIMIT_EXCEEDED")

    @patch.object(time, 'time')
    def test_check_resource_usage_timeout_exceeded(self, mock_time):
        """Test der Zeitlimitüberwachung."""
        # Mock-Zeit konfigurieren
        mock_time.side_effect = [0, 10]  # Startet bei 0, nach Aufruf 10 (10 Sekunden vergangen)
        
        monitor = ResourceMonitor(
            max_cpu_percent=50.0,
            max_memory_mb=100.0,
            max_file_size_mb=10.0,
            max_open_files=5,
            timeout_seconds=5
        )
        monitor.start_time = 0
        
        # Ressourcenprüfung durchführen
        monitor._check_resource_usage()
        
        self.assertEqual(monitor.violation, "TIMEOUT_EXCEEDED")


class TestSandboxResourceLimits(unittest.TestCase):
    """Tests für die Ressourcenlimits der Sandbox."""
    
    def setUp(self):
        """Testumgebung vorbereiten."""
        self.sandbox = MCodeSecuritySandbox(
            security_level=SecurityLevel.HIGH,
            max_cpu_usage=0.5,  # 50%
            max_memory_usage=50 * 1024 * 1024,  # 50 MB
            max_file_size=5 * 1024 * 1024,  # 5 MB
            max_open_files=3,
            max_execution_time=2  # 2 Sekunden
        )
    
    def test_cpu_limit_exceeded(self):
        """Test der CPU-Limitüberschreitung."""
        # Code, der viel CPU verbraucht
        cpu_intensive_code = """
import numpy as np

# CPU-intensive Operation
result = 0
for i in range(10000000):
    result += i * i
    if i % 1000 == 0:
        # Sicherstellen, dass die CPU belastet wird
        np.random.rand(1000, 1000)
"""
        
        # Code ausführen und Ergebnis prüfen
        result = self.sandbox.execute_code(cpu_intensive_code)
        
        self.assertFalse(result["success"])
        self.assertIn("Ressourcenlimit überschritten", result["error"])
        self.assertIn("CPU", result["error"].lower())
    
    def test_memory_limit_exceeded(self):
        """Test der Speicher-Limitüberschreitung."""
        # Code, der viel Speicher verbraucht
        memory_intensive_code = """
import numpy as np

# Speicherintensive Operation - versuche große Arrays zu erstellen
# 50 MB ist zu wenig für mehrere dieser Arrays
arrays = []
for i in range(10):
    # Jeder Array ist etwa 8 MB groß (1000x1000 Float64)
    arrays.append(np.random.rand(1000, 1000))
"""
        
        # Code ausführen und Ergebnis prüfen
        result = self.sandbox.execute_code(memory_intensive_code)
        
        self.assertFalse(result["success"])
        self.assertIn("Ressourcenlimit überschritten", result["error"])
        
    def test_timeout_exceeded(self):
        """Test der Zeitlimitüberschreitung."""
        # Code, der lange läuft
        long_running_code = """
import time

# Schleife, die länger als das Zeitlimit läuft
for i in range(10):
    time.sleep(1)  # 10 Sekunden insgesamt, über dem Limit von 2 Sekunden
"""
        
        # Code ausführen und Ergebnis prüfen
        result = self.sandbox.execute_code(long_running_code)
        
        self.assertFalse(result["success"])
        self.assertIn("Ressourcenlimit überschritten", result["error"])
        self.assertIn("zeit", result["error"].lower())
    
    def test_safe_execution_within_limits(self):
        """Test der sicheren Ausführung innerhalb der Limits."""
        # Code, der innerhalb der Limits bleibt
        safe_code = """
# Einfache Berechnung
result = sum(range(1000))
"""
        
        # Code ausführen und Ergebnis prüfen
        result = self.sandbox.execute_code(safe_code)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["result"]["result"], sum(range(1000)))
        
        # Ressourcennutzung prüfen
        self.assertIn("resource_usage", result)
        self.assertIn("cpu_percent", result["resource_usage"])
        self.assertIn("memory_mb", result["resource_usage"])
        self.assertIn("execution_time", result["resource_usage"])
    
    def test_execute_function_resource_monitoring(self):
        """Test der Funktionsausführung mit Ressourcenüberwachung."""
        
        # Funktion, die innerhalb der Limits bleibt
        def safe_function():
            return sum(range(1000))
        
        # Funktion ausführen und Ergebnis prüfen
        result = self.sandbox.execute_function(safe_function)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], sum(range(1000)))
        
        # Ressourcennutzung prüfen
        self.assertIn("resource_usage", result)
        
        # Funktion, die das Zeitlimit überschreitet
        def timeout_function():
            time.sleep(5)  # Überschreitet das Limit von 2 Sekunden
            return 42
        
        # Funktion ausführen und Ergebnis prüfen
        result = self.sandbox.execute_function(timeout_function)
        
        self.assertFalse(result["success"])
        self.assertIn("Ressourcenlimit überschritten", result["error"])
    
    def test_blacklisted_modules_rejected(self):
        """Test der Ablehnung von Blacklist-Modulen."""
        # Code, der ein Blacklist-Modul importiert
        unsafe_import_code = """
import os

# Versuche, eine Systemoperation durchzuführen
os.system('echo "Test"')
"""
        
        # Code analysieren, sollte abgelehnt werden
        can_execute, warnings = self.sandbox.analyze_code(unsafe_import_code)
        
        self.assertFalse(can_execute)
        self.assertTrue(any("verboten" in warning for warning in warnings))
        
        # Code ausführen, sollte fehlschlagen
        result = self.sandbox.execute_code(unsafe_import_code)
        
        self.assertFalse(result["success"])
        self.assertIn("verweigert", result["error"])


if __name__ == "__main__":
    unittest.main()
