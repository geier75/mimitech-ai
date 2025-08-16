#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_energy_efficiency_manager.py
================================

Unit-Tests für die EnergyEfficiencyManager-Klasse zur Überwachung und Optimierung
des Energieverbrauchs bei föderiertem Training.

Teil der MISO Ultimate AGI - Phase 6 (Federated Learning System)
"""

import os
import sys
import json
import time
import uuid
import unittest
from unittest import mock
import tempfile
import shutil
from pathlib import Path

# Ergänze Suchpfad für Importe aus dem Hauptprojekt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from miso.federated_learning.EnergyEfficiencyManager import EnergyEfficiencyManager
except ImportError:
    print("Konnte EnergyEfficiencyManager nicht importieren. Prüfe, ob der Pfad korrekt ist.")
    raise

class TestEnergyEfficiencyManager(unittest.TestCase):
    """Test-Suite für den EnergyEfficiencyManager."""
    
    def setUp(self):
        """Testumgebung für jeden Test einrichten."""
        # Erstelle temporäres Verzeichnis für Test-Output
        self.test_dir = tempfile.mkdtemp()
        
        # Erstelle den Manager mit deaktiviertem Monitoring für deterministische Tests
        self.energy_manager = EnergyEfficiencyManager(
            base_output_dir=self.test_dir,
            enable_monitoring=False  # Deaktiviere Monitoring für Tests
        )
    
    def tearDown(self):
        """Aufräumen nach jedem Test."""
        # Bereinige temporäres Verzeichnis
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Testet die korrekte Initialisierung des EnergyEfficiencyManager."""
        self.assertIsNotNone(self.energy_manager)
        self.assertIsNotNone(self.energy_manager.session_id)
        self.assertEqual(self.energy_manager.base_output_dir, self.test_dir)
        self.assertFalse(self.energy_manager.is_monitoring)
        
        # Prüfe, ob die Verzeichnisstruktur korrekt erstellt wurde
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'energy_logs')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'energy_stats')))
    
    def test_system_detection(self):
        """Testet die Systemerkennung."""
        system_info = self.energy_manager.system_info
        self.assertIsNotNone(system_info)
        self.assertIn('os', system_info)
        self.assertIn('architecture', system_info)
        self.assertIn('cpu_count', system_info)
        
        # Stichprobenartige Prüfung der Werte
        self.assertGreater(system_info['cpu_count'], 0)
        self.assertIn(system_info['os'], ['Darwin', 'Linux', 'Windows'])
    
    @mock.patch('miso.federated_learning.EnergyEfficiencyManager.subprocess.run')
    @mock.patch('miso.federated_learning.EnergyEfficiencyManager.psutil')
    def test_collect_current_stats(self, mock_psutil, mock_subprocess):
        """Testet die Sammlung von Systemstatistiken mit Mock-Daten."""
        # Mock für psutil.cpu_percent
        mock_psutil.cpu_percent.return_value = 35.0  # 35% CPU-Auslastung
        
        # Mock für psutil.virtual_memory
        mock_memory = mock.MagicMock()
        mock_memory.percent = 45.0  # 45% Speichernutzung
        mock_psutil.virtual_memory.return_value = mock_memory
        
        # Mock für psutil.sensors_battery
        mock_battery = mock.MagicMock()
        mock_battery.percent = 65.0  # 65% Batteriestand
        mock_psutil.sensors_battery.return_value = mock_battery
        
        # Test ausführen
        monitoring_tools = {"available": ["psutil"], "psutil": mock_psutil}
        stats = self.energy_manager._collect_current_stats(monitoring_tools)
        
        # Ergebnisse prüfen
        self.assertEqual(stats["cpu_usage"], 0.35)  # 35% -> 0.35
        self.assertEqual(stats["memory_usage"], 0.45)  # 45% -> 0.45
        self.assertEqual(stats["battery_level"], 0.65)  # 65% -> 0.65
    
    def test_optimization_recommendation(self):
        """Testet die Empfehlungen zur Backend-Optimierung."""
        # Test mit Apple Silicon (simuliert)
        with mock.patch.dict(self.energy_manager.system_info, 
                           {'is_apple_silicon': True, 'ml_frameworks': ['mlx']}):
            backend = self.energy_manager.optimize_backend_selection()
            self.assertEqual(backend, "mlx")
        
        # Test mit CUDA-GPUs (simuliert)
        with mock.patch.dict(self.energy_manager.system_info, 
                           {'is_apple_silicon': False, 'has_cuda': True}):
            backend = self.energy_manager.optimize_backend_selection()
            self.assertEqual(backend, "pytorch_cuda")
        
        # Test mit CPU-Fallback
        with mock.patch.dict(self.energy_manager.system_info, 
                           {'is_apple_silicon': False, 'has_cuda': False}):
            backend = self.energy_manager.optimize_backend_selection()
            self.assertEqual(backend, "pytorch_cpu")
    
    def test_training_schedule_recommendation(self):
        """Testet die Empfehlungen für den Trainingszeitplan."""
        # Mock für aktuelle Systemstatistiken
        with mock.patch.object(self.energy_manager, '_collect_current_stats') as mock_stats:
            mock_stats.return_value = {
                "cpu_usage": 0.3,  # 30%
                "gpu_usage": 0.2,  # 20%
                "memory_usage": 0.4,  # 40%
                "temperature": 0.5,  # 50%
                "power_consumption": 0.3,  # 30%
                "battery_level": 0.8  # 80%
            }
            
            # Test für kleine Aufgabe
            recommendation_small = self.energy_manager.recommend_training_schedule(task_size="small")
            self.assertIn("batch_size", recommendation_small)
            self.assertIn("best_time", recommendation_small)
            
            # Prüfe, dass Batch-Größe der Aufgabengröße entspricht
            self.assertEqual(recommendation_small["batch_size"], 8)
            
            # Test für Aufgaben mit hoher Priorität
            recommendation_high = self.energy_manager.recommend_training_schedule(
                task_size="medium", priority="high")
            self.assertEqual(recommendation_high["best_time"], "now")
            
            # Test für niedrigen Batteriestand
            mock_stats.return_value["battery_level"] = 0.2  # 20%
            recommendation_low_battery = self.energy_manager.recommend_training_schedule()
            # Batch-Größe sollte reduziert sein
            self.assertLess(recommendation_low_battery["batch_size"], 16)
    
    def test_learning_rate_adjustment(self):
        """Testet die Anpassung der Lernrate basierend auf Energiebeschränkungen."""
        base_lr = 0.001
        
        # Normaler Betrieb - keine Anpassung
        self.energy_manager.energy_stats["battery_level"] = [0.9]  # 90%
        self.energy_manager.energy_stats["temperature"] = [0.5]  # 50%
        self.energy_manager.is_monitoring = True
        
        adjusted_lr = self.energy_manager.adjust_learning_rate(base_lr)
        self.assertEqual(adjusted_lr, base_lr)  # Keine Änderung
        
        # Niedriger Batteriestand - reduzierte Lernrate
        self.energy_manager.energy_stats["battery_level"] = [0.2]  # 20%
        adjusted_lr_low_battery = self.energy_manager.adjust_learning_rate(base_lr)
        self.assertLess(adjusted_lr_low_battery, base_lr)  # Reduzierte Lernrate
        
        # Hohe Temperatur - reduzierte Lernrate
        self.energy_manager.energy_stats["battery_level"] = [0.9]  # 90%
        self.energy_manager.energy_stats["temperature"] = [0.9]  # 90%
        adjusted_lr_high_temp = self.energy_manager.adjust_learning_rate(base_lr)
        self.assertLess(adjusted_lr_high_temp, base_lr)  # Reduzierte Lernrate
    
    def test_should_defer_training(self):
        """Testet die Entscheidung, Training aufgrund von Energiebeschränkungen zu verschieben."""
        # Monitoring und Optimierungen aktivieren
        self.energy_manager.is_monitoring = True
        self.energy_manager.enable_optimizations = True
        
        # Normale Bedingungen - Training sollte fortgesetzt werden
        self.energy_manager.energy_stats["battery_level"] = [0.7]  # 70%
        self.energy_manager.energy_stats["temperature"] = [0.6]  # 60%
        defer, reason = self.energy_manager.should_defer_training()
        self.assertFalse(defer)
        
        # Kritisch niedriger Batteriestand - Training sollte verschoben werden
        self.energy_manager.energy_stats["battery_level"] = [0.1]  # 10%
        defer_low_battery, reason_low_battery = self.energy_manager.should_defer_training()
        self.assertTrue(defer_low_battery)
        self.assertIn("Batteriestand", reason_low_battery)
        
        # Hohe Temperatur - Training sollte verschoben werden
        self.energy_manager.energy_stats["battery_level"] = [0.7]  # 70%
        self.energy_manager.energy_stats["temperature"] = [0.95]  # 95%
        defer_high_temp, reason_high_temp = self.energy_manager.should_defer_training()
        self.assertTrue(defer_high_temp)
        self.assertIn("Temperatur", reason_high_temp)
    
    def test_device_constraints(self):
        """Testet die Erkennung von Gerätebeschränkungen."""
        # Optimale Bedingungen - keine Beschränkungen
        with mock.patch('miso.federated_learning.EnergyEfficiencyManager.psutil') as mock_psutil:
            # Mock für Speichernutzung
            mock_memory = mock.MagicMock()
            mock_memory.available = 8 * 1024 * 1024 * 1024  # 8 GB verfügbar
            mock_memory.total = 16 * 1024 * 1024 * 1024  # 16 GB Gesamtspeicher
            mock_psutil.virtual_memory.return_value = mock_memory
            
            # Mock für Batteriestand
            mock_battery = mock.MagicMock()
            mock_battery.percent = 90  # 90% Batteriestand
            mock_psutil.sensors_battery.return_value = mock_battery
            
            # Energie-Statistiken mit optimalen Bedingungen
            self.energy_manager.energy_stats["cpu_usage"] = [0.3]
            self.energy_manager.energy_stats["temperature"] = [0.4]
            self.energy_manager.is_monitoring = True
            
            constraints = self.energy_manager.get_device_constraints()
            
            self.assertFalse(constraints["memory_limited"])
            self.assertFalse(constraints["battery_limited"])
            self.assertFalse(constraints["thermal_limited"])
            self.assertFalse(constraints["compute_limited"])
        
        # Beschränkte Bedingungen
        with mock.patch('miso.federated_learning.EnergyEfficiencyManager.psutil') as mock_psutil:
            # Mock für Speichernutzung - wenig verfügbar
            mock_memory = mock.MagicMock()
            mock_memory.available = 2 * 1024 * 1024 * 1024  # 2 GB verfügbar
            mock_memory.total = 16 * 1024 * 1024 * 1024  # 16 GB Gesamtspeicher
            mock_psutil.virtual_memory.return_value = mock_memory
            
            # Mock für Batteriestand - niedrig
            mock_battery = mock.MagicMock()
            mock_battery.percent = 15  # 15% Batteriestand
            mock_psutil.sensors_battery.return_value = mock_battery
            
            # Energie-Statistiken mit hoher Auslastung
            self.energy_manager.energy_stats["cpu_usage"] = [0.9]
            self.energy_manager.energy_stats["temperature"] = [0.8]
            self.energy_manager.is_monitoring = True
            
            constraints = self.energy_manager.get_device_constraints()
            
            self.assertTrue(constraints["memory_limited"])
            self.assertTrue(constraints["battery_limited"])
            self.assertTrue(constraints["thermal_limited"])
            self.assertTrue(constraints["compute_limited"])
    
    def test_model_parameter_adjustment(self):
        """Testet die Anpassung von Modellparametern basierend auf Gerätebeschränkungen."""
        # Original-Parameter
        model_params = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "precision": "float32"
        }
        
        # Test ohne Beschränkungen
        with mock.patch.object(self.energy_manager, 'get_device_constraints') as mock_constraints:
            mock_constraints.return_value = {
                "memory_limited": False,
                "battery_limited": False,
                "thermal_limited": False,
                "compute_limited": False
            }
            
            # Nur Backend-Empfehlung sollte hinzukommen
            adjusted = self.energy_manager.check_and_apply_device_constraints(model_params)
            self.assertEqual(adjusted["batch_size"], 32)
            self.assertEqual(adjusted["learning_rate"], 0.001)
            self.assertEqual(adjusted["precision"], "float32")
            self.assertIn("recommended_backend", adjusted)
        
        # Test mit Speicher- und Batteriebeschränkungen
        with mock.patch.object(self.energy_manager, 'get_device_constraints') as mock_constraints:
            mock_constraints.return_value = {
                "memory_limited": True,
                "battery_limited": True,
                "thermal_limited": False,
                "compute_limited": False
            }
            
            adjusted = self.energy_manager.check_and_apply_device_constraints(model_params)
            
            # Batch-Größe sollte reduziert sein
            self.assertLess(adjusted["batch_size"], model_params["batch_size"])
            # Lernrate sollte reduziert sein
            self.assertLess(adjusted["learning_rate"], model_params["learning_rate"])
            # Präzision sollte geringer sein
            self.assertEqual(adjusted["precision"], "float16")
            
            # Prüfe auf Angabe der angewendeten Beschränkungen im Ergebnis
            self.assertTrue(adjusted["memory_limit_applied"])
            self.assertTrue(adjusted["battery_limit_applied"])
    
    def test_energy_usage_calculation(self):
        """Testet die Berechnung der Energienutzungsstatistiken."""
        # Simuliere gesammelte Daten für einen Zeitraum
        now = time.time()
        self.energy_manager.is_monitoring = True
        self.energy_manager.energy_stats = {
            "timestamp": [now - 10, now - 5, now],
            "cpu_usage": [0.3, 0.4, 0.5],
            "gpu_usage": [0.2, 0.3, 0.4],
            "power_consumption": [0.3, 0.4, 0.5],
            "temperature": [0.5, 0.6, 0.7],
            "battery_level": [0.8, 0.79, 0.78]  # Leichter Batterieverbrauch
        }
        
        # Berechne Energieverbrauch für den Zeitraum
        stats = self.energy_manager.calculate_energy_usage(now - 10, now)
        
        # Prüfe Ergebnisse
        self.assertEqual(stats["duration_seconds"], 10)
        self.assertAlmostEqual(stats["avg_cpu_usage"], 0.4, places=1)
        self.assertAlmostEqual(stats["avg_gpu_usage"], 0.3, places=1)
        self.assertAlmostEqual(stats["avg_power_consumption"], 0.4, places=1)
        self.assertAlmostEqual(stats["battery_change_percent"], -2.0, places=1)
        
        # Prüfe Energieschätzung in Joule
        self.assertTrue("estimated_energy_joules" in stats)
        # Max Power = 100W, avg_power = 0.4, duration = 10s -> 400J
        self.assertAlmostEqual(stats["estimated_energy_joules"], 400, delta=50)
    
    def test_integration_with_trainer(self):
        """Testet die Integration mit einem simulierten LocalSelfTrainer."""
        # Simulierter Trainer
        mock_trainer = mock.MagicMock()
        
        # Integriere mit dem Trainer
        self.energy_manager.integrate_with_local_trainer(mock_trainer)
        
        # Prüfe, ob die Referenz korrekt gesetzt wurde
        self.assertEqual(mock_trainer.energy_manager, self.energy_manager)
        
        # Prüfe, ob die Hilfsmethoden verfügbar sind
        self.assertTrue(callable(mock_trainer.get_energy_stats))
        self.assertTrue(callable(mock_trainer.check_energy_constraints))
        self.assertTrue(callable(mock_trainer.optimize_backend))
    
    def test_integration_with_optimizer(self):
        """Testet die Integration mit einem simulierten TinyModelOptimizer."""
        # Simulierter Optimizer
        mock_optimizer = mock.MagicMock()
        
        # Integriere mit dem Optimizer
        self.energy_manager.integrate_with_model_optimizer(mock_optimizer)
        
        # Prüfe, ob die Referenz korrekt gesetzt wurde
        self.assertEqual(mock_optimizer.energy_manager, self.energy_manager)
        
        # Prüfe, ob die Hilfsmethoden verfügbar sind
        self.assertTrue(callable(mock_optimizer.get_energy_efficient_config))
        self.assertTrue(callable(mock_optimizer.should_optimize_for_energy))

if __name__ == '__main__':
    unittest.main()
