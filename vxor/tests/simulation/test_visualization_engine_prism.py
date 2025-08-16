#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Tests für VisualizationEngine mit PRISM-Integration

Testfälle für die Integration der VisualizationEngine mit der PRISM-Engine.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch, call

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.tests.simulation.test_visualization_engine_prism")

# Füge Projektverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importiere zu testende Module
from miso.simulation.visualization_engine import VisualizationEngine, VisualizationType, VisualizationConfig
from miso.simulation.prism_engine import PrismEngine


class TestVisualizationEnginePrismIntegration(unittest.TestCase):
    """Tests für die Integration der VisualizationEngine mit der PRISM-Engine"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        # Mock-PRISM-Engine erstellen
        self.mock_prism_engine = MagicMock(spec=PrismEngine)
        
        # Konfiguriere Mock-Verhalten
        self.mock_prism_engine.register_visualization_callback = MagicMock()
        
        self.mock_prism_engine.time_scope = MagicMock()
        self.mock_prism_engine.time_scope.get_timeline_data = MagicMock()
        self.mock_prism_engine.time_scope.get_timeline_data.return_value = {
            "timeline_id": "test_timeline",
            "nodes": [
                {"id": "node1", "timestamp": time.time(), "data": {"value": 1}},
                {"id": "node2", "timestamp": time.time() + 10, "data": {"value": 2}},
                {"id": "node3", "timestamp": time.time() + 20, "data": {"value": 3}}
            ],
            "edges": [
                {"source": "node1", "target": "node2", "probability": 0.8},
                {"source": "node2", "target": "node3", "probability": 0.7}
            ],
            "metadata": {
                "probability": 0.9,
                "stability": 0.85,
                "paradox_risk": 0.1
            }
        }
        
        self.mock_prism_engine.matrix = MagicMock()
        self.mock_prism_engine.matrix.get_visualization_data = MagicMock()
        self.mock_prism_engine.matrix.get_visualization_data.return_value = {
            "dimensions": 3,
            "size": [5, 5, 5],
            "data": np.random.random((5, 5, 5)).tolist(),
            "coordinates": [
                {"x": 0, "y": 0, "z": 0, "value": 0.8, "label": "Point 1"},
                {"x": 1, "y": 1, "z": 1, "value": 0.7, "label": "Point 2"},
                {"x": 2, "y": 2, "z": 2, "value": 0.6, "label": "Point 3"}
            ]
        }
        
        self.mock_prism_engine.get_probability_metrics_for_timeline = MagicMock()
        self.mock_prism_engine.get_probability_metrics_for_timeline.return_value = {
            "overall_probability": 0.85,
            "stability_index": 0.9,
            "entropy": 0.2,
            "divergence_risk": 0.15,
            "paradox_potential": 0.05
        }
        
        # VisualizationEngine mit Mock-PRISM-Engine erstellen
        with patch('matplotlib.pyplot.figure'), patch('matplotlib.pyplot.show'):
            self.vis_engine = VisualizationEngine(prism_engine=self.mock_prism_engine)
        
        # VisualizationEngine ohne PRISM-Engine für Vergleichstests
        with patch('matplotlib.pyplot.figure'), patch('matplotlib.pyplot.show'):
            self.vis_engine_no_prism = VisualizationEngine()
        
        # Testdaten
        self.test_timeline_id = "test_timeline"
    
    def test_initialization_with_prism_engine(self):
        """Test der Initialisierung mit PRISM-Engine"""
        self.assertIsNotNone(self.vis_engine.prism_engine)
        self.assertEqual(self.vis_engine.prism_engine, self.mock_prism_engine)
        
        # Überprüfe, ob die PRISM-Visualisierungstypen initialisiert wurden
        self.assertTrue(hasattr(self.vis_engine, 'prism_visualization_types'))
        self.assertIsInstance(self.vis_engine.prism_visualization_types, dict)
        
        # Überprüfe, ob der Callback registriert wurde
        self.mock_prism_engine.register_visualization_callback.assert_called_once()
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.show')
    def test_visualize_prism_matrix(self, mock_show, mock_figure):
        """Test der PRISM-Matrix-Visualisierung"""
        # Teste mit PRISM-Engine
        vis_id = self.vis_engine.visualize_prism_matrix()
        
        # Überprüfe, ob die PRISM-Engine-Methode aufgerufen wurde
        self.mock_prism_engine.matrix.get_visualization_data.assert_called_once()
        
        # Überprüfe, ob eine Visualisierungs-ID zurückgegeben wurde
        self.assertIsNotNone(vis_id)
        
        # Überprüfe, ob die Visualisierung in den aktiven Visualisierungen gespeichert wurde
        self.assertIn(vis_id, self.vis_engine.active_visualizations)
        
        # Überprüfe, ob der Visualisierungstyp korrekt ist
        self.assertEqual(
            self.vis_engine.active_visualizations[vis_id]['config'].type,
            VisualizationType.MATRIX_3D
        )
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.show')
    def test_visualize_timeline_with_prism(self, mock_show, mock_figure):
        """Test der Zeitlinienvisualisierung mit PRISM-Integration"""
        # Teste mit PRISM-Engine
        vis_id = self.vis_engine.visualize_timeline_with_prism(self.test_timeline_id)
        
        # Überprüfe, ob die PRISM-Engine-Methoden aufgerufen wurden
        self.mock_prism_engine.time_scope.get_timeline_data.assert_called_with(self.test_timeline_id)
        self.mock_prism_engine.get_probability_metrics_for_timeline.assert_called_with(self.test_timeline_id)
        
        # Überprüfe, ob eine Visualisierungs-ID zurückgegeben wurde
        self.assertIsNotNone(vis_id)
        
        # Überprüfe, ob die Visualisierung in den aktiven Visualisierungen gespeichert wurde
        self.assertIn(vis_id, self.vis_engine.active_visualizations)
        
        # Überprüfe, ob der Visualisierungstyp korrekt ist
        self.assertEqual(
            self.vis_engine.active_visualizations[vis_id]['config'].type,
            VisualizationType.TIMELINE_3D
        )
        
        # Teste ohne PRISM-Engine (Fallback)
        vis_id_no_prism = self.vis_engine_no_prism.visualize_timeline_with_prism(self.test_timeline_id)
        
        # Überprüfe, ob eine Visualisierungs-ID zurückgegeben wurde
        self.assertIsNotNone(vis_id_no_prism)
        
        # Überprüfe, ob die Visualisierung in den aktiven Visualisierungen gespeichert wurde
        self.assertIn(vis_id_no_prism, self.vis_engine_no_prism.active_visualizations)
        
        # Überprüfe, ob der Visualisierungstyp korrekt ist (Fallback auf 2D)
        self.assertEqual(
            self.vis_engine_no_prism.active_visualizations[vis_id_no_prism]['config'].type,
            VisualizationType.TIMELINE_2D
        )
    
    def test_handle_prism_event(self):
        """Test der Verarbeitung von PRISM-Engine-Ereignissen"""
        # Füge einen Visualisierungstyp hinzu
        self.vis_engine.update_queues['PRISM_MATRIX'] = MagicMock()
        
        # Simuliere ein PRISM-Engine-Ereignis
        test_event_data = {"test": "data"}
        self.vis_engine._handle_prism_event('PRISM_MATRIX', test_event_data)
        
        # Überprüfe, ob die Ereignisdaten zur Warteschlange hinzugefügt wurden
        self.vis_engine.update_queues['PRISM_MATRIX'].put.assert_called_with(test_event_data)
    
    def test_get_probability_metrics_for_timeline(self):
        """Test des Abrufs von Wahrscheinlichkeitsmetriken für eine Zeitlinie"""
        # Teste mit PRISM-Engine
        metrics = self.vis_engine._get_probability_metrics_for_timeline(self.test_timeline_id)
        
        # Überprüfe, ob die PRISM-Engine-Methode aufgerufen wurde
        self.mock_prism_engine.get_probability_metrics_for_timeline.assert_called_with(self.test_timeline_id)
        
        # Überprüfe, ob die Metriken korrekt zurückgegeben wurden
        self.assertEqual(metrics['overall_probability'], 0.85)
        self.assertEqual(metrics['stability_index'], 0.9)
        
        # Teste ohne PRISM-Engine
        metrics_no_prism = self.vis_engine_no_prism._get_probability_metrics_for_timeline(self.test_timeline_id)
        
        # Überprüfe, ob ein leeres Dictionary zurückgegeben wurde
        self.assertEqual(metrics_no_prism, {})
    
    def test_exception_handling(self):
        """Test der Ausnahmebehandlung bei PRISM-Engine-Fehlern"""
        # Konfiguriere Mock, um eine Ausnahme auszulösen
        self.mock_prism_engine.time_scope.get_timeline_data.side_effect = Exception("Test-Ausnahme")
        
        # Visualisierung sollte trotz Ausnahme erstellt werden (Fallback)
        with patch('matplotlib.pyplot.figure'), patch('matplotlib.pyplot.show'):
            vis_id = self.vis_engine.visualize_timeline_with_prism(self.test_timeline_id)
        
        # Überprüfe, ob eine Visualisierungs-ID zurückgegeben wurde
        self.assertIsNotNone(vis_id)
        
        # Überprüfe, ob die Visualisierung in den aktiven Visualisierungen gespeichert wurde
        self.assertIn(vis_id, self.vis_engine.active_visualizations)
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        # Schließe alle Visualisierungen
        with patch('matplotlib.pyplot.close'):
            if hasattr(self.vis_engine, 'close_all_visualizations'):
                self.vis_engine.close_all_visualizations()
            if hasattr(self.vis_engine_no_prism, 'close_all_visualizations'):
                self.vis_engine_no_prism.close_all_visualizations()
        
        self.vis_engine = None
        self.vis_engine_no_prism = None
        self.mock_prism_engine = None


if __name__ == "__main__":
    unittest.main()
