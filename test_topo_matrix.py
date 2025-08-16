#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test für das TopoNet-Modul der MPRIME Engine.
"""

import unittest
import sys
import os
import numpy as np
from typing import Dict, Any, List

# Füge das Hauptverzeichnis zum Pfad hinzu, um die MISO-Module zu importieren
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from miso.math.mprime.topo_matrix import TopoNet

class TestTopoNet(unittest.TestCase):
    """
    Testklasse für das TopoNet-Modul
    """
    
    def setUp(self):
        """
        Einrichtung für die Tests
        """
        self.topo_net = TopoNet()
    
    def test_initialization(self):
        """
        Testet die korrekte Initialisierung
        """
        self.assertIsNotNone(self.topo_net)
        self.assertEqual(self.topo_net.default_dimensions, 3)
        self.assertEqual(self.topo_net.max_dimensions, 10)
        self.assertIsNone(self.topo_net.matrix)
    
    def test_create_matrix(self):
        """
        Testet die Erstellung einer topologischen Matrix
        """
        dimensions = 3
        size = 5
        
        # Erstelle eine Matrix
        result = self.topo_net.create_matrix(dimensions, size)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Matrix korrekt erstellt wurde
        self.assertEqual(result["dimensions"], dimensions)
        self.assertEqual(result["size"], size)
        self.assertIsInstance(result["matrix"], np.ndarray)
        
        # Überprüfe die Form der Matrix
        expected_shape = tuple([size] * dimensions)
        self.assertEqual(result["matrix"].shape, expected_shape)
    
    def test_add_node(self):
        """
        Testet das Hinzufügen eines Knotens zur Matrix
        """
        # Erstelle eine Matrix
        self.topo_net.create_matrix(3, 5)
        
        # Füge einen Knoten hinzu
        node_id = "node1"
        coordinates = [1, 2, 3]
        value = 42
        
        result = self.topo_net.add_node(node_id, coordinates, value)
        
        # Überprüfe, ob das Ergebnis True ist
        self.assertTrue(result)
        
        # Überprüfe, ob der Knoten korrekt hinzugefügt wurde
        self.assertIn(node_id, self.topo_net.nodes)
        self.assertEqual(self.topo_net.nodes[node_id]["coordinates"], coordinates)
        self.assertEqual(self.topo_net.nodes[node_id]["value"], value)
        
        # Überprüfe, ob der Wert in der Matrix korrekt gesetzt wurde
        self.assertEqual(self.topo_net.matrix["matrix"][tuple(coordinates)], value)
    
    def test_add_edge(self):
        """
        Testet das Hinzufügen einer Kante zwischen zwei Knoten
        """
        # Erstelle eine Matrix
        self.topo_net.create_matrix(3, 5)
        
        # Füge zwei Knoten hinzu
        node1_id = "node1"
        node1_coords = [1, 2, 3]
        node1_value = 42
        
        node2_id = "node2"
        node2_coords = [3, 4, 1]
        node2_value = 24
        
        self.topo_net.add_node(node1_id, node1_coords, node1_value)
        self.topo_net.add_node(node2_id, node2_coords, node2_value)
        
        # Füge eine Kante hinzu
        edge_id = "edge1"
        weight = 0.75
        
        result = self.topo_net.add_edge(edge_id, node1_id, node2_id, weight)
        
        # Überprüfe, ob das Ergebnis True ist
        self.assertTrue(result)
        
        # Überprüfe, ob die Kante korrekt hinzugefügt wurde
        self.assertIn(edge_id, self.topo_net.edges)
        self.assertEqual(self.topo_net.edges[edge_id]["source"], node1_id)
        self.assertEqual(self.topo_net.edges[edge_id]["target"], node2_id)
        self.assertEqual(self.topo_net.edges[edge_id]["weight"], weight)
    
    def test_apply_transformation(self):
        """
        Testet die Anwendung einer Transformation auf die Matrix
        """
        # Erstelle eine Matrix
        self.topo_net.create_matrix(3, 5)
        
        # Füge einen Knoten hinzu
        node_id = "node1"
        coordinates = [1, 2, 3]
        value = 42
        
        self.topo_net.add_node(node_id, coordinates, value)
        
        # Definiere eine Transformationsmatrix (Rotation)
        transformation = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        
        # Wende die Transformation an
        result = self.topo_net.apply_transformation(transformation)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Transformation korrekt angewendet wurde
        # Die neuen Koordinaten sollten [2, -1, 3] sein (gerundet auf Ganzzahlen)
        new_coords = self.topo_net.nodes[node_id]["coordinates"]
        expected_coords = [2, -1, 3]
        
        # Überprüfe, ob die Koordinaten ungefähr gleich sind (Rundungsfehler berücksichtigen)
        for i in range(len(new_coords)):
            self.assertAlmostEqual(new_coords[i], expected_coords[i], delta=0.001)
    
    def test_dimensional_bend(self):
        """
        Testet die Dimensionsbeugung
        """
        # Erstelle eine 3D-Matrix
        self.topo_net.create_matrix(3, 5)
        
        # Füge einen Knoten hinzu
        node_id = "node1"
        coordinates = [1, 2, 3]
        value = 42
        
        self.topo_net.add_node(node_id, coordinates, value)
        
        # Führe eine Dimensionsbeugung durch (auf 4 Dimensionen)
        target_dimensions = 4
        result = self.topo_net.dimensional_bend(target_dimensions)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Dimensionsbeugung korrekt durchgeführt wurde
        self.assertEqual(result["dimensions"], target_dimensions)
        self.assertEqual(result["size"], 5)
        self.assertIsInstance(result["matrix"], np.ndarray)
        
        # Überprüfe die Form der Matrix
        expected_shape = (5, 5, 5, 5)
        self.assertEqual(result["matrix"].shape, expected_shape)
        
        # Überprüfe, ob der Knoten korrekt transformiert wurde
        new_coords = self.topo_net.nodes[node_id]["coordinates"]
        self.assertEqual(len(new_coords), target_dimensions)
        
        # Die ersten 3 Koordinaten sollten gleich bleiben
        for i in range(3):
            self.assertEqual(new_coords[i], coordinates[i])
    
    def test_compute_path(self):
        """
        Testet die Berechnung eines Pfades zwischen zwei Knoten
        """
        # Erstelle eine Matrix
        self.topo_net.create_matrix(3, 5)
        
        # Füge Knoten hinzu
        self.topo_net.add_node("A", [0, 0, 0], 1)
        self.topo_net.add_node("B", [0, 1, 0], 2)
        self.topo_net.add_node("C", [1, 1, 0], 3)
        self.topo_net.add_node("D", [1, 0, 0], 4)
        
        # Füge Kanten hinzu
        self.topo_net.add_edge("AB", "A", "B", 1)
        self.topo_net.add_edge("BC", "B", "C", 1)
        self.topo_net.add_edge("CD", "C", "D", 1)
        self.topo_net.add_edge("DA", "D", "A", 1)
        
        # Berechne den Pfad von A nach C
        result = self.topo_net.compute_path("A", "C")
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob der Pfad korrekt berechnet wurde
        self.assertEqual(result["source"], "A")
        self.assertEqual(result["target"], "C")
        self.assertIsInstance(result["path"], list)
        
        # Der Pfad sollte A -> B -> C sein
        expected_path = ["A", "B", "C"]
        self.assertEqual(result["path"], expected_path)
    
    def test_export_to_json(self):
        """
        Testet den Export der Matrix als JSON
        """
        # Erstelle eine Matrix
        self.topo_net.create_matrix(2, 3)
        
        # Füge Knoten hinzu
        self.topo_net.add_node("A", [0, 0], 1)
        self.topo_net.add_node("B", [0, 1], 2)
        self.topo_net.add_node("C", [1, 1], 3)
        
        # Füge Kanten hinzu
        self.topo_net.add_edge("AB", "A", "B", 1)
        self.topo_net.add_edge("BC", "B", "C", 1)
        
        # Exportiere als JSON
        result = self.topo_net.export_to_json()
        
        # Überprüfe, ob das Ergebnis ein String ist
        self.assertIsInstance(result, str)
        
        # Überprüfe, ob der JSON-String die erwarteten Elemente enthält
        self.assertIn('"dimensions": 2', result)
        self.assertIn('"size": 3', result)
        self.assertIn('"nodes":', result)
        self.assertIn('"edges":', result)
        self.assertIn('"A":', result)
        self.assertIn('"B":', result)
        self.assertIn('"C":', result)
        self.assertIn('"AB":', result)
        self.assertIn('"BC":', result)

if __name__ == '__main__':
    unittest.main()
