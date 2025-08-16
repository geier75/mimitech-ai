#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Circular Import Detector

Dieses Tool analysiert den Python-Code im MISO-Ultimate-Projekt und identifiziert
zirkuläre Importabhängigkeiten, die zu Initialisierungsproblemen führen können.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
ZTM-Level: STRICT
"""

import os
import sys
import ast
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import json
import networkx as nx
from collections import defaultdict, deque

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("circularity_detector")

# ZTM-Protokoll-Initialisierung
ZTM_ACTIVE = os.environ.get('MISO_ZTM_MODE', '1') == '1'
ZTM_LOG_LEVEL = os.environ.get('MISO_ZTM_LOG_LEVEL', 'INFO')

def ztm_log(message: str, level: str = 'INFO', module: str = 'CIRCULARITY_DETECTOR'):
    """ZTM-konforme Logging-Funktion mit Audit-Trail"""
    if not ZTM_ACTIVE:
        return
    
    log_func = getattr(logger, level.lower())
    log_func(f"[ZTM:{module}] {message}")

class ImportVisitor(ast.NodeVisitor):
    """AST-Visitor zum Extrahieren von Importinformationen aus Python-Dateien"""
    
    def __init__(self):
        self.imports = set()
        self.from_imports = set()
        self.all_imports = set()
        
    def visit_Import(self, node):
        """Erfasst reguläre Import-Anweisungen (z.B. 'import foo')"""
        for name in node.names:
            self.imports.add(name.name)
            self.all_imports.add(name.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Erfasst From-Import-Anweisungen (z.B. 'from foo import bar')"""
        if node.module is not None:
            self.from_imports.add(node.module)
            self.all_imports.add(node.module)
        self.generic_visit(node)

class CircularImportDetector:
    """Detector für zirkuläre Importabhängigkeiten in MISO-Ultimate"""
    
    def __init__(self, root_dirs: List[str], excluded_dirs: List[str] = None):
        """
        Initialisiert den Circular Import Detector
        
        Args:
            root_dirs: Liste von Root-Verzeichnissen zur Analyse
            excluded_dirs: Liste von zu ignorierenden Verzeichnissen
        """
        self.root_dirs = [Path(d) for d in root_dirs]
        self.excluded_dirs = [Path(d) for d in excluded_dirs] if excluded_dirs else []
        self.module_map = {}  # Zuordnung von Modulnamen zu Dateipfaden
        self.dependency_graph = nx.DiGraph()  # Gerichteter Graph für Abhängigkeiten
        self.import_map = {}  # Zuordnung von Dateipfaden zu Imports
        
        if ZTM_ACTIVE:
            ztm_log(f"CircularImportDetector initialisiert mit {len(root_dirs)} Root-Verzeichnissen")
    
    def _should_skip_dir(self, dir_path: Path) -> bool:
        """Überprüft, ob ein Verzeichnis übersprungen werden soll"""
        for excluded in self.excluded_dirs:
            if str(dir_path).startswith(str(excluded)):
                return True
        
        # Standardverzeichnisse überspringen
        skip_dirs = ['.git', '__pycache__', 'venv', 'env', '.venv', '.env', '.ipynb_checkpoints']
        if dir_path.name in skip_dirs:
            return True
        
        return False
    
    def _is_python_file(self, file_path: Path) -> bool:
        """Überprüft, ob es sich um eine Python-Datei handelt"""
        return file_path.suffix == '.py'
    
    def _get_module_name(self, file_path: Path, root_dir: Path) -> str:
        """Erzeugt den Modulnamen aus dem Dateipfad relativ zum Root-Verzeichnis"""
        rel_path = file_path.relative_to(root_dir)
        parts = list(rel_path.parts)
        
        if parts[-1] == '__init__.py':
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]  # Entferne .py-Endung
            
        return '.'.join(parts)
    
    def _build_module_map(self):
        """Baut die Zuordnung von Modulnamen zu Dateipfaden auf"""
        for root_dir in self.root_dirs:
            ztm_log(f"Analysiere Verzeichnis: {root_dir}", level="INFO")
            
            for dirpath, dirnames, filenames in os.walk(root_dir):
                dir_path = Path(dirpath)
                
                # Überspringe ausgeschlossene Verzeichnisse
                if self._should_skip_dir(dir_path):
                    dirnames[:] = []  # Überspringe alle Unterverzeichnisse
                    continue
                
                # Überprüfe alle Python-Dateien im Verzeichnis
                for filename in filenames:
                    file_path = dir_path / filename
                    
                    if not self._is_python_file(file_path):
                        continue
                    
                    try:
                        module_name = self._get_module_name(file_path, root_dir)
                        self.module_map[module_name] = file_path
                        
                        # Wenn diese Datei ein __init__.py ist, füge auch das übergeordnete Paket hinzu
                        if filename == '__init__.py':
                            parent_module = module_name
                            self.module_map[parent_module] = file_path
                    except Exception as e:
                        ztm_log(f"Fehler beim Verarbeiten von {file_path}: {e}", level="ERROR")
        
        ztm_log(f"Modul-Map erstellt mit {len(self.module_map)} Einträgen", level="INFO")
    
    def _extract_imports(self, file_path: Path) -> Set[str]:
        """Extrahiert alle Imports aus einer Python-Datei"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            tree = ast.parse(content)
            visitor = ImportVisitor()
            visitor.visit(tree)
            
            # Rückgabe aller gefundenen Imports
            return visitor.all_imports
        except Exception as e:
            ztm_log(f"Fehler beim Extrahieren von Imports aus {file_path}: {e}", level="ERROR")
            return set()
    
    def _build_dependency_graph(self):
        """Baut den Abhängigkeitsgraphen basierend auf den Imports auf"""
        # Füge jeden Knoten zum Graphen hinzu
        for module_name in self.module_map:
            self.dependency_graph.add_node(module_name)
        
        # Extrahiere Imports für jedes Modul und füge Kanten hinzu
        for module_name, file_path in self.module_map.items():
            imports = self._extract_imports(file_path)
            self.import_map[module_name] = imports
            
            for imported_module in imports:
                # Überprüfe, ob der Import im Modul-Map ist
                if imported_module in self.module_map:
                    # Füge eine Kante vom aktuellen Modul zum importierten Modul hinzu
                    self.dependency_graph.add_edge(module_name, imported_module)
        
        ztm_log(f"Abhängigkeitsgraph erstellt mit {self.dependency_graph.number_of_nodes()} Knoten und {self.dependency_graph.number_of_edges()} Kanten", level="INFO")
    
    def detect_cycles(self) -> List[List[str]]:
        """
        Erkennt zirkuläre Abhängigkeiten im Abhängigkeitsgraphen
        
        Returns:
            Liste von erkannten Zyklen (jeder Zyklus ist eine Liste von Modulnamen)
        """
        try:
            ztm_log("Suche nach zirkulären Abhängigkeiten...", level="INFO")
            cycles = list(nx.simple_cycles(self.dependency_graph))
            ztm_log(f"{len(cycles)} zirkuläre Abhängigkeiten gefunden", level="INFO")
            return sorted(cycles, key=len)
        except Exception as e:
            ztm_log(f"Fehler bei der Erkennung von Zyklen: {e}", level="ERROR")
            return []
    
    def analyze(self) -> Dict:
        """
        Analysiert das Projekt auf zirkuläre Abhängigkeiten
        
        Returns:
            Dictionary mit Analyseergebnissen
        """
        ztm_log("Beginne Analyse...", level="INFO")
        
        # Baue Modul-Map und Abhängigkeitsgraph auf
        self._build_module_map()
        self._build_dependency_graph()
        
        # Erkenne Zyklen
        cycles = self.detect_cycles()
        
        # Erstelle Empfehlungen
        recommendations = []
        for cycle in cycles:
            cycle_info = {
                "cycle": cycle,
                "recommendations": self._generate_recommendations(cycle)
            }
            recommendations.append(cycle_info)
        
        # Erstelle Ergebnisreport
        report = {
            "detected_cycles": len(cycles),
            "cycles_details": recommendations,
            "module_count": len(self.module_map),
            "dependency_edges": self.dependency_graph.number_of_edges()
        }
        
        ztm_log("Analyse abgeschlossen", level="INFO")
        return report
    
    def _generate_recommendations(self, cycle: List[str]) -> Dict:
        """
        Generiert Empfehlungen zur Behebung eines zirkulären Imports
        
        Args:
            cycle: Liste von Modulnamen im Zyklus
            
        Returns:
            Dictionary mit Empfehlungen
        """
        recommendations = {
            "factory_pattern": [],
            "dependency_injection": [],
            "lazy_loading": []
        }
        
        # Factory-Pattern-Empfehlungen
        for module in cycle:
            factory_recommendation = {
                "module": module,
                "explanation": f"Ersetze direkte Imports durch eine Factory-Funktion in {module}"
            }
            recommendations["factory_pattern"].append(factory_recommendation)
        
        # Dependency-Injection-Empfehlungen für Module mit vielen Abhängigkeiten
        for module in cycle:
            if len(self.import_map.get(module, [])) > 3:  # Wenn das Modul viele Imports hat
                di_recommendation = {
                    "module": module,
                    "explanation": f"Verwende Dependency Injection in {module}, um Abhängigkeiten zu reduzieren"
                }
                recommendations["dependency_injection"].append(di_recommendation)
        
        # Lazy-Loading-Empfehlungen
        for i, module in enumerate(cycle):
            next_module = cycle[(i + 1) % len(cycle)]
            lazy_recommendation = {
                "module": module,
                "target": next_module,
                "explanation": f"Verwende Lazy-Loading für den Import von {next_module} in {module}"
            }
            recommendations["lazy_loading"].append(lazy_recommendation)
        
        return recommendations
    
    def generate_report(self, output_file: Optional[str] = None) -> None:
        """
        Generiert einen detaillierten Bericht zur Analyse
        
        Args:
            output_file: Pfad zur Ausgabedatei (optional)
        """
        report = self.analyze()
        
        # Formatierter String-Report
        formatted_report = []
        formatted_report.append("="*80)
        formatted_report.append("MISO Ultimate - Circular Import Analysis Report")
        formatted_report.append("="*80)
        formatted_report.append(f"Insgesamt analysierte Module: {report['module_count']}")
        formatted_report.append(f"Gefundene zirkuläre Abhängigkeiten: {report['detected_cycles']}")
        formatted_report.append(f"Gesamtanzahl der Abhängigkeiten: {report['dependency_edges']}")
        formatted_report.append("\n")
        
        if report['detected_cycles'] > 0:
            formatted_report.append("-"*80)
            formatted_report.append("Gefundene zirkuläre Abhängigkeiten:")
            formatted_report.append("-"*80)
            
            for i, cycle_info in enumerate(report['cycles_details']):
                cycle = cycle_info['cycle']
                formatted_report.append(f"Zyklus {i+1}: {' -> '.join(cycle)} -> {cycle[0]}")
                
                formatted_report.append("\nEmpfohlene Lösungsansätze:")
                
                if cycle_info['recommendations']['factory_pattern']:
                    formatted_report.append("  Factory-Pattern:")
                    for rec in cycle_info['recommendations']['factory_pattern']:
                        formatted_report.append(f"    - {rec['explanation']}")
                
                if cycle_info['recommendations']['dependency_injection']:
                    formatted_report.append("  Dependency Injection:")
                    for rec in cycle_info['recommendations']['dependency_injection']:
                        formatted_report.append(f"    - {rec['explanation']}")
                
                if cycle_info['recommendations']['lazy_loading']:
                    formatted_report.append("  Lazy-Loading:")
                    for rec in cycle_info['recommendations']['lazy_loading']:
                        formatted_report.append(f"    - {rec['explanation']}")
                
                formatted_report.append("\n")
        else:
            formatted_report.append("Keine zirkulären Abhängigkeiten gefunden.")
        
        # Ausgabe des Reports
        full_report = "\n".join(formatted_report)
        print(full_report)
        
        # Speichern in Datei, falls gewünscht
        if output_file:
            try:
                # Speichere als Text-Report
                with open(output_file, 'w', encoding='utf-8') as file:
                    file.write(full_report)
                
                # Speichere auch als JSON für weitere Analysen
                json_output = output_file.replace('.txt', '.json')
                with open(json_output, 'w', encoding='utf-8') as file:
                    json.dump(report, file, indent=2)
                
                ztm_log(f"Report gespeichert in {output_file} und {json_output}", level="INFO")
            except Exception as e:
                ztm_log(f"Fehler beim Speichern des Reports: {e}", level="ERROR")

def main():
    """Hauptfunktion für Kommandozeilenaufruf"""
    parser = argparse.ArgumentParser(description='MISO Ultimate - Circular Import Detector')
    parser.add_argument('--root-dirs', '-r', nargs='+', required=True, help='Root-Verzeichnisse zur Analyse')
    parser.add_argument('--exclude-dirs', '-e', nargs='*', default=[], help='Zu ignorierende Verzeichnisse')
    parser.add_argument('--output', '-o', help='Ausgabedatei für den Report (optional)')
    
    args = parser.parse_args()
    
    detector = CircularImportDetector(args.root_dirs, args.exclude_dirs)
    detector.generate_report(args.output)

if __name__ == "__main__":
    main()
