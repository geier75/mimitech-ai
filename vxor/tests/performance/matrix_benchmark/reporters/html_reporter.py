#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML-Reporter für Matrix-Benchmarks.

Dieses Modul stellt eine HTMLReporter-Klasse bereit, die interaktive HTML-Berichte
mit erweiterten Visualisierungen und historischen Vergleichen generiert.
"""

import os
import sys
import json
import logging
import platform
import psutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

from .base_reporter import BenchmarkReporter

# Setup logging
logger = logging.getLogger(__name__)

class HTMLReporter(BenchmarkReporter):
    """HTML-Reporter für Benchmark-Ergebnisse mit interaktiven Elementen.
    
    Diese Klasse erstellt HTML-Berichte mit interaktiven Filtern, Visualisierungen und
    historischen Vergleichen, optimiert für die MISO T-Mathematics Engine.
    """
    
    def __init__(self, use_plotly=True, include_historical=True, theme="light"):
        """Initialisiert den HTML-Reporter.
        
        Args:
            use_plotly: Ob Plotly.js für interaktive Diagramme verwendet werden soll
            include_historical: Ob historische Vergleiche einbezogen werden sollen
            theme: Design-Thema ('light' oder 'dark')
        """
        self.use_plotly = use_plotly
        self.include_historical = include_historical
        self.theme = theme
        self.historical_datasets = []
    
    def load_historical_data(self, history_dir="benchmark_history"):
        """Lädt historische Benchmark-Ergebnisse.
        
        Args:
            history_dir: Verzeichnis mit historischen Daten
        """
        if not os.path.exists(history_dir):
            logger.warning(f"Historisches Verzeichnis nicht gefunden: {history_dir}")
            return
            
        history_files = sorted(
            [f for f in os.listdir(history_dir) if f.endswith('.json')],
            key=lambda f: os.path.getmtime(os.path.join(history_dir, f)),
            reverse=True
        )[:5]  # Nehme nur die 5 neuesten Dateien
        
        logger.info(f"Lade {len(history_files)} historische Datensätze aus {history_dir}")
        
        for file in history_files:
            try:
                with open(os.path.join(history_dir, file), 'r') as f:
                    data = json.load(f)
                    if 'results' in data:
                        # Verwende BenchmarkResult.from_dict für jedes Ergebnis
                        self.historical_datasets.append({
                            'timestamp': data.get('timestamp', file),
                            'results': data['results']  # Dies würde normalerweise konvertiert werden
                        })
                        logger.debug(f"Historischer Datensatz geladen: {file}")
            except Exception as e:
                logger.warning(f"Fehler beim Laden der historischen Daten aus {file}: {str(e)}")
    
    def generate_report(self, results, output_file: str) -> None:
        """Generiert einen HTML-Bericht aus den Benchmark-Ergebnissen.
        
        Args:
            results: Liste der Benchmark-Ergebnisse
            output_file: Pfad zur HTML-Ausgabedatei
        """
        # Stelle sicher, dass das Ausgabeverzeichnis existiert
        self._ensure_directory_exists(output_file)
        
        # Lade historische Daten, falls aktiviert
        if self.include_historical:
            self.load_historical_data()
        
        logger.info(f"Generiere HTML-Bericht mit {len(results)} Ergebnissen...")
        
        # Bereite Daten für das Template vor
        template_data = self._prepare_template_data(results)
        
        # Generiere HTML mit einem Template-System
        html_content = self._render_template(template_data)
        
        # Speichere HTML-Datei
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        logger.info(f"HTML-Bericht in {output_file} erstellt.")
    
    def _prepare_template_data(self, results) -> Dict[str, Any]:
        """Bereitet Daten für das HTML-Template vor.
        
        Args:
            results: Liste der Benchmark-Ergebnisse
            
        Returns:
            Dictionary mit aufbereiteten Daten für das Template
        """
        # Extrahiere verschiedene Kategorien für Filter
        operations = sorted(list(set(r.operation.name for r in results)))
        backends = sorted(list(set(r.backend.name for r in results)))
        dimensions = sorted(list(set(r.dimension for r in results)))
        precisions = sorted(list(set(r.precision.name for r in results)))
        
        # Bereite Chart-Daten vor
        chart_data = self._prepare_chart_data(results)
        
        # Finde schnellstes Backend und schnellste Operation
        fastest_backend, backend_speedup = self._get_fastest_backend(results)
        fastest_operation, operation_time = self._get_fastest_operation(results)
        
        # Generiere Empfehlungen basierend auf den Ergebnissen
        recommendations = self._generate_recommendations(results)
        
        # Historische Vergleiche, falls verfügbar
        historical_comparisons = None
        if self.include_historical and self.historical_datasets:
            historical_comparisons = self._prepare_historical_comparison(
                results, self.historical_datasets
            )
        
        # Sammle alle Daten
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
            "operations": operations,
            "backends": backends,
            "dimensions": dimensions,
            "precisions": precisions,
            "chart_data": chart_data,
            "fastest_backend": fastest_backend,
            "backend_speedup": backend_speedup,
            "fastest_operation": fastest_operation,
            "operation_time": operation_time,
            "recommendations": recommendations,
            "historical_comparisons": historical_comparisons,
            "theme": self.theme,
            "system_info": self._get_system_info()
        }
    
    def _render_template(self, data) -> str:
        """Rendert das HTML-Template mit den gegebenen Daten.
        
        Args:
            data: Daten für das Template
            
        Returns:
            HTML-String mit dem vollständigen Bericht
        """
        # Generiere Header mit CSS und JS
        header = self._get_html_header(data["theme"])
        
        # Generiere die verschiedenen Abschnitte
        overview_section = self._generate_overview_section(data)
        details_section = self._generate_details_section(data)
        charts_section = self._generate_charts_section(data)
        recommendations_section = self._generate_recommendations_section(data)
        
        # Historischer Abschnitt, falls verfügbar
        historical_section = ""
        if data["historical_comparisons"]:
            historical_section = self._generate_historical_section(data)
        
        # Navigation erstellen
        navbar = """
        <div class="navbar">
            <button class="tab-button active" onclick="openTab(event, 'Overview')">Übersicht</button>
            <button class="tab-button" onclick="openTab(event, 'Details')">Details</button>
            <button class="tab-button" onclick="openTab(event, 'Charts')">Diagramme</button>
            <button class="tab-button" onclick="openTab(event, 'Recommendations')">Empfehlungen</button>
        """
        
        if data["historical_comparisons"]:
            navbar += '<button class="tab-button" onclick="openTab(event, \'Historical\')">Historischer Vergleich</button>'
        
        navbar += "</div>"
        
        # Interaktives JavaScript für Charts und Filter
        interactive_js = self._get_interactive_js(data)
        
        # Footer
        footer = self._get_html_footer()
        
        # Zusammenstellung des gesamten HTML-Dokuments
        html = f"""
        <!DOCTYPE html>
        <html lang="de">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Matrix-Benchmark-Ergebnisse</title>
            {header}
        </head>
        <body class="{data['theme']}">
            <div class="container">
                {navbar}
                
                {overview_section}
                {details_section}
                {charts_section}
                {recommendations_section}
                {historical_section}
                
                <div class="report-footer">
                    {footer}
                </div>
            </div>
            
            {interactive_js}
        </body>
        </html>
        """
        
        return html
    
    def _get_system_info(self) -> Dict[str, str]:
        """Sammelt Systeminformationen für den Bericht.
        
        Returns:
            Dictionary mit Systeminformationen
        """
        system_info = {
            "Python Version": sys.version.split('\n')[0],
            "Betriebssystem": f"{platform.system()} {platform.release()}",
            "Plattform": platform.platform(),
            "Prozessor": platform.processor(),
            "RAM": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
            "Datum": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # T-Mathematics Engine Version, falls verfügbar
        try:
            from .... import t_mathematics
            system_info["t_math_version"] = getattr(t_mathematics, "__version__", "unbekannt")
        except ImportError:
            system_info["t_math_version"] = "nicht verfügbar"
        
        # MLX verfügbar?
        try:
            import mlx
            # Verschiedene Wege, die Version zu ermitteln
            if hasattr(mlx, "__version__"):
                system_info["mlx_version"] = mlx.__version__
            elif hasattr(mlx, "version"):
                system_info["mlx_version"] = mlx.version
            else:
                # Wenn keine Version verfügbar ist, vermerke nur die Verfügbarkeit
                system_info["mlx_available"] = "Ja"
        except ImportError:
            system_info["mlx_available"] = "Nein"
        
        # NumPy verfügbar?
        try:
            import numpy
            system_info["numpy_version"] = numpy.__version__
        except ImportError:
            pass
        
        # PyTorch verfügbar?
        try:
            import torch
            system_info["torch_version"] = torch.__version__
            system_info["cuda_available"] = str(torch.cuda.is_available())
        except ImportError:
            pass
        
        # Apple Silicon?
        system_info["is_apple_silicon"] = platform.machine() == "arm64" and platform.system() == "Darwin"
        
        # Verfügbarkeit von MISO-Komponenten prüfen
        try:
            from .... import echo_prime
            system_info["echo_prime_available"] = "Ja"
        except ImportError:
            system_info["echo_prime_available"] = "Nein"
        
        try:
            from .... import mprime
            system_info["mprime_available"] = "Ja"
        except ImportError:
            system_info["mprime_available"] = "Nein"
        
        return system_info
    
    def _ensure_directory_exists(self, file_path):
        """Stellt sicher, dass das Verzeichnis für die angegebene Datei existiert.
        
        Args:
            file_path: Pfad zur Datei
        """
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logger.info(f"Verzeichnis erstellt: {directory}")
            except Exception as e:
                logger.error(f"Fehler beim Erstellen des Verzeichnisses {directory}: {str(e)}")

    # Die weiteren Methoden des HTMLReporters werden in den folgenden Dateien implementiert
    from .html_reporter_charts import _prepare_chart_data, _get_fastest_backend, _get_fastest_operation
    from .html_reporter_recommendations import _generate_recommendations, _prepare_historical_comparison
    from .html_reporter_template import _get_html_header, _get_html_footer, _get_interactive_js
    from .html_reporter_sections import (_generate_overview_section, _generate_details_section, 
                                        _generate_charts_section, _generate_recommendations_section,
                                        _generate_historical_section)
