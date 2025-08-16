#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abschnittsfunktionen für den HTML-Reporter.

Dieses Modul stellt Funktionen zur Generierung der einzelnen HTML-Abschnitte bereit.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List

# Setup logging
logger = logging.getLogger(__name__)

def _generate_overview_section(self, data) -> str:
    """Generiert den Übersichtsabschnitt des Berichts.
    
    Args:
        data: Daten für das Template
        
    Returns:
        HTML-String mit dem Übersichtsabschnitt
    """
    # Formatiere Systeminfos
    system_info = data["system_info"]
    system_info_html = ""
    for key, value in system_info.items():
        system_info_html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"
    
    # Formatiere Schnellübersicht
    overview_html = f"""
    <div class="card">
        <h2>Schnellübersicht</h2>
        <div class="grid-container">
            <div class="grid-item">
                <h3>Schnellstes Backend</h3>
                <p class="highlight">{data["fastest_backend"]}</p>
                <p>Mit {data["backend_speedup"]:.2f}x Beschleunigung</p>
            </div>
            <div class="grid-item">
                <h3>Schnellste Operation</h3>
                <p class="highlight">{data["fastest_operation"]}</p>
                <p>Ausführungszeit: {data["operation_time"]:.6f}s</p>
            </div>
            <div class="grid-item">
                <h3>Getestete Operationen</h3>
                <p class="highlight">{len(data["operations"])}</p>
            </div>
            <div class="grid-item">
                <h3>Getestete Backends</h3>
                <p class="highlight">{len(data["backends"])}</p>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>Systeminfos</h2>
        <table>
            <tbody>
                {system_info_html}
            </tbody>
        </table>
    </div>
    """
    
    # T-Mathematics Engine spezifische Informationen
    t_math_version = system_info.get("t_math_version", "Nicht verfügbar")
    is_apple_silicon = system_info.get("is_apple_silicon", False)
    
    # Box mit T-Mathematics-Infos hinzufügen
    if t_math_version != "Nicht verfügbar":
        overview_html += f"""
        <div class="t-mathematics-card">
            <h3>T-Mathematics Engine</h3>
            <p>Version: {t_math_version}</p>
            <p>Apple Silicon Unterstützung: {'Ja' if is_apple_silicon else 'Nein'}</p>
            <p>Diese Benchmark verwendet die T-Mathematics Engine für optimierte Matrix-Operationen.</p>
        </div>
        """
    
    # Spezifische Komponenten für MISO-Projekt hervorheben
    overview_html += f"""
    <div class="highlight-box">
        <h3>MISO-Komponenten im Benchmark</h3>
        <button class="expandable-button" data-target="misoComponents">Komponenten anzeigen</button>
        <div id="misoComponents" class="expandable-content">
            <div class="miso-component">
                <h4>ECHO-PRIME Kernel</h4>
                <p>Zeitlinien-Operationen mit Tensoren</p>
            </div>
            <div class="mprime-component">
                <h4>MPRIME Engine</h4>
                <p>Symbolische Mathematik mit SymbolTree und TopoNet</p>
            </div>
            <div class="miso-component">
                <h4>Q-Logik Framework</h4>
                <p>Erweiterte Paradoxauflösung mit Tensoroperationen</p>
            </div>
        </div>
    </div>
    """
    
    return f"""
    <div id="Overview" class="tab-content">
        <div class="report-header">
            <h1>Matrix-Benchmark-Ergebnisse</h1>
            <p class="timestamp">Generiert am {data["timestamp"]}</p>
        </div>
        
        <div class="export-buttons">
            <button id="exportCSV" class="export-button">Als CSV exportieren</button>
            <button id="exportPDF" class="export-button">Als PDF exportieren</button>
            <button id="printResults" class="export-button">Drucken</button>
        </div>
        
        {overview_html}
    </div>
    """

def _generate_details_section(self, data) -> str:
    """Generiert den Detailabschnitt des Berichts.
    
    Args:
        data: Daten für das Template
        
    Returns:
        HTML-String mit dem Detailabschnitt
    """
    # Filter für Operationen, Backends und Dimensionen
    operations_options = ''.join([f'<option value="{op}">{op}</option>' for op in data["operations"]])
    backends_options = ''.join([f'<option value="{backend}">{backend}</option>' for backend in data["backends"]])
    dimensions_options = ''.join([f'<option value="{dim}">{dim}</option>' for dim in data["dimensions"]])
    
    # Erstelle Tabellenzeilen für Resultate
    result_rows = ""
    for i, result in enumerate(data["results"]):
        op_name = result.operation.name
        backend_name = result.backend.name
        dimension = result.dimension
        precision = result.precision.name
        
        # Speicheränderung, falls verfügbar
        memory_change = getattr(result, 'memory_change', 0)
        if memory_change != 0:
            memory_change_formatted = f"{memory_change / (1024 * 1024):.2f} MB"
        else:
            memory_change_formatted = "N/A"
        
        result_rows += f"""
        <tr class="result-row" 
            data-operation="{op_name}" 
            data-backend="{backend_name}" 
            data-dimension="{dimension}"
            data-precision="{precision}">
            <td>{op_name}</td>
            <td>{backend_name}</td>
            <td>{dimension}</td>
            <td>{precision}</td>
            <td>{result.mean_time:.6f}</td>
            <td>{result.std_dev:.6f}</td>
            <td>{result.min_time:.6f}</td>
            <td>{result.max_time:.6f}</td>
            <td>{result.success_rate:.2f}%</td>
            <td>{memory_change_formatted}</td>
        </tr>
        """
    
    return f"""
    <div id="Details" class="tab-content">
        <h2>Detaillierte Ergebnisse</h2>
        
        <div class="filters">
            <div class="filter-group">
                <label class="filter-label" for="operation-filter">Operation:</label>
                <select id="operation-filter" class="filter-select">
                    <option value="all">Alle</option>
                    {operations_options}
                </select>
            </div>
            
            <div class="filter-group">
                <label class="filter-label" for="backend-filter">Backend:</label>
                <select id="backend-filter" class="filter-select">
                    <option value="all">Alle</option>
                    {backends_options}
                </select>
            </div>
            
            <div class="filter-group">
                <label class="filter-label" for="dimension-filter">Dimension:</label>
                <select id="dimension-filter" class="filter-select">
                    <option value="all">Alle</option>
                    {dimensions_options}
                </select>
            </div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Operation</th>
                    <th>Backend</th>
                    <th>Dimension</th>
                    <th>Präzision</th>
                    <th>Durchschn. Zeit (s)</th>
                    <th>Std. Abw. (s)</th>
                    <th>Min Zeit (s)</th>
                    <th>Max Zeit (s)</th>
                    <th>Erfolgsrate (%)</th>
                    <th>Speicheränderung</th>
                </tr>
            </thead>
            <tbody>
                {result_rows}
            </tbody>
        </table>
    </div>
    """

def _generate_charts_section(self, data) -> str:
    """Generiert den Diagrammabschnitt des Berichts.
    
    Args:
        data: Daten für das Template
        
    Returns:
        HTML-String mit dem Diagrammabschnitt
    """
    return f"""
    <div id="Charts" class="tab-content">
        <h2>Leistungsdiagramme</h2>
        
        <div class="card">
            <h3>Performance nach Backend</h3>
            <div id="backendPerformanceChart" class="chart-container"></div>
        </div>
        
        <div class="card">
            <h3>Performance nach Dimension</h3>
            <div id="dimensionPerformanceChart" class="chart-container"></div>
        </div>
        
        <div class="card">
            <h3>Speicherverbrauch</h3>
            <div id="memoryUsageChart" class="chart-container"></div>
        </div>
        
        <div class="t-mathematics-card">
            <h3>T-Mathematics Engine Performance</h3>
            <div id="tMathematicsChart" class="chart-container"></div>
        </div>
    </div>
    """

def _generate_recommendations_section(self, data) -> str:
    """Generiert den Empfehlungsabschnitt des Berichts.
    
    Args:
        data: Daten für das Template
        
    Returns:
        HTML-String mit dem Empfehlungsabschnitt
    """
    return f"""
    <div id="Recommendations" class="tab-content">
        <h2>Empfehlungen und Optimierungen</h2>
        
        <div class="card">
            <h3>Leistungsoptimierungen</h3>
            <ul id="recommendationsList">
                <!-- Wird mit JavaScript gefüllt -->
            </ul>
        </div>
        
        <div class="miso-component">
            <h3>MISO-spezifische Empfehlungen</h3>
            <p>Die folgenden Empfehlungen beziehen sich auf die MISO-Komponenten:</p>
            <ul>
                <li>Für Zeitlinien-Operationen in ECHO-PRIME wird die Verwendung der T-Mathematics Engine mit MLX empfohlen.</li>
                <li>MPRIME Engine-Operationen profitieren besonders von der Tensorparallelisierung.</li>
                <li>Für die Paradoxauflösung werden optimierte Matrix-Dimensionen empfohlen.</li>
            </ul>
        </div>
    </div>
    """

def _generate_historical_section(self, data) -> str:
    """Generiert den historischen Vergleichsabschnitt des Berichts.
    
    Args:
        data: Daten für das Template
        
    Returns:
        HTML-String mit dem historischen Vergleichsabschnitt
    """
    if not data["historical_comparisons"]:
        return f"""
        <div id="Historical" class="tab-content">
            <h2>Historische Vergleiche</h2>
            <p>Keine historischen Daten verfügbar.</p>
        </div>
        """
    
    return f"""
    <div id="Historical" class="tab-content">
        <h2>Historische Vergleiche</h2>
        
        <div class="card">
            <h3>Performance-Trends</h3>
            <div id="historicalPerformanceChart" class="chart-container"></div>
        </div>
        
        <div class="card">
            <h3>T-Mathematics und Hardware-Optimierungen</h3>
            <div id="historicalBoostChart" class="chart-container"></div>
        </div>
        
        <div class="card">
            <h3>Details zu historischen Änderungen</h3>
            <table>
                <thead>
                    <tr>
                        <th>Zeitstempel</th>
                        <th>Verbesserungen</th>
                        <th>Regressionen</th>
                        <th>T-Mathematics Version</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Wird mit JavaScript gefüllt -->
                </tbody>
            </table>
        </div>
    </div>
    """
