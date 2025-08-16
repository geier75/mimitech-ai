#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template-Funktionen für den HTML-Reporter.

Dieses Modul stellt Funktionen für die Generierung des HTML-Templates bereit,
einschließlich CSS, JavaScript und Basis-HTML-Struktur.
"""

import json
from datetime import datetime
from typing import Dict, Any

def _get_html_header(self, theme):
    """Generiert den HTML-Header mit CSS-Stilen.
    
    Args:
        theme: Das zu verwendende Theme ('light' oder 'dark')
        
    Returns:
        HTML-String mit Header-Elementen
    """
    # Bestimme Theme-Farben
    if theme == 'dark':
        background_color = '#1e1e1e'
        text_color = '#e0e0e0'
        accent_color = '#2c7be5'
        card_bg = '#2d2d2d'
        border_color = '#444'
        hover_color = '#3a3a3a'
    else:  # light theme
        background_color = '#f8f9fa'
        text_color = '#333'
        accent_color = '#2c7be5'
        card_bg = '#fff'
        border_color = '#ddd'
        hover_color = '#f0f0f0'
    
    # CSS für das Styling
    css = f"""
    <style>
        :root {{
            --background-color: {background_color};
            --text-color: {text_color};
            --accent-color: {accent_color};
            --card-bg: {card_bg};
            --border-color: {border_color};
            --hover-color: {hover_color};
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
            margin: 0;
            padding: 0;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .report-header {{
            margin-bottom: 30px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
        }}
        
        .report-header h1 {{
            color: var(--accent-color);
            margin-bottom: 5px;
        }}
        
        .timestamp {{
            color: var(--text-color);
            opacity: 0.7;
            font-size: 0.9em;
        }}
        
        .navbar {{
            display: flex;
            flex-wrap: wrap;
            margin-top: 20px;
            gap: 10px;
        }}
        
        .tab-button {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 5px;
            transition: all 0.3s;
            color: var(--text-color);
            font-weight: 500;
        }}
        
        .tab-button:hover {{
            background-color: var(--hover-color);
        }}
        
        .tab-button.active {{
            background-color: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }}
        
        .tab-content {{
            display: none;
            padding: 20px;
            background-color: var(--card-bg);
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .card {{
            background-color: var(--card-bg);
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .t-mathematics-card {{
            background-color: #f0f7ff;
            border-left: 5px solid #2c7be5;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .dark .t-mathematics-card {{
            background-color: #1a3050;
            border-left: 5px solid #2c7be5;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background-color: var(--hover-color);
            font-weight: bold;
        }}
        
        tr:hover {{
            background-color: var(--hover-color);
        }}
        
        .filters {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .filter-group {{
            display: flex;
            flex-direction: column;
        }}
        
        .filter-label {{
            margin-bottom: 5px;
            font-weight: 500;
        }}
        
        .filter-select {{
            padding: 8px 10px;
            border-radius: 5px;
            border: 1px solid var(--border-color);
            background-color: var(--card-bg);
            color: var(--text-color);
        }}
        
        .chart-container {{
            height: 400px;
            margin-bottom: 30px;
        }}
        
        .grid-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .grid-item {{
            background-color: var(--card-bg);
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .recommendation-item {{
            background-color: var(--card-bg);
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
            border-left: 3px solid var(--accent-color);
        }}
        
        .export-buttons {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}
        
        .export-button {{
            padding: 8px 15px;
            background-color: var(--accent-color);
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }}
        
        .export-button:hover {{
            background-color: #1a5cbf;
        }}
        
        .report-footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            font-size: 0.9em;
            color: var(--text-color);
            opacity: 0.7;
        }}
        
        /* MISO-spezifische Stile */
        .miso-component {{
            background-color: #f5f5ff;
            border-left: 5px solid #7c4dff;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        
        .dark .miso-component {{
            background-color: #2d2d4d;
        }}
        
        .mprime-component {{
            background-color: #fff8e1;
            border-left: 5px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        
        .dark .mprime-component {{
            background-color: #4d4d2d;
        }}
        
        .improvement {{
            color: #4caf50;
        }}
        
        .regression {{
            color: #f44336;
        }}
        
        .unchanged {{
            color: #9e9e9e;
        }}
        
        .highlight-box {{
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
            background-color: rgba(44, 123, 229, 0.1);
            border-left: 5px solid var(--accent-color);
        }}
        
        .expandable-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}
        
        .expandable-content.expanded {{
            max-height: 5000px;
        }}
        
        .expandable-button {{
            background: none;
            border: none;
            color: var(--accent-color);
            cursor: pointer;
            padding: 5px 0;
            display: flex;
            align-items: center;
        }}
        
        .expandable-button::after {{
            content: '▼';
            margin-left: 5px;
            font-size: 0.8em;
        }}
        
        .expandable-button.expanded::after {{
            content: '▲';
        }}
    </style>
    
    <!-- Plotly.js für interaktive Charts -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    """
    
    return css

def _get_html_footer(self):
    """Generiert den HTML-Footer.
    
    Returns:
        HTML-String mit Footer-Elementen
    """
    return f"""
    <p>Generiert mit MISO Matrix-Benchmark-Tool</p>
    <p>T-Mathematics Engine Optimierung für Matrix-Operationen</p>
    <p>© {datetime.now().year} MISO Project</p>
    """

def _get_interactive_js(self, data):
    """Generiert den JavaScript-Code für interaktive Elemente.
    
    Args:
        data: Dictionary mit den Daten für die Charts
        
    Returns:
        HTML-String mit JavaScript-Code
    """
    # Konvertiere die Daten zu JSON für JavaScript
    chart_data_json = json.dumps(data["chart_data"])
    
    # Bereite Recommendations für JavaScript vor
    recommendations_json = json.dumps(data["recommendations"])
    
    # Bereite historische Daten vor, falls vorhanden
    historical_data_json = "null"
    if data["historical_comparisons"]:
        historical_data_json = json.dumps(data["historical_comparisons"])
    
    # JavaScript für die interaktiven Elemente
    js_code = f"""
    <script>
    // Daten für die Charts
    const chartData = {chart_data_json};
    
    // Empfehlungen
    const recommendations = {recommendations_json};
    
    // Historische Daten
    const historicalData = {historical_data_json};
    
    // Tab-Funktionalität
    function openTab(evt, tabName) {{
        const tabContents = document.getElementsByClassName("tab-content");
        for (let i = 0; i < tabContents.length; i++) {{
            tabContents[i].style.display = "none";
        }}
        
        const tabButtons = document.getElementsByClassName("tab-button");
        for (let i = 0; i < tabButtons.length; i++) {{
            tabButtons[i].className = tabButtons[i].className.replace(" active", "");
        }}
        
        document.getElementById(tabName).style.display = "block";
        evt.currentTarget.className += " active";
        
        // Aktualisiere Charts, wenn der entsprechende Tab geöffnet wird
        if (tabName === "Charts") {{
            initializeCharts();
        }}
        
        if (tabName === "Historical" && historicalData) {{
            initializeHistoricalCharts();
        }}
    }}
    
    // Event-Listener für Filter
    function applyFilters() {{
        const selectedOperation = document.getElementById('operation-filter').value;
        const selectedBackend = document.getElementById('backend-filter').value;
        const selectedDimension = document.getElementById('dimension-filter').value;
        
        // Alle Ergebniszeilen durchgehen und filtern
        const rows = document.querySelectorAll('.result-row');
        rows.forEach(row => {{
            const operation = row.getAttribute('data-operation');
            const backend = row.getAttribute('data-backend');
            const dimension = row.getAttribute('data-dimension');
            
            const matchesOperation = selectedOperation === 'all' || operation === selectedOperation;
            const matchesBackend = selectedBackend === 'all' || backend === selectedBackend;
            const matchesDimension = selectedDimension === 'all' || dimension === selectedDimension;
            
            // Zeile anzeigen oder verstecken
            row.style.display = (matchesOperation && matchesBackend && matchesDimension) ? '' : 'none';
        }});
        
        // Charts aktualisieren, wenn sichtbar
        if (document.getElementById('Charts').style.display === 'block') {{
            updateCharts(selectedOperation, selectedBackend, selectedDimension);
        }}
    }}
    
    // Initialisierung der Charts
    function initializeCharts() {{
        // Performance-Vergleich nach Backend
        createBackendPerformanceChart();
        
        // Performance-Vergleich nach Dimension
        createDimensionPerformanceChart();
        
        // Speicherverbrauch
        createMemoryUsageChart();
        
        // T-Mathematics Spezialisierte Charts
        createTMathematicsChart();
    }}
    
    // Chart-Erstellung
    function createBackendPerformanceChart() {{
        const operations = Object.keys(chartData.operations);
        const data = [];
        
        operations.forEach(operation => {{
            const backends = Object.keys(chartData.operations[operation].backends);
            backends.forEach(backend => {{
                const times = chartData.operations[operation].backends[backend].times;
                const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
                
                data.push({{
                    x: [backend],
                    y: [avgTime],
                    type: 'bar',
                    name: operation
                }});
            }});
        }});
        
        const layout = {{
            title: 'Durchschnittliche Ausführungszeit nach Backend',
            xaxis: {{ title: 'Backend' }},
            yaxis: {{ title: 'Zeit (s)' }},
            barmode: 'group'
        }};
        
        Plotly.newPlot('backendPerformanceChart', data, layout);
    }}
    
    function createDimensionPerformanceChart() {{
        const operations = Object.keys(chartData.operations);
        const data = [];
        
        operations.forEach(operation => {{
            const backends = Object.keys(chartData.operations[operation].backends);
            backends.forEach(backend => {{
                data.push({{
                    x: chartData.operations[operation].backends[backend].dimensions,
                    y: chartData.operations[operation].backends[backend].times,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: `${{operation}} - ${{backend}}`
                }});
            }});
        }});
        
        const layout = {{
            title: 'Ausführungszeit nach Dimension',
            xaxis: {{ title: 'Dimension' }},
            yaxis: {{ 
                title: 'Zeit (s)',
                type: 'log' 
            }}
        }};
        
        Plotly.newPlot('dimensionPerformanceChart', data, layout);
    }}
    
    function createMemoryUsageChart() {{
        const data = [];
        
        for (const operation in chartData.memory) {{
            for (const backend in chartData.memory[operation]) {{
                const memoryData = chartData.memory[operation][backend];
                const dimensions = memoryData.map(d => d.dimension);
                const memoryChanges = memoryData.map(d => d.memory_change);
                
                data.push({{
                    x: dimensions,
                    y: memoryChanges,
                    type: 'bar',
                    name: `${{operation}} - ${{backend}}`
                }});
            }}
        }}
        
        const layout = {{
            title: 'Speicherverbrauch nach Dimension',
            xaxis: {{ title: 'Dimension' }},
            yaxis: {{ title: 'Speicheränderung (MB)' }}
        }};
        
        Plotly.newPlot('memoryUsageChart', data, layout);
    }}
    
    // Spezieller Chart für T-Mathematics Engine Performance
    function createTMathematicsChart() {{
        // Filtern für MLX-spezifische Daten (T-Mathematics nutzt MLX für Apple Silicon)
        const operations = Object.keys(chartData.operations);
        const data = [];
        
        operations.forEach(operation => {{
            const backends = Object.keys(chartData.operations[operation].backends);
            backends.forEach(backend => {{
                if (backend === 'MLX') {{
                    data.push({{
                        x: chartData.operations[operation].backends[backend].dimensions,
                        y: chartData.operations[operation].backends[backend].times,
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: `${{operation}} (T-Mathematics mit MLX)`
                    }});
                }}
            }});
        }});
        
        const layout = {{
            title: 'T-Mathematics Engine Performance (MLX)',
            xaxis: {{ title: 'Matrix-Dimension' }},
            yaxis: {{ 
                title: 'Ausführungszeit (s)',
                type: 'log'
            }}
        }};
        
        Plotly.newPlot('tMathematicsChart', data, layout);
    }}
    
    // Historische Charts, falls Daten vorhanden
    function initializeHistoricalCharts() {{
        if (!historicalData) return;
        
        createHistoricalPerformanceChart();
        createHistoricalComparisonChart();
    }}
    
    function createHistoricalPerformanceChart() {{
        if (!historicalData) return;
        
        const data = [];
        const timestamps = historicalData.timestamps;
        
        // Sammle Verbesserungen/Regressionen über die Zeit
        const improvements = [];
        const regressions = [];
        
        historicalData.datasets.forEach((dataset, index) => {{
            improvements.push(dataset.improvements.length);
            regressions.push(dataset.regressions.length);
        }});
        
        // Verbesserungen
        data.push({{
            x: timestamps,
            y: improvements,
            type: 'bar',
            name: 'Verbesserungen',
            marker: {{ color: '#4caf50' }}
        }});
        
        // Regressionen
        data.push({{
            x: timestamps,
            y: regressions,
            type: 'bar',
            name: 'Regressionen',
            marker: {{ color: '#f44336' }}
        }});
        
        const layout = {{
            title: 'Historische Performance-Änderungen',
            barmode: 'group',
            xaxis: {{ title: 'Zeitstempel' }},
            yaxis: {{ title: 'Anzahl der Änderungen' }}
        }};
        
        Plotly.newPlot('historicalPerformanceChart', data, layout);
    }}
    
    function createHistoricalComparisonChart() {{
        if (!historicalData) return;
        
        // Besondere Hervorhebung für Apple Silicon und T-Mathematics Verbesserungen
        const appleData = historicalData.apple_silicon_boost;
        const data = [];
        
        // Apple Silicon Boost
        const boostDates = [];
        const boostValues = [];
        
        appleData.forEach(item => {{
            if (item.boost) {{
                boostDates.push(item.timestamp);
                boostValues.push(1); // Markierung für Boost
            }}
        }});
        
        if (boostDates.length > 0) {{
            data.push({{
                x: boostDates,
                y: boostValues,
                type: 'scatter',
                mode: 'markers',
                marker: {{ 
                    size: 15,
                    symbol: 'star',
                    color: '#ffc107'
                }},
                name: 'Apple Silicon Optimierung'
            }});
        }}
        
        // T-Mathematics Versionen
        const versionDates = historicalData.timestamps;
        const versionLabels = historicalData.t_math_versions;
        
        if (versionDates.length > 0) {{
            data.push({{
                x: versionDates,
                y: Array(versionDates.length).fill(0), // Platziere am unteren Rand
                type: 'scatter',
                mode: 'markers+text',
                marker: {{ size: 10, color: '#2c7be5' }},
                text: versionLabels,
                textposition: 'bottom center',
                name: 'T-Mathematics Version'
            }});
        }}
        
        const layout = {{
            title: 'T-Mathematics Engine & Apple Silicon Optimierungen',
            xaxis: {{ title: 'Zeitstempel' }},
            yaxis: {{ 
                title: 'Status',
                showticklabels: false,
                range: [-0.5, 1.5]
            }},
            showlegend: true
        }};
        
        Plotly.newPlot('historicalBoostChart', data, layout);
    }}
    
    // Expandable Sections
    function toggleExpandable(elementId) {{
        const content = document.getElementById(elementId);
        const button = document.querySelector('[data-target="' + elementId + '"]');
        
        content.classList.toggle('expanded');
        button.classList.toggle('expanded');
    }}
    
    // Export-Funktionen
    function exportToCSV() {{
        // CSV-Export-Logik
        let csvContent = "Operation,Backend,Dimension,Precision,Mean Time,Std Dev,Min Time,Max Time,Success Rate\\n";
        
        document.querySelectorAll('.result-row').forEach(row => {{
            if (row.style.display !== 'none') {{
                const operation = row.getAttribute('data-operation');
                const backend = row.getAttribute('data-backend');
                const dimension = row.getAttribute('data-dimension');
                const precision = row.getAttribute('data-precision');
                const meanTime = row.children[4].textContent;
                const stdDev = row.children[5].textContent;
                const minTime = row.children[6].textContent;
                const maxTime = row.children[7].textContent;
                const successRate = row.children[8].textContent;
                
                csvContent += `"${{operation}}","${{backend}}",${{dimension}},"${{precision}}",${{meanTime}},${{stdDev}},${{minTime}},${{maxTime}},${{successRate}}\\n`;
            }}
        }});
        
        const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', 'benchmark_results.csv');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }}
    
    function exportToPDF() {{
        alert('PDF-Export wird in einer späteren Version implementiert.');
    }}
    
    function printResults() {{
        window.print();
    }}
    
    // Laden und Anzeigen der Empfehlungen
    function loadRecommendations() {{
        if (recommendations && recommendations.length > 0) {{
            const recList = document.getElementById('recommendationsList');
            recommendations.forEach(rec => {{
                const li = document.createElement('li');
                li.className = 'recommendation-item';
                li.innerHTML = rec;
                recList.appendChild(li);
            }});
        }}
    }}
    
    // Initialisierung beim Laden der Seite
    document.addEventListener('DOMContentLoaded', function() {{
        // Tab-Funktionalität initialisieren
        const defaultTab = document.querySelector('.tab-button');
        if (defaultTab) {{
            defaultTab.click();
        }}
        
        // Filter-Event-Listener
        const filters = document.querySelectorAll('.filter-select');
        filters.forEach(filter => {{
            filter.addEventListener('change', applyFilters);
        }});
        
        // Export-Button-Event-Listener
        document.getElementById('exportCSV').addEventListener('click', exportToCSV);
        document.getElementById('exportPDF').addEventListener('click', exportToPDF);
        document.getElementById('printResults').addEventListener('click', printResults);
        
        // Empfehlungen laden
        loadRecommendations();
        
        // Expandable-Button-Event-Listener
        const expandableButtons = document.querySelectorAll('.expandable-button');
        expandableButtons.forEach(button => {{
            const targetId = button.getAttribute('data-target');
            button.addEventListener('click', () => toggleExpandable(targetId));
        }});
    }});
    </script>
    """
    
    return js_code
