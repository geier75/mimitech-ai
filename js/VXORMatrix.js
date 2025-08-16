/**
 * VXORMatrix.js
 * Modul für Matrix-Benchmark-Visualisierungen im VXOR-Dashboard
 * 
 * Verwaltet Matrix-spezifische Charts, Tabellen und Benutzerinteraktionen
 * Stellt Funktionalität bereit für:
 * - Grundlegende Matrix-Operationen (Multiplikation, Addition, etc.)
 * - Fortgeschrittene Matrix-Algorithmen (SVD, Eigenwerte, etc.)
 * - Matrix-Größen-Auswahl und Leistungsvergleich
 */

const VXORMatrix = (function() {
    'use strict';
    
    // Private Variablen
    let matrixBasicChart;
    let matrixAdvancedChart;
    let selectedMatrixSize = "1024";
    let lastData = null;
    
    // Chart-Kontexte und DOM-Elemente
    let matrixBasicChartCtx;
    let matrixAdvancedChartCtx;
    let matrixMetricsTableBody;
    let matrixSizeSelector;
    
    /**
     * Initialisiert die Matrix-Benchmarks
     */
    function init() {
        console.log('Initialisiere VXORMatrix-Modul...');
        
        // DOM-Elemente finden
        matrixBasicChartCtx = document.getElementById('matrix-basic-chart')?.getContext('2d');
        matrixAdvancedChartCtx = document.getElementById('matrix-advanced-chart')?.getContext('2d');
        matrixMetricsTableBody = document.querySelector('#matrix-metrics tbody');
        matrixSizeSelector = document.getElementById('matrix-size-selector');
        
        if (!matrixBasicChartCtx || !matrixAdvancedChartCtx) {
            console.error('Matrix-Chart-Canvas nicht gefunden');
            return;
        }
        
        // Charts initialisieren
        initMatrixCharts();
        
        // Event-Listener registrieren
        setupEventListeners();
        
        // Beim Kernmodul für Matrix-Daten registrieren
        VXORUtils.onDataUpdate('matrix', handleMatrixData);
        
        // Theme-Änderungen überwachen
        VXORUtils.onThemeChange(updateTheme);
        
        console.log('VXORMatrix-Modul initialisiert');
    }
    
    /**
     * Initialisiert die Matrix-Charts
     */
    function initMatrixCharts() {
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw.toFixed(2)} ms`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Ausführungszeit (ms)'
                    }
                }
            }
        };
        
        // Grundlegende Matrix-Operationen Chart
        matrixBasicChart = new Chart(matrixBasicChartCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: `Matrix-Operationen (${selectedMatrixSize}×${selectedMatrixSize})`,
                    data: [],
                    backgroundColor: 'rgba(48, 63, 159, 0.7)',
                    borderColor: 'rgba(48, 63, 159, 1)',
                    borderWidth: 1
                }]
            },
            options: chartOptions
        });
        
        // Fortgeschrittene Matrix-Algorithmen Chart
        matrixAdvancedChart = new Chart(matrixAdvancedChartCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Fortgeschrittene Matrix-Algorithmen',
                    data: [],
                    backgroundColor: 'rgba(103, 58, 183, 0.7)',
                    borderColor: 'rgba(103, 58, 183, 1)',
                    borderWidth: 1
                }]
            },
            options: chartOptions
        });
    }
    
    /**
     * Richtet Event-Listener ein
     */
    function setupEventListeners() {
        // Matrix-Größen-Auswahl
        if (matrixSizeSelector) {
            matrixSizeSelector.addEventListener('change', handleMatrixSizeChange);
        } else {
            console.warn('Matrix-Größen-Selector nicht gefunden, erstelle dynamisch...');
            createMatrixSizeSelector();
        }
        
        // Auf Daten-Update-Events vom Kernmodul reagieren
        document.addEventListener('vxor:dataUpdated', (event) => {
            if (event.detail.category === 'matrix' || event.detail.category === 'all') {
                handleMatrixData(event.detail.data);
            }
        });
    }
    
    /**
     * Erstellt einen Matrix-Größen-Selector, falls er nicht existiert
     */
    function createMatrixSizeSelector() {
        const matrixSection = document.getElementById('matrix-benchmarks');
        if (!matrixSection) return;
        
        // Container für den Selector erstellen
        const sizeSelector = document.createElement('div');
        sizeSelector.className = 'size-selector';
        sizeSelector.innerHTML = `
            <label for="matrix-size-selector">Matrix-Größe:</label>
            <select id="matrix-size-selector" aria-label="Matrix-Größe auswählen">
                <option value="1024">1024×1024</option>
                <option value="2048">2048×2048</option>
                <option value="4096">4096×4096</option>
            </select>
        `;
        
        // Vor dem ersten Chart einfügen
        const firstChart = matrixSection.querySelector('.chart-card');
        if (firstChart) {
            matrixSection.insertBefore(sizeSelector, firstChart);
        } else {
            matrixSection.appendChild(sizeSelector);
        }
        
        // Reference aktualisieren und Event-Listener hinzufügen
        matrixSizeSelector = document.getElementById('matrix-size-selector');
        if (matrixSizeSelector) {
            matrixSizeSelector.addEventListener('change', handleMatrixSizeChange);
        }
    }
    
    /**
     * Behandelt Änderungen an der Matrix-Größenauswahl
     * @param {Event} event - Das Change-Event
     */
    function handleMatrixSizeChange(event) {
        selectedMatrixSize = event.target.value;
        
        // Benutzereingabe-Event protokollieren
        if (VXORUtils.EventTypes && VXORUtils.EventTypes.USER_INTERACTION) {
            document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.USER_INTERACTION, {
                detail: {
                    type: 'matrix_size_change',
                    value: selectedMatrixSize
                }
            }));
        }
        
        // Daten aktualisieren, falls vorhanden
        if (lastData) {
            updateMatrixCharts(lastData);
        } else {
            // Daten neu laden, falls keine vorhanden
            VXORUtils.fetchBenchmarkData(VXORUtils.getCurrentState().selectedComponent, 'matrix');
        }
    }
    
    /**
     * Verarbeitet Matrix-Benchmark-Daten und aktualisiert UI
     * @param {Object} data - Die Benchmark-Daten
     */
    function handleMatrixData(data) {
        if (!data || !data.results) {
            console.error('Keine gültigen Matrix-Daten erhalten');
            return;
        }
        
        // Daten speichern für spätere Verwendung
        lastData = data;
        
        // UI aktualisieren
        updateMatrixCharts(data);
    }
    
    /**
     * Aktualisiert die Matrix-Charts mit neuen Daten
     * @param {Object} data - Die Benchmark-Daten
     */
    function updateMatrixCharts(data) {
        // Filtere Matrix-spezifische Daten
        const matrixResults = data.results.filter(result => 
            result.component?.includes('Matrix') || 
            result.metric?.includes('Matrix') ||
            result.metric?.includes('Inversion') ||
            result.metric?.startsWith('FLOPS')
        );
        
        // Update Matrix-Größen-basierte Tabelle
        updateMatrixSizeTable(matrixResults);
        
        // Update Matrix-Basic Chart (grundlegende Operationen)
        const basicOps = ['Matrix-Mult', 'Matrix-Add', 'Matrix-Vektor', 'Transposition'];
        const basicLabels = [];
        const basicValues = [];
        
        data.performance?.labels?.forEach((label, index) => {
            if (basicOps.some(op => label.includes(op)) && label.includes(selectedMatrixSize)) {
                // Vereinfache Label zur besseren Lesbarkeit
                const simplifiedLabel = label.replace(`(${selectedMatrixSize}x${selectedMatrixSize})`, '')
                                             .replace('Matrix-', '')
                                             .trim();
                basicLabels.push(simplifiedLabel);
                basicValues.push(data.performance.values[index]);
            }
        });
        
        if (matrixBasicChart) {
            matrixBasicChart.data.labels = basicLabels;
            matrixBasicChart.data.datasets[0].data = basicValues;
            matrixBasicChart.data.datasets[0].label = `Matrix-Operationen (${selectedMatrixSize}×${selectedMatrixSize})`;
            matrixBasicChart.update();
        }
        
        // Update Matrix-Advanced Chart (fortgeschrittene Algorithmen)
        const advancedOps = ['Singulärwertzerlegung', 'Eigenwert', 'Cholesky', 'LU-Zerlegung', 'QR-Zerlegung'];
        const advancedLabels = [];
        const advancedValues = [];
        
        data.performance?.labels?.forEach((label, index) => {
            if (advancedOps.some(op => label.includes(op)) && (label.includes(selectedMatrixSize) || !label.match(/\d+x\d+/))) {
                // Vereinfache das Label
                const simplifiedLabel = advancedOps.find(op => label.includes(op));
                advancedLabels.push(simplifiedLabel);
                advancedValues.push(data.performance.values[index]);
            }
        });
        
        if (matrixAdvancedChart) {
            matrixAdvancedChart.data.labels = advancedLabels;
            matrixAdvancedChart.data.datasets[0].data = advancedValues;
            matrixAdvancedChart.update();
        }
    }
    
    /**
     * Aktualisiert die Tabelle mit Matrix-Größen-Benchmarks
     * @param {Array} matrixResults - Die gefilterten Matrix-Ergebnisse
     */
    function updateMatrixSizeTable(matrixResults) {
        if (!matrixMetricsTableBody) {
            console.warn('Matrix-Metrics-Table nicht gefunden');
            return;
        }
        
        matrixMetricsTableBody.innerHTML = '';
        
        const matrixSizeTests = {
            'Matrix-Mult': 'Matrix-Multiplikation',
            'Inversion': 'Matrixinversion',
            'Eigenwert': 'Eigenwertberechnung',
            'SVD': 'Singulärwertzerlegung'
        };
        
        // Erstelle Zeilen für jeden Test
        for (const [testId, testName] of Object.entries(matrixSizeTests)) {
            const row = document.createElement('tr');
            const sizes = ['1024', '2048', '4096'];
            
            let rowHtml = `<td>${testName}</td>`;
            let unit = 'ms';
            
            // Füge Werte für jede Größe hinzu
            sizes.forEach(size => {
                const result = matrixResults.find(r => 
                    r.metric?.includes(testId) && r.metric?.includes(size)
                );
                
                if (result) {
                    rowHtml += `<td>${result.value.toFixed(2)}</td>`;
                    unit = result.unit || 'ms';
                } else {
                    rowHtml += `<td>-</td>`;
                }
            });
            
            rowHtml += `<td>${unit}</td>`;
            row.innerHTML = rowHtml;
            row.setAttribute('aria-label', `Benchmark für ${testName}`);
            matrixMetricsTableBody.appendChild(row);
        }
    }
    
    /**
     * Aktualisiert das Theme für die Charts
     * @param {boolean} isDarkMode - Ob Dark-Mode aktiv ist
     */
    function updateTheme(isDarkMode) {
        const textColor = isDarkMode ? '#e0e0e0' : '#333333';
        const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
        
        const updateChartTheme = (chart) => {
            if (!chart) return;
            
            // Achsenbeschriftungen
            chart.options.scales.x.ticks.color = textColor;
            chart.options.scales.y.ticks.color = textColor;
            chart.options.scales.x.grid.color = gridColor;
            chart.options.scales.y.grid.color = gridColor;
            chart.options.scales.y.title.color = textColor;
            
            // Legende
            chart.options.plugins.legend.labels.color = textColor;
            
            chart.update();
        };
        
        // Themes für alle Charts aktualisieren
        updateChartTheme(matrixBasicChart);
        updateChartTheme(matrixAdvancedChart);
    }
    
    /**
     * Matrix-Modul exportieren
     */
    return {
        init,
        
        // Erlaubt externen Zugriff auf bestimmte Funktionen
        updateMatrixCharts,
        setMatrixSize: (size) => {
            if (['1024', '2048', '4096'].includes(size)) {
                selectedMatrixSize = size;
                if (matrixSizeSelector) {
                    matrixSizeSelector.value = size;
                }
                if (lastData) {
                    updateMatrixCharts(lastData);
                }
            }
        },
        getSelectedMatrixSize: () => selectedMatrixSize,
        
        // Nur für Testzwecke
        _getCharts: () => ({ matrixBasicChart, matrixAdvancedChart })
    };
})();

// Beim Kernmodul registrieren, sobald das DOM geladen ist
document.addEventListener('DOMContentLoaded', () => {
    if (typeof VXORUtils !== 'undefined') {
        VXORUtils.registerModule('matrix', VXORMatrix);
    } else {
        console.error('VXORUtils nicht gefunden! Matrix-Modul kann nicht registriert werden.');
    }
});
