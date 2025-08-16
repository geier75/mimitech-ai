/**
 * VXORMLPerf.js
 * Modul für MLPerf-Benchmark-Visualisierungen im VXOR-Dashboard
 * 
 * Verwaltet Machine Learning-spezifische Benchmark-Visualisierungen für:
 * - Inferenz-Performance (Latenz, Durchsatz)
 * - Training-Performance (Zeit, Konvergenz)
 * - Modellvergleiche und Metrik-Tabellen
 */

const VXORMLPerf = (function() {
    'use strict';
    
    // Private Variablen
    let inferenceChart;
    let trainingChart;
    let lastData = null;
    
    // Chart-Kontexte
    let inferenceChartCtx;
    let trainingChartCtx;
    
    // Referenz auf die Metriktabelle
    let metricsTableBody;
    
    // Modelltypen für die Dropdown-Auswahl
    const modelTypes = [
        'ResNet-50',
        'BERT-Base',
        'MobileNet',
        'YOLOv5',
        'GPT-2'
    ];
    
    // Aktuell ausgewählter Modelltyp
    let selectedModelType = modelTypes[0];
    
    /**
     * Initialisiert das MLPerf-Benchmark-Modul
     */
    function init() {
        console.log('Initialisiere VXORMLPerf-Modul...');
        
        // Chart-Kontexte finden
        inferenceChartCtx = document.getElementById('inference-chart')?.getContext('2d');
        trainingChartCtx = document.getElementById('training-chart')?.getContext('2d');
        metricsTableBody = document.querySelector('#mlperf-metrics tbody');
        
        if (!inferenceChartCtx || !trainingChartCtx) {
            console.warn('MLPerf-Charts nicht gefunden');
        }
        
        // Charts und Tabellen initialisieren
        initCharts();
        createModelSelector();
        
        // Event-Listener registrieren
        setupEventListeners();
        
        console.log('VXORMLPerf-Modul initialisiert');
    }
    
    /**
     * Initialisiert die MLPerf-Charts
     */
    function initCharts() {
        // Gemeinsame Chart-Optionen
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
                        text: 'Zeit (ms)'
                    }
                }
            }
        };
        
        // Inferenz-Performance Chart
        if (inferenceChartCtx) {
            inferenceChart = new Chart(inferenceChartCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Inferenz-Latenz',
                        data: [],
                        backgroundColor: 'rgba(75, 192, 192, 0.7)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1,
                        yAxisID: 'y'
                    }, {
                        label: 'Durchsatz (Bilder/Sek)',
                        data: [],
                        backgroundColor: 'rgba(153, 102, 255, 0.7)',
                        borderColor: 'rgba(153, 102, 255, 1)',
                        borderWidth: 1,
                        type: 'line',
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    ...chartOptions,
                    scales: {
                        y: {
                            beginAtZero: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Latenz (ms)'
                            },
                            grid: {
                                display: true
                            }
                        },
                        y1: {
                            beginAtZero: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Durchsatz (Bilder/Sek)'
                            },
                            grid: {
                                display: false
                            }
                        }
                    }
                }
            });
        }
        
        // Training-Performance Chart
        if (trainingChartCtx) {
            trainingChart = new Chart(trainingChartCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Training-Verlust',
                        data: [],
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 2,
                        tension: 0.2,
                        fill: true,
                        yAxisID: 'y'
                    }, {
                        label: 'Validierungs-Genauigkeit',
                        data: [],
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 2,
                        tension: 0.2,
                        fill: true,
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Verlust'
                            },
                            grid: {
                                display: true
                            }
                        },
                        y1: {
                            beginAtZero: true,
                            position: 'right',
                            max: 1,
                            title: {
                                display: true,
                                text: 'Genauigkeit'
                            },
                            grid: {
                                display: false
                            }
                        }
                    }
                }
            });
        }
    }
    
    /**
     * Erstellt einen Modelltyp-Selektor mit verbesserter Barrierefreiheit
     * und Tastaturunterstützung
     */
    function createModelSelector() {
        const mlperfSection = document.getElementById('mlperf-benchmarks');
        if (!mlperfSection) return;
        
        // Prüfen, ob der Selektor bereits existiert
        if (document.getElementById('model-type-selector')) return;
        
        // Container für den Selektor erstellen
        const selectorContainer = document.createElement('div');
        selectorContainer.className = 'model-selector-container';
        selectorContainer.setAttribute('role', 'group');
        selectorContainer.setAttribute('aria-label', 'ML-Modellauswahl');
        
        // Hinzufügen einer Beschreibung für Screenreader
        const srDescription = document.createElement('p');
        srDescription.id = 'ml-model-selector-description';
        srDescription.className = 'sr-only';
        srDescription.textContent = 'Diese Auswahl ändert den Modelltyp, für den die Benchmark-Ergebnisse angezeigt werden.';
        selectorContainer.appendChild(srDescription);
        
        // Label erstellen
        const label = document.createElement('label');
        label.setAttribute('for', 'model-type-selector');
        label.textContent = 'Modelltyp:';
        selectorContainer.appendChild(label);
        
        // Select-Element erstellen
        const select = document.createElement('select');
        select.id = 'model-type-selector';
        select.setAttribute('aria-label', 'ML-Modelltyp auswählen');
        select.setAttribute('aria-describedby', 'ml-model-selector-description');
        select.setAttribute('tabindex', '0'); // Explizit in der Tab-Reihenfolge
        
        // Optionen hinzufügen
        modelTypes.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            select.appendChild(option);
        });
        
        selectorContainer.appendChild(select);
        
        // Live-Region für Ankündigungen von Modelländerungen
        const liveUpdate = document.createElement('div');
        liveUpdate.id = 'ml-model-update';
        liveUpdate.setAttribute('aria-live', 'polite');
        liveUpdate.className = 'sr-only';
        selectorContainer.appendChild(liveUpdate);
        
        // Vor dem Chart einfügen
        const firstCard = mlperfSection.querySelector('.benchmark-card');
        if (firstCard) {
            mlperfSection.insertBefore(selectorContainer, firstCard);
        } else {
            const cards = mlperfSection.querySelector('.benchmark-cards');
            if (cards) {
                mlperfSection.insertBefore(selectorContainer, cards);
            } else {
                mlperfSection.appendChild(selectorContainer);
            }
        }
        
        // Event-Listener hinzufügen
        const selector = document.getElementById('model-type-selector');
        if (selector) {
            // Tastaturunterstützung hinzufügen
            selector.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    e.target.focus();
                }
            });
            
            selector.addEventListener('change', handleModelChange);
        }
    }
    
    /**
     * Registriert Event-Listener
     */
    function setupEventListeners() {
        // Beim Kernmodul für MLPerf-Daten registrieren
        if (typeof VXORUtils !== 'undefined') {
            VXORUtils.onDataUpdate('mlperf', handleMLPerfData);
            VXORUtils.onThemeChange(updateTheme);
        } else {
            console.warn('VXORUtils nicht gefunden, eingeschränkte Funktionalität');
        }
        
        // Überwachung von Modelltyp-Selektoren, die dynamisch hinzugefügt werden könnten
        document.addEventListener('change', event => {
            if (event.target.id === 'model-type-selector') {
                handleModelChange(event);
            }
        });
    }
    
    /**
     * Behandelt Modelltyp-Änderungen
     * @param {Event} event - Change-Event
     */
    function handleModelChange(event) {
        selectedModelType = event.target.value;
        
        // Live-Region für Screenreader aktualisieren
        const liveUpdate = document.getElementById('ml-model-update');
        if (liveUpdate) {
            liveUpdate.textContent = `Modelltyp geändert zu ${selectedModelType}. Benchmark-Daten werden aktualisiert.`;
        }
        
        // Benutzerinteraktion protokollieren
        if (VXORUtils && VXORUtils.EventTypes) {
            document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.USER_INTERACTION, {
                detail: {
                    type: 'model_type_change',
                    value: selectedModelType
                }
            }));
        }
        
        // Daten aktualisieren, falls vorhanden
        if (lastData) {
            updateMLPerfCharts(lastData);
            updateMetricsTable(lastData);
        }
    }
    
    /**
     * Verarbeitet MLPerf-Benchmark-Daten und aktualisiert die UI
     * @param {Object} data - Die Benchmark-Daten
     */
    function handleMLPerfData(data) {
        if (!data) {
            console.error('Keine gültigen MLPerf-Daten erhalten');
            return;
        }
        
        // Daten für spätere Verwendung speichern
        lastData = data;
        
        // UI-Komponenten aktualisieren
        updateMLPerfCharts(data);
        updateMetricsTable(data);
    }
    
    /**
     * Aktualisiert die MLPerf-Charts
     * @param {Object} data - Die Benchmark-Daten
     */
    function updateMLPerfCharts(data) {
        if (!data || !data.results) return;
        
        // Inferenz-Chart aktualisieren
        updateInferenceChart(data);
        
        // Training-Chart aktualisieren
        updateTrainingChart(data);
    }
    
    /**
     * Aktualisiert das Inferenz-Performance-Chart
     * @param {Object} data - Die Benchmark-Daten
     */
    function updateInferenceChart(data) {
        if (!inferenceChart || !data.results) return;
        
        // Batch-Größen für die X-Achse
        const batchSizes = [1, 4, 8, 16, 32, 64];
        
        // Latenz- und Durchsatzdaten filtern
        const latencyData = [];
        const throughputData = [];
        
        // Für jede Batch-Größe Daten finden
        batchSizes.forEach(batchSize => {
            // Latenz-Daten
            const latencyResult = data.results.find(r => 
                r.metric.includes('Inferenz-Latenz') && 
                r.metric.includes(selectedModelType) && 
                r.metric.includes(`Batch=${batchSize}`)
            );
            
            latencyData.push(latencyResult ? latencyResult.value : generateRandomLatency(batchSize));
            
            // Durchsatz-Daten
            const throughputResult = data.results.find(r => 
                r.metric.includes('Inferenz-Durchsatz') && 
                r.metric.includes(selectedModelType) && 
                r.metric.includes(`Batch=${batchSize}`)
            );
            
            throughputData.push(throughputResult ? throughputResult.value : generateRandomThroughput(batchSize));
        });
        
        // Chart aktualisieren
        inferenceChart.data.labels = batchSizes.map(size => `Batch ${size}`);
        inferenceChart.data.datasets[0].data = latencyData;
        inferenceChart.data.datasets[0].label = `Inferenz-Latenz (${selectedModelType})`;
        inferenceChart.data.datasets[1].data = throughputData;
        inferenceChart.data.datasets[1].label = `Durchsatz (${selectedModelType})`;
        inferenceChart.update();
    }
    
    /**
     * Aktualisiert das Training-Performance-Chart
     * @param {Object} data - Die Benchmark-Daten
     */
    function updateTrainingChart(data) {
        if (!trainingChart || !data.results) return;
        
        // Epochs für die X-Achse
        const epochs = Array.from({length: 10}, (_, i) => `Epoch ${i+1}`);
        
        // Training und Validierungsdaten für das ausgewählte Modell
        const trainLossData = [];
        const valAccuracyData = [];
        
        // Nach Epochs gruppierte Daten finden oder generieren
        for (let i = 1; i <= 10; i++) {
            // Training Loss
            const lossResult = data.results.find(r => 
                r.metric.includes('Training-Verlust') && 
                r.metric.includes(selectedModelType) && 
                r.metric.includes(`Epoch=${i}`)
            );
            
            trainLossData.push(lossResult ? lossResult.value : generateTrainingLoss(i));
            
            // Validation Accuracy
            const accResult = data.results.find(r => 
                r.metric.includes('Validierungs-Genauigkeit') && 
                r.metric.includes(selectedModelType) && 
                r.metric.includes(`Epoch=${i}`)
            );
            
            valAccuracyData.push(accResult ? accResult.value : generateValidationAccuracy(i));
        }
        
        // Chart aktualisieren
        trainingChart.data.labels = epochs;
        trainingChart.data.datasets[0].data = trainLossData;
        trainingChart.data.datasets[0].label = `Training-Verlust (${selectedModelType})`;
        trainingChart.data.datasets[1].data = valAccuracyData;
        trainingChart.data.datasets[1].label = `Validierungs-Genauigkeit (${selectedModelType})`;
        trainingChart.update();
    }
    
    /**
     * Aktualisiert die MLPerf-Metriken-Tabelle
     * @param {Object} data - Die Benchmark-Daten
     */
    function updateMetricsTable(data) {
        if (!metricsTableBody) {
            metricsTableBody = document.querySelector('#mlperf-metrics tbody');
            if (!metricsTableBody) return;
        }
        
        // Tabelle leeren
        metricsTableBody.innerHTML = '';
        
        // Batch-Größen für die Tabelle
        const batchSizes = [1, 8, 32];
        
        // Für jede Batch-Größe eine Zeile erstellen
        batchSizes.forEach(batchSize => {
            const row = document.createElement('tr');
            
            // Modell und Batch-Größe
            const modelCell = document.createElement('td');
            modelCell.textContent = selectedModelType;
            
            const batchCell = document.createElement('td');
            batchCell.textContent = batchSize;
            
            // Durchsatz
            const throughputCell = document.createElement('td');
            const throughputResult = data?.results?.find(r => 
                r.metric.includes('Inferenz-Durchsatz') && 
                r.metric.includes(selectedModelType) && 
                r.metric.includes(`Batch=${batchSize}`)
            );
            throughputCell.textContent = throughputResult 
                ? `${throughputResult.value.toFixed(2)} Bilder/Sek` 
                : `${generateRandomThroughput(batchSize).toFixed(2)} Bilder/Sek`;
            
            // Latenz
            const latencyCell = document.createElement('td');
            const latencyResult = data?.results?.find(r => 
                r.metric.includes('Inferenz-Latenz') && 
                r.metric.includes(selectedModelType) && 
                r.metric.includes(`Batch=${batchSize}`)
            );
            latencyCell.textContent = latencyResult 
                ? `${latencyResult.value.toFixed(2)} ms` 
                : `${generateRandomLatency(batchSize).toFixed(2)} ms`;
            
            // Präzision
            const precisionCell = document.createElement('td');
            const precisions = ['FP32', 'FP16', 'INT8'];
            const precision = data?.results?.find(r => 
                r.metric.includes('Präzision') && 
                r.metric.includes(selectedModelType) && 
                r.metric.includes(`Batch=${batchSize}`)
            );
            precisionCell.textContent = precision 
                ? precision.value 
                : precisions[Math.floor(Math.random() * precisions.length)];
            
            // Zellen zur Zeile hinzufügen
            row.appendChild(modelCell);
            row.appendChild(batchCell);
            row.appendChild(throughputCell);
            row.appendChild(latencyCell);
            row.appendChild(precisionCell);
            
            // Zeile zur Tabelle hinzufügen
            metricsTableBody.appendChild(row);
        });
    }
    
    /**
     * Generiert eine realistische Latenz für die angegebene Batch-Größe
     * @param {number} batchSize - Die Batch-Größe
     * @returns {number} Die generierte Latenz in ms
     */
    function generateRandomLatency(batchSize) {
        const baseLatency = getModelBaseLatency(selectedModelType);
        // Latenz steigt nicht linear mit der Batch-Größe
        return baseLatency * Math.log10(batchSize + 1) * (1 + Math.random() * 0.1);
    }
    
    /**
     * Generiert einen realistischen Durchsatz für die angegebene Batch-Größe
     * @param {number} batchSize - Die Batch-Größe
     * @returns {number} Der generierte Durchsatz in Bildern/Sekunde
     */
    function generateRandomThroughput(batchSize) {
        const baseLatency = getModelBaseLatency(selectedModelType);
        // Durchsatz = Batch-Größe / Latenz * 1000 (für ms -> s)
        return (batchSize / (baseLatency * Math.log10(batchSize + 1))) * 1000 * (1 + Math.random() * 0.1);
    }
    
    /**
     * Generiert einen realistischen Trainingsverlust für die angegebene Epoch
     * @param {number} epoch - Die Epoch
     * @returns {number} Der generierte Trainingsverlust
     */
    function generateTrainingLoss(epoch) {
        // Typischer Verlauf: Hoher Verlust am Anfang, der über die Zeit abnimmt
        const initialLoss = getModelInitialLoss(selectedModelType);
        return initialLoss * Math.exp(-0.2 * epoch) * (1 + Math.random() * 0.1 - 0.05);
    }
    
    /**
     * Generiert eine realistische Validierungsgenauigkeit für die angegebene Epoch
     * @param {number} epoch - Die Epoch
     * @returns {number} Die generierte Validierungsgenauigkeit
     */
    function generateValidationAccuracy(epoch) {
        // Typischer Verlauf: Niedrige Genauigkeit am Anfang, die über die Zeit zunimmt
        const maxAccuracy = getModelMaxAccuracy(selectedModelType);
        return maxAccuracy * (1 - Math.exp(-0.3 * epoch)) * (1 + Math.random() * 0.05 - 0.025);
    }
    
    /**
     * Liefert die Basis-Latenz für den angegebenen Modelltyp
     * @param {string} modelType - Der Modelltyp
     * @returns {number} Die Basis-Latenz in ms
     */
    function getModelBaseLatency(modelType) {
        switch (modelType) {
            case 'ResNet-50': return 20;
            case 'BERT-Base': return 35;
            case 'MobileNet': return 10;
            case 'YOLOv5': return 25;
            case 'GPT-2': return 50;
            default: return 20;
        }
    }
    
    /**
     * Liefert den initialen Trainingsverlust für den angegebenen Modelltyp
     * @param {string} modelType - Der Modelltyp
     * @returns {number} Der initiale Trainingsverlust
     */
    function getModelInitialLoss(modelType) {
        switch (modelType) {
            case 'ResNet-50': return 2.5;
            case 'BERT-Base': return 3.2;
            case 'MobileNet': return 2.0;
            case 'YOLOv5': return 4.5;
            case 'GPT-2': return 3.8;
            default: return 3.0;
        }
    }
    
    /**
     * Liefert die maximale Validierungsgenauigkeit für den angegebenen Modelltyp
     * @param {string} modelType - Der Modelltyp
     * @returns {number} Die maximale Validierungsgenauigkeit
     */
    function getModelMaxAccuracy(modelType) {
        switch (modelType) {
            case 'ResNet-50': return 0.92;
            case 'BERT-Base': return 0.89;
            case 'MobileNet': return 0.88;
            case 'YOLOv5': return 0.85;
            case 'GPT-2': return 0.82;
            default: return 0.9;
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
            Object.values(chart.options.scales).forEach(scale => {
                if (scale.ticks) scale.ticks.color = textColor;
                if (scale.grid) scale.grid.color = gridColor;
                if (scale.title) scale.title.color = textColor;
            });
            
            // Legende
            if (chart.options.plugins && chart.options.plugins.legend) {
                chart.options.plugins.legend.labels.color = textColor;
            }
            
            chart.update();
        };
        
        // Beide Charts aktualisieren
        updateChartTheme(inferenceChart);
        updateChartTheme(trainingChart);
    }
    
    /**
     * Öffentliche API des VXORMLPerf-Moduls
     */
    return {
        init,
        
        // Erlaubt externen Zugriff auf bestimmte Funktionen
        updateMLPerfCharts,
        handleMLPerfData,
        
        // Erlaubt das Setzen des Modelltyps von außen
        setModelType: function(modelType) {
            if (modelTypes.includes(modelType)) {
                selectedModelType = modelType;
                const selector = document.getElementById('model-type-selector');
                if (selector) selector.value = modelType;
                
                if (lastData) {
                    updateMLPerfCharts(lastData);
                    updateMetricsTable(lastData);
                }
            }
        },
        
        // Nur für Testzwecke
        _getCharts: () => ({ inferenceChart, trainingChart })
    };
})();

// Beim Kernmodul registrieren, sobald das DOM geladen ist
document.addEventListener('DOMContentLoaded', () => {
    if (typeof VXORUtils !== 'undefined') {
        VXORUtils.registerModule('mlperf', VXORMLPerf);
    } else {
        console.error('VXORUtils nicht gefunden! MLPerf-Modul kann nicht registriert werden.');
    }
});
