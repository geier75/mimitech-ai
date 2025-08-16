/**
 * VXORTMath.js
 * Modul für T-Mathematics-Benchmark-Visualisierungen im VXOR-Dashboard
 * 
 * Verwaltet T-Mathematics-spezifische Charts und Visualisierungen für:
 * - Numerische Methoden (Integration, Differentiation, etc.)
 * - Optimierungsalgorithmen (Gradient Descent, Particle Swarm, etc.)
 * - Tensor-Operationen und andere mathematische Berechnungen
 */

const VXORTMath = (function() {
    'use strict';
    
    // Private Variablen
    let numericsChart;
    let optimizationChart;
    let lastData = null;
    
    // Chart-Kontexte
    let numericsChartCtx;
    let optimizationChartCtx;
    
    // Optimierungsalgorithmen für die Konvergenz-Visualisierung
    const optimizationAlgorithms = [
        'Gradient Descent', 
        'Newton-Verfahren',
        'Particle Swarm',
        'Simulated Annealing',
        'Genetischer Algorithmus'
    ];
    
    /**
     * Initialisiert das T-Mathematics-Benchmark-Modul
     */
    function init() {
        console.log('Initialisiere VXORTMath-Modul...');
        
        // Chart-Kontexte finden
        numericsChartCtx = document.getElementById('numerics-chart')?.getContext('2d');
        optimizationChartCtx = document.getElementById('optimization-chart')?.getContext('2d');
        
        if (!numericsChartCtx || !optimizationChartCtx) {
            console.warn('T-Mathematics-Charts nicht gefunden');
            createChartsIfNeeded();
        }
        
        // Charts initialisieren
        initCharts();
        
        // Event-Listener registrieren
        setupEventListeners();
        
        console.log('VXORTMath-Modul initialisiert');
    }
    
    /**
     * Initialisiert die T-Mathematics-Charts
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
                        text: 'Ausführungszeit (ms)'
                    }
                }
            }
        };
        
        // Numerische Methoden Chart
        if (numericsChartCtx) {
            numericsChart = new Chart(numericsChartCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Numerische Methoden',
                        data: [],
                        backgroundColor: 'rgba(0, 188, 212, 0.7)',
                        borderColor: 'rgba(0, 188, 212, 1)',
                        borderWidth: 1
                    }]
                },
                options: chartOptions
            });
        }
        
        // Optimierungsalgorithmen Chart
        if (optimizationChartCtx) {
            optimizationChart = new Chart(optimizationChartCtx, {
                type: 'line',
                data: {
                    labels: Array.from({length: 10}, (_, i) => `Iteration ${i+1}`),
                    datasets: [{
                        label: 'Konvergenzrate',
                        data: [],
                        backgroundColor: 'rgba(255, 152, 0, 0.2)',
                        borderColor: 'rgba(255, 152, 0, 1)',
                        borderWidth: 2,
                        tension: 0.2,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `Fehler: ${context.raw.toFixed(4)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Fehler'
                            },
                            reverse: true // Da kleinere Fehler besser sind (höher im Chart)
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Iteration'
                            }
                        }
                    }
                }
            });
        }
    }
    
    /**
     * Erstellt die Charts, falls sie nicht im DOM existieren
     */
    function createChartsIfNeeded() {
        const tmathsSection = document.getElementById('tmaths-benchmarks');
        if (!tmathsSection) return;
        
        const benchmarkCards = tmathsSection.querySelector('.benchmark-cards');
        if (!benchmarkCards) {
            // Container für Benchmark-Karten erstellen
            const cardsContainer = document.createElement('div');
            cardsContainer.className = 'benchmark-cards';
            tmathsSection.appendChild(cardsContainer);
        }
        
        // Numerische Methoden Chart erstellen, falls nicht vorhanden
        if (!document.getElementById('numerics-chart')) {
            const numericCard = document.createElement('div');
            numericCard.className = 'benchmark-card';
            numericCard.innerHTML = `
                <h3>Numerische Methoden</h3>
                <canvas id="numerics-chart" width="400" height="300" aria-label="Numerische Methoden Benchmark"></canvas>
            `;
            benchmarkCards.appendChild(numericCard);
            numericsChartCtx = document.getElementById('numerics-chart').getContext('2d');
        }
        
        // Optimierungsalgorithmen Chart erstellen, falls nicht vorhanden
        if (!document.getElementById('optimization-chart')) {
            const optimizationCard = document.createElement('div');
            optimizationCard.className = 'benchmark-card';
            optimizationCard.innerHTML = `
                <h3>Optimierungsalgorithmen</h3>
                <canvas id="optimization-chart" width="400" height="300" aria-label="Optimierungsalgorithmen Benchmark"></canvas>
            `;
            benchmarkCards.appendChild(optimizationCard);
            optimizationChartCtx = document.getElementById('optimization-chart').getContext('2d');
        }
    }
    
    /**
     * Registriert Event-Listener
     */
    function setupEventListeners() {
        // Beim Kernmodul für T-Mathematics-Daten registrieren
        if (typeof VXORUtils !== 'undefined') {
            VXORUtils.onDataUpdate('tmaths', handleTMathData);
            
            // Theme-Änderungen überwachen
            VXORUtils.onThemeChange(updateTheme);
        }
        
        // Algorithmus-Auswahl für Optimierungschart (falls vorhanden)
        const algoSelector = document.getElementById('optimization-algo-selector');
        if (algoSelector) {
            algoSelector.addEventListener('change', (e) => {
                updateOptimizationChart(e.target.value);
            });
        } else {
            createAlgorithmSelector();
        }
    }
    
    /**
     * Erstellt einen Algorithmus-Selektor für Optimierungsalgorithmen
     */
    function createAlgorithmSelector() {
        const optimizationCard = document.querySelector('.benchmark-card:has(#optimization-chart)');
        if (!optimizationCard) return;
        
        const selectorContainer = document.createElement('div');
        selectorContainer.className = 'algo-selector-container';
        selectorContainer.innerHTML = `
            <label for="optimization-algo-selector">Algorithmus:</label>
            <select id="optimization-algo-selector" aria-label="Optimierungsalgorithmus auswählen">
                ${optimizationAlgorithms.map(algo => 
                    `<option value="${algo}">${algo}</option>`
                ).join('')}
            </select>
        `;
        
        // Vor dem Chart einfügen
        const canvas = optimizationCard.querySelector('canvas');
        optimizationCard.insertBefore(selectorContainer, canvas);
        
        // Event-Listener hinzufügen
        const selector = document.getElementById('optimization-algo-selector');
        if (selector) {
            selector.addEventListener('change', (e) => {
                updateOptimizationChart(e.target.value);
                
                // UI-Interaktion protokollieren
                if (VXORUtils && VXORUtils.EventTypes) {
                    document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.USER_INTERACTION, {
                        detail: {
                            type: 'optimization_algorithm_change',
                            value: e.target.value
                        }
                    }));
                }
            });
        }
    }
    
    /**
     * Verarbeitet T-Mathematics-Benchmark-Daten und aktualisiert die UI
     * @param {Object} data - Die Benchmark-Daten
     */
    function handleTMathData(data) {
        if (!data || !data.results) {
            console.error('Keine gültigen T-Mathematics-Daten erhalten');
            return;
        }
        
        // Daten für spätere Verwendung speichern
        lastData = data;
        
        // UI-Komponenten aktualisieren
        updateTMathCharts(data);
    }
    
    /**
     * Aktualisiert die T-Mathematics-Charts
     * @param {Object} data - Die Benchmark-Daten
     */
    function updateTMathCharts(data) {
        if (!data) return;
        
        // Numerische Methoden Chart aktualisieren
        updateNumericsChart(data);
        
        // Optimierungsalgorithmen Chart aktualisieren
        updateOptimizationChart();
    }
    
    /**
     * Aktualisiert das Numerische-Methoden-Chart
     * @param {Object} data - Die Benchmark-Daten
     */
    function updateNumericsChart(data) {
        if (!numericsChart || !data.results) return;
        
        // Numerische Methoden filtern
        const numericMethods = ['Polynom-Auswertung', 'Nullstellen', 'Integration', 'Differentiation', 
                               'Quadratur', 'Interpolation', 'Extrapolation'];
        
        const labels = [];
        const values = [];
        
        // Daten aus den Ergebnissen extrahieren
        if (data.results && Array.isArray(data.results)) {
            data.results.forEach(result => {
                const methodName = result.metric;
                if (numericMethods.some(nm => methodName.includes(nm))) {
                    labels.push(methodName);
                    values.push(result.value);
                }
            });
        } 
        
        // Alternativ aus performance-Daten extrahieren
        else if (data.performance && data.performance.labels) {
            data.performance.labels.forEach((label, index) => {
                if (numericMethods.some(nm => label.includes(nm))) {
                    labels.push(label);
                    values.push(data.performance.values[index]);
                }
            });
        }
        
        // Chart aktualisieren
        numericsChart.data.labels = labels;
        numericsChart.data.datasets[0].data = values;
        numericsChart.update();
    }
    
    /**
     * Aktualisiert das Optimierungsalgorithmen-Chart
     * @param {string} [algorithm] - Optionaler Algorithmus-Name
     */
    function updateOptimizationChart(algorithm) {
        if (!optimizationChart) return;
        
        // Falls kein Algorithmus angegeben, den aktuell ausgewählten oder den ersten verwenden
        const algoSelector = document.getElementById('optimization-algo-selector');
        const selectedAlgo = algorithm || (algoSelector ? algoSelector.value : optimizationAlgorithms[0]);
        
        // Daten für Konvergenz generieren (simuliert)
        const iterations = Array.from({length: 10}, (_, i) => `Iteration ${i+1}`);
        let convergenceData;
        
        // Für jede Algorithmen-Art ein realistisches Konvergenzverhalten simulieren
        if (selectedAlgo === 'Gradient Descent') {
            // Lineare Konvergenz mit Plateaus
            convergenceData = [1.0, 0.8, 0.7, 0.6, 0.55, 0.5, 0.48, 0.47, 0.45, 0.44];
        } else if (selectedAlgo === 'Newton-Verfahren') {
            // Quadratische Konvergenz (schneller)
            convergenceData = [1.0, 0.5, 0.2, 0.08, 0.02, 0.005, 0.001, 0.0005, 0.0002, 0.0001];
        } else if (selectedAlgo === 'Particle Swarm') {
            // Ungleichmäßige Konvergenz mit Sprüngen
            convergenceData = [1.0, 0.7, 0.65, 0.4, 0.38, 0.36, 0.2, 0.15, 0.12, 0.1];
        } else if (selectedAlgo === 'Simulated Annealing') {
            // Stochastische Konvergenz mit gelegentlichen Verschlechterungen
            convergenceData = [1.0, 0.8, 0.85, 0.6, 0.45, 0.5, 0.35, 0.25, 0.2, 0.18];
        } else if (selectedAlgo === 'Genetischer Algorithmus') {
            // Generationenbasierte Konvergenz mit Plateaus
            convergenceData = [1.0, 0.9, 0.75, 0.74, 0.5, 0.49, 0.48, 0.3, 0.29, 0.25];
        } else {
            // Fallback
            convergenceData = Array.from({length: 10}, (_, i) => 1.0 / (i + 1));
        }
        
        // Etwas Rauschen hinzufügen für realistischere Visualisierung
        convergenceData = convergenceData.map(val => val + (Math.random() * 0.03 - 0.015));
        
        // Chart aktualisieren
        optimizationChart.data.labels = iterations;
        optimizationChart.data.datasets[0].data = convergenceData;
        optimizationChart.data.datasets[0].label = `${selectedAlgo} Konvergenz`;
        
        // Farben basierend auf Algorithmus anpassen
        let colorHue;
        switch (selectedAlgo) {
            case 'Gradient Descent': colorHue = 33; break;    // Orange
            case 'Newton-Verfahren': colorHue = 200; break;   // Blau
            case 'Particle Swarm': colorHue = 120; break;     // Grün
            case 'Simulated Annealing': colorHue = 0; break;  // Rot
            case 'Genetischer Algorithmus': colorHue = 270; break; // Lila
            default: colorHue = 200;
        }
        
        optimizationChart.data.datasets[0].borderColor = `hsla(${colorHue}, 80%, 50%, 1)`;
        optimizationChart.data.datasets[0].backgroundColor = `hsla(${colorHue}, 80%, 50%, 0.2)`;
        
        optimizationChart.update();
    }
    
    /**
     * Aktualisiert das Theme für Charts
     * @param {boolean} isDarkMode - Ob Dark-Mode aktiv ist
     */
    function updateTheme(isDarkMode) {
        const textColor = isDarkMode ? '#e0e0e0' : '#333333';
        const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
        
        const updateChartTheme = (chart) => {
            if (!chart) return;
            
            // Achsenbeschriftungen
            if (chart.options.scales.x) {
                chart.options.scales.x.ticks.color = textColor;
                chart.options.scales.x.grid.color = gridColor;
                if (chart.options.scales.x.title) {
                    chart.options.scales.x.title.color = textColor;
                }
            }
            
            if (chart.options.scales.y) {
                chart.options.scales.y.ticks.color = textColor;
                chart.options.scales.y.grid.color = gridColor;
                if (chart.options.scales.y.title) {
                    chart.options.scales.y.title.color = textColor;
                }
            }
            
            // Legende
            if (chart.options.plugins && chart.options.plugins.legend) {
                chart.options.plugins.legend.labels.color = textColor;
            }
            
            chart.update();
        };
        
        // Alle Charts aktualisieren
        updateChartTheme(numericsChart);
        updateChartTheme(optimizationChart);
    }
    
    /**
     * Öffentliche API des VXORTMath-Moduls
     */
    return {
        init,
        
        // Erlaubt externen Zugriff auf bestimmte Funktionen
        updateTMathCharts,
        updateNumericsChart,
        updateOptimizationChart,
        
        // Nur für Testzwecke
        _getCharts: () => ({ numericsChart, optimizationChart })
    };
})();

// Beim Kernmodul registrieren, sobald das DOM geladen ist
document.addEventListener('DOMContentLoaded', () => {
    if (typeof VXORUtils !== 'undefined') {
        VXORUtils.registerModule('tmaths', VXORTMath);
    } else {
        console.error('VXORUtils nicht gefunden! TMath-Modul kann nicht registriert werden.');
    }
});
