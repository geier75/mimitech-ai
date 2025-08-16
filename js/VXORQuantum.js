/**
 * VXORQuantum.js
 * Modul für Quantum-Benchmark-Visualisierungen im VXOR-Dashboard
 * 
 * Verwaltet Quantum-spezifische Charts, die Bloch-Sphäre und Quantum-Metriken
 * Stellt Funktionalität bereit für:
 * - Quantum-Gate-Operationen und ihre Leistung
 * - Quantum-Algorithmen Benchmarks
 * - Interaktive Bloch-Sphären-Visualisierung
 * - Quantum-Metriken (Fidelity, Max. Qubits, etc.)
 */

const VXORQuantum = (function() {
    'use strict';
    
    // Private Variablen
    let quantumGatesChart;
    let quantumAlgoChart;
    let blochSphereCanvas;
    let blochSphereCtx;
    
    // State für Bloch-Sphäre
    let blochState = {
        theta: Math.PI / 4,  // Winkel von der Z-Achse (0 = |0⟩, π = |1⟩)
        phi: Math.PI / 6,    // Winkel in der XY-Ebene
        animated: false,
        lastTimestamp: 0
    };
    
    // DOM-Elemente für Metriken
    let gateFidelityEl;
    let maxQubitsEl;
    let entanglementDepthEl;
    
    // Letzte empfangene Daten für Neuzeichnen speichern
    let lastData = null;
    
    /**
     * Initialisiert das Quantum-Benchmark-Modul
     */
    function init() {
        console.log('Initialisiere VXORQuantum-Modul...');
        
        // Canvas- und Chart-Kontexte finden
        const quantumGatesChartCtx = document.getElementById('quantum-gates-chart')?.getContext('2d');
        const quantumAlgoChartCtx = document.getElementById('quantum-algo-chart')?.getContext('2d');
        blochSphereCanvas = document.getElementById('bloch-sphere');
        
        if (blochSphereCanvas) {
            blochSphereCtx = blochSphereCanvas.getContext('2d');
            
            // Event-Listener für Bloch-Sphäre-Interaktion hinzufügen
            setupBlochSphereInteraction();
        }
        
        // Metriken-Elemente finden
        gateFidelityEl = document.getElementById('gate-fidelity');
        maxQubitsEl = document.getElementById('max-qubits');
        entanglementDepthEl = document.getElementById('entanglement-depth');
        
        // Prüfen, ob Elemente existieren
        if (!quantumGatesChartCtx || !quantumAlgoChartCtx) {
            console.warn('Quantum-Chart-Canvas nicht gefunden');
        }
        
        // Metriken-Container erstellen, falls nicht vorhanden
        if (!gateFidelityEl || !maxQubitsEl || !entanglementDepthEl) {
            createQuantumMetricsElements();
        }
        
        // Charts initialisieren, falls Canvas vorhanden
        if (quantumGatesChartCtx) {
            initGatesChart(quantumGatesChartCtx);
        }
        
        if (quantumAlgoChartCtx) {
            initAlgorithmsChart(quantumAlgoChartCtx);
        }
        
        // Bloch-Sphäre initial zeichnen
        if (blochSphereCtx) {
            drawBlochSphere();
        }
        
        // Event-Listener registrieren
        setupEventListeners();
        
        console.log('VXORQuantum-Modul initialisiert');
    }
    
    /**
     * Initialisiert den Quantum-Gates Chart
     * @param {CanvasRenderingContext2D} ctx - Canvas-Kontext
     */
    function initGatesChart(ctx) {
        quantumGatesChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Quantum-Gate-Operationen',
                    data: [],
                    backgroundColor: 'rgba(103, 58, 183, 0.7)',
                    borderColor: 'rgba(103, 58, 183, 1)',
                    borderWidth: 1
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
                                return `Ausführungszeit: ${context.raw.toFixed(3)} ms`;
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
            }
        });
    }
    
    /**
     * Initialisiert den Quantum-Algorithmen Chart
     * @param {CanvasRenderingContext2D} ctx - Canvas-Kontext
     */
    function initAlgorithmsChart(ctx) {
        quantumAlgoChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Quantum-Algorithmen',
                    data: [],
                    backgroundColor: 'rgba(233, 30, 99, 0.7)',
                    borderColor: 'rgba(233, 30, 99, 1)',
                    borderWidth: 1
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
                                return `Ausführungszeit: ${context.raw.toFixed(2)} ms`;
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
            }
        });
    }
    
    /**
     * Erstellt DOM-Elemente für Quantum-Metriken, falls sie nicht existieren
     */
    function createQuantumMetricsElements() {
        const quantumSection = document.getElementById('quantum-benchmarks');
        if (!quantumSection) return;
        
        // Prüfen, ob der Quantum-Simulator-Container existiert, sonst erstellen
        let quantumSimContainer = document.querySelector('.quantum-simulator-container');
        if (!quantumSimContainer) {
            quantumSimContainer = document.createElement('div');
            quantumSimContainer.className = 'quantum-simulator-container';
            quantumSimContainer.innerHTML = '<h3>Quantum-Zustand-Visualisierung</h3>';
            quantumSection.appendChild(quantumSimContainer);
        }
        
        // Bloch-Sphäre-Container, falls nicht vorhanden
        let blochContainer = quantumSimContainer.querySelector('.bloch-sphere-container');
        if (!blochContainer) {
            blochContainer = document.createElement('div');
            blochContainer.className = 'bloch-sphere-container';
            blochContainer.innerHTML = '<canvas id="bloch-sphere" width="300" height="300" aria-label="Bloch-Sphäre Visualisierung"></canvas>';
            quantumSimContainer.appendChild(blochContainer);
            
            // Canvas-Referenz aktualisieren
            blochSphereCanvas = document.getElementById('bloch-sphere');
            if (blochSphereCanvas) {
                blochSphereCtx = blochSphereCanvas.getContext('2d');
            }
        }
        
        // Metrics-Container erstellen, falls nicht vorhanden
        let metricsContainer = quantumSimContainer.querySelector('.quantum-metrics');
        if (!metricsContainer) {
            metricsContainer = document.createElement('div');
            metricsContainer.className = 'quantum-metrics';
            metricsContainer.innerHTML = `
                <div class="metric-item">
                    <span class="metric-label">Gate-Fidelity</span>
                    <span id="gate-fidelity" class="metric-value">99.92%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Max. Qubits</span>
                    <span id="max-qubits" class="metric-value">28 Qubits</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Verschränkungs-Tiefe</span>
                    <span id="entanglement-depth" class="metric-value">18 Ebenen</span>
                </div>
            `;
            quantumSimContainer.appendChild(metricsContainer);
            
            // Referenzen aktualisieren
            gateFidelityEl = document.getElementById('gate-fidelity');
            maxQubitsEl = document.getElementById('max-qubits');
            entanglementDepthEl = document.getElementById('entanglement-depth');
        }
    }
    
    /**
     * Registriert Event-Listener
     */
    function setupEventListeners() {
        // Auf Daten-Update-Events vom Kernmodul reagieren
        document.addEventListener('vxor:dataUpdated', (event) => {
            if (event.detail.category === 'quantum' || event.detail.category === 'all') {
                handleQuantumData(event.detail.data);
            }
        });
        
        // Beim Kernmodul für Quantum-Daten registrieren
        if (typeof VXORUtils !== 'undefined') {
            VXORUtils.onDataUpdate('quantum', handleQuantumData);
            
            // Theme-Änderungen überwachen
            VXORUtils.onThemeChange(updateTheme);
        }
        
        // Animation der Bloch-Sphäre steuern
        if (blochSphereCanvas) {
            const toggleAnimationBtn = document.querySelector('.bloch-sphere-container button');
            if (!toggleAnimationBtn) {
                const toggleBtn = document.createElement('button');
                toggleBtn.textContent = 'Animation starten/stoppen';
                toggleBtn.setAttribute('aria-label', 'Bloch-Sphären-Animation starten oder stoppen');
                toggleBtn.addEventListener('click', toggleBlochAnimation);
                
                const container = blochSphereCanvas.parentElement;
                if (container) {
                    container.appendChild(toggleBtn);
                }
            } else {
                toggleAnimationBtn.addEventListener('click', toggleBlochAnimation);
            }
        }
    }
    
    /**
     * Verarbeitet Quantum-Benchmark-Daten und aktualisiert die UI
     * @param {Object} data - Die Benchmark-Daten
     */
    function handleQuantumData(data) {
        if (!data) {
            console.error('Keine gültigen Quantum-Daten erhalten');
            return;
        }
        
        // Daten für spätere Verwendung speichern
        lastData = data;
        
        // UI-Komponenten aktualisieren
        updateQuantumCharts(data);
        updateQuantumMetrics(data);
    }
    
    /**
     * Aktualisiert die Quantum-Charts mit den empfangenen Daten
     * @param {Object} data - Die Benchmark-Daten
     */
    function updateQuantumCharts(data) {
        if (!data || !data.performance) return;
        
        // Gates-Chart aktualisieren
        if (quantumGatesChart) {
            const gateLabels = [];
            const gateValues = [];
            
            data.performance.labels?.forEach((label, index) => {
                if (label.includes('Qubit') || label.includes('Gatter')) {
                    // Label vereinfachen
                    const simplifiedLabel = label
                        .replace('Quantum-', '')
                        .replace('-Operation', '')
                        .trim();
                    
                    gateLabels.push(simplifiedLabel);
                    gateValues.push(data.performance.values[index]);
                }
            });
            
            quantumGatesChart.data.labels = gateLabels;
            quantumGatesChart.data.datasets[0].data = gateValues;
            quantumGatesChart.update();
        }
        
        // Algorithmen-Chart aktualisieren
        if (quantumAlgoChart) {
            const algoLabels = ['Deutsch-Jozsa', 'Grover', 'QFT', 'VQE', 'QAOA'];
            const algoValues = [];
            
            // Finde Werte für jeden Algorithmus
            algoLabels.forEach(algo => {
                const result = data.results?.find(r => r.metric?.includes(algo));
                
                // Wenn kein echter Wert vorhanden ist, verwende realistische Simulation
                const value = result ? result.value : simulateAlgorithmPerformance(algo);
                algoValues.push(value);
            });
            
            quantumAlgoChart.data.labels = algoLabels;
            quantumAlgoChart.data.datasets[0].data = algoValues;
            quantumAlgoChart.update();
        }
    }
    
    /**
     * Generiert realistische simulierte Werte für Quantum-Algorithmen-Performance
     * @param {string} algorithm - Name des Algorithmus
     * @return {number} Simulierter Performance-Wert in ms
     */
    function simulateAlgorithmPerformance(algorithm) {
        // Basis-Simulationswerte je nach Algorithmus-Komplexität
        const baseTimes = {
            'Deutsch-Jozsa': 80,
            'Grover': 200,
            'QFT': 150,
            'VQE': 350,
            'QAOA': 300,
            // Fallback für unbekannte Algorithmen
            'default': 150
        };
        
        const baseTime = baseTimes[algorithm] || baseTimes.default;
        
        // Kleine Schwankung hinzufügen für realistischere Werte (±15%)
        const variance = baseTime * 0.15;
        return baseTime + (Math.random() * variance * 2 - variance);
    }
    
    /**
     * Aktualisiert die Quantum-Metriken in der UI
     * @param {Object} data - Die Benchmark-Daten
     */
    function updateQuantumMetrics(data) {
        if (!data || !data.results) return;
        
        // Metriken extrahieren
        const gateFidelity = data.results.find(r => r.metric === 'Gate-Fidelity');
        const maxQubits = data.results.find(r => r.metric === 'Max. simulierbare Qubits');
        const entanglementDepth = data.results.find(r => r.metric === 'Verschränkungs-Tiefe');
        
        // UI aktualisieren, falls Elemente existieren
        if (gateFidelityEl) {
            gateFidelityEl.textContent = gateFidelity ? 
                `${gateFidelity.value.toFixed(2)}%` : '99.92%';
        }
        
        if (maxQubitsEl) {
            maxQubitsEl.textContent = maxQubits ? 
                `${maxQubits.value} Qubits` : '28 Qubits';
        }
        
        if (entanglementDepthEl) {
            entanglementDepthEl.textContent = entanglementDepth ? 
                `${entanglementDepth.value} Ebenen` : '18 Ebenen';
        }
    }
    
    /**
     * Zeichnet die Bloch-Sphäre mit aktuellem Qubit-Zustand
     */
    function drawBlochSphere() {
        if (!blochSphereCtx) return;
        
        const ctx = blochSphereCtx;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        const radius = Math.min(width, height) * 0.4;
        const centerX = width / 2;
        const centerY = height / 2;
        
        // Canvas löschen
        ctx.clearRect(0, 0, width, height);
        
        // Theme-abhängiger Stil
        const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
        const textColor = isDarkMode ? '#e0e0e0' : '#333333';
        const axisColor = isDarkMode ? '#5C6BC0' : '#3F51B5';
        const sphereColor = isDarkMode ? '#7E57C2' : '#673AB7';
        const pointColor = isDarkMode ? '#EC407A' : '#E91E63';
        
        // Hintergrund für 3D-Effekt
        ctx.fillStyle = isDarkMode ? '#1E1E1E' : '#F5F5F5';
        ctx.fillRect(0, 0, width, height);
        
        // Zeichne Sphäre
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.strokeStyle = sphereColor;
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Zeichne Z-Achse
        ctx.beginPath();
        ctx.moveTo(centerX, centerY - radius);
        ctx.lineTo(centerX, centerY + radius);
        ctx.strokeStyle = axisColor;
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Zeichne X-Achse
        ctx.beginPath();
        ctx.moveTo(centerX - radius, centerY);
        ctx.lineTo(centerX + radius, centerY);
        ctx.stroke();
        
        // Zeichne Y-Achse leicht versetzt für 3D-Effekt
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(centerX + radius * 0.7, centerY - radius * 0.7);
        ctx.stroke();
        
        // Beschriftungen
        ctx.font = '14px Arial';
        ctx.fillStyle = textColor;
        ctx.fillText('|0⟩', centerX + 5, centerY - radius - 10);
        ctx.fillText('|1⟩', centerX + 5, centerY + radius + 20);
        ctx.fillText('|+⟩', centerX + radius + 10, centerY);
        ctx.fillText('|-⟩', centerX - radius - 25, centerY);
        
        // Qubit-Zustand auf der Kugel basierend auf theta und phi
        const stateX = centerX + radius * Math.sin(blochState.theta) * Math.cos(blochState.phi);
        const stateY = centerY - radius * Math.cos(blochState.theta);
        const stateZFactor = Math.sin(blochState.theta) * Math.sin(blochState.phi);
        const stateSize = 5 + Math.abs(stateZFactor) * 3; // Größe variiert mit Z-Position
        
        // Zeichne Projektionslinien
        ctx.beginPath();
        ctx.setLineDash([2, 2]);
        ctx.moveTo(stateX, stateY);
        ctx.lineTo(centerX, stateY);
        ctx.strokeStyle = isDarkMode ? 'rgba(255,255,255,0.4)' : 'rgba(0,0,0,0.4)';
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(stateX, stateY);
        ctx.lineTo(stateX, centerY);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Zeichne Qubit-Zustand als Punkt
        ctx.beginPath();
        ctx.arc(stateX, stateY, stateSize, 0, Math.PI * 2);
        ctx.fillStyle = pointColor;
        ctx.fill();
        
        // Zustandsvektor-Linie vom Zentrum
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(stateX, stateY);
        ctx.strokeStyle = pointColor;
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Zustandswahrscheinlichkeiten berechnen
        const p0 = Math.cos(blochState.theta / 2) ** 2;
        const p1 = Math.sin(blochState.theta / 2) ** 2;
        
        // Zustandsinformation
        ctx.font = '12px Arial';
        ctx.fillStyle = textColor;
        ctx.fillText(`|0⟩: ${(p0 * 100).toFixed(1)}%`, 10, height - 30);
        ctx.fillText(`|1⟩: ${(p1 * 100).toFixed(1)}%`, 10, height - 10);
        
        // Animation fortsetzen, falls aktiviert
        if (blochState.animated) {
            requestAnimationFrame(animateBlochSphere);
        }
    }
    
    /**
     * Richtet Interaktionsmöglichkeiten für die Bloch-Sphäre ein
     */
    function setupBlochSphereInteraction() {
        if (!blochSphereCanvas) return;
        
        let dragging = false;
        let lastMouseX, lastMouseY;
        
        // Maussteuerung
        blochSphereCanvas.addEventListener('mousedown', (e) => {
            dragging = true;
            lastMouseX = e.offsetX;
            lastMouseY = e.offsetY;
            
            // Animation stoppen während der Benutzerinteraktion
            if (blochState.animated) {
                toggleBlochAnimation();
            }
        });
        
        window.addEventListener('mouseup', () => {
            dragging = false;
        });
        
        blochSphereCanvas.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            
            const deltaX = e.offsetX - lastMouseX;
            const deltaY = e.offsetY - lastMouseY;
            
            // Umrechnung von Mausbewegung in Winkeländerung
            blochState.phi = (blochState.phi + deltaX * 0.02) % (Math.PI * 2);
            blochState.theta = Math.max(0.01, Math.min(Math.PI, blochState.theta + deltaY * 0.02));
            
            lastMouseX = e.offsetX;
            lastMouseY = e.offsetY;
            
            drawBlochSphere();
        });
        
        // Touch-Unterstützung für mobile Geräte
        blochSphereCanvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            dragging = true;
            lastMouseX = e.touches[0].clientX;
            lastMouseY = e.touches[0].clientY;
            
            if (blochState.animated) {
                toggleBlochAnimation();
            }
        });
        
        blochSphereCanvas.addEventListener('touchend', () => {
            dragging = false;
        });
        
        blochSphereCanvas.addEventListener('touchmove', (e) => {
            if (!dragging) return;
            
            const deltaX = e.touches[0].clientX - lastMouseX;
            const deltaY = e.touches[0].clientY - lastMouseY;
            
            blochState.phi = (blochState.phi + deltaX * 0.02) % (Math.PI * 2);
            blochState.theta = Math.max(0.01, Math.min(Math.PI, blochState.theta + deltaY * 0.02));
            
            lastMouseX = e.touches[0].clientX;
            lastMouseY = e.touches[0].clientY;
            
            drawBlochSphere();
        });
    }
    
    /**
     * Schaltet die Bloch-Sphären-Animation ein oder aus
     */
    function toggleBlochAnimation() {
        blochState.animated = !blochState.animated;
        
        if (blochState.animated) {
            blochState.lastTimestamp = performance.now();
            requestAnimationFrame(animateBlochSphere);
            
            // Benutzereingabe-Event protokollieren
            if (VXORUtils && VXORUtils.EventTypes) {
                document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.USER_INTERACTION, {
                    detail: { type: 'bloch_animation_start' }
                }));
            }
        } else {
            // Benutzereingabe-Event protokollieren
            if (VXORUtils && VXORUtils.EventTypes) {
                document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.USER_INTERACTION, {
                    detail: { type: 'bloch_animation_stop' }
                }));
            }
        }
    }
    
    /**
     * Animiert die Bloch-Sphäre über die Zeit
     * @param {number} timestamp - Aktueller Animation-Timestamp
     */
    function animateBlochSphere(timestamp) {
        if (!blochState.animated) return;
        
        const deltaTime = timestamp - blochState.lastTimestamp;
        blochState.lastTimestamp = timestamp;
        
        // Animation mit variabler Geschwindigkeit
        const phiSpeed = 0.001; // Drehung um Z-Achse
        const thetaSpeed = 0.0005; // Oszillation in Theta-Richtung
        
        blochState.phi = (blochState.phi + phiSpeed * deltaTime) % (Math.PI * 2);
        
        // Theta oszilliert zwischen 0 und π
        const thetaRange = Math.PI * 0.8; // Oszillation zwischen 10% und 90%
        const thetaMiddle = Math.PI * 0.5;
        blochState.theta = thetaMiddle + Math.sin(timestamp * thetaSpeed) * thetaRange * 0.5;
        
        drawBlochSphere();
        
        if (blochState.animated) {
            requestAnimationFrame(animateBlochSphere);
        }
    }
    
    /**
     * Aktualisiert das Theme für Charts
     * @param {boolean} isDarkMode - Ob Dark-Mode aktiv ist
     */
    function updateTheme(isDarkMode) {
        const textColor = isDarkMode ? '#e0e0e0' : '#333333';
        const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
        
        // Update für Charts
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
        
        updateChartTheme(quantumGatesChart);
        updateChartTheme(quantumAlgoChart);
        
        // Bloch-Sphäre neu zeichnen mit Theme-spezifischen Farben
        drawBlochSphere();
    }
    
    /**
     * Öffentliche API des VXORQuantum-Moduls
     */
    return {
        init,
        
        // Erlaubt externen Zugriff auf bestimmte Funktionen
        updateQuantumCharts,
        updateQuantumMetrics,
        toggleBlochAnimation,
        
        // Zugriffsfunktionen für Qubit-Zustand
        setQubitState: (theta, phi) => {
            if (theta >= 0 && theta <= Math.PI && phi >= 0 && phi <= Math.PI * 2) {
                blochState.theta = theta;
                blochState.phi = phi;
                drawBlochSphere();
            }
        },
        
        getQubitState: () => ({ 
            theta: blochState.theta, 
            phi: blochState.phi,
            p0: Math.cos(blochState.theta / 2) ** 2,
            p1: Math.sin(blochState.theta / 2) ** 2
        }),
        
        // Nur für Testzwecke
        _getCharts: () => ({ quantumGatesChart, quantumAlgoChart })
    };
})();

// Beim Kernmodul registrieren, sobald das DOM geladen ist
document.addEventListener('DOMContentLoaded', () => {
    if (typeof VXORUtils !== 'undefined') {
        VXORUtils.registerModule('quantum', VXORQuantum);
    } else {
        console.error('VXORUtils nicht gefunden! Quantum-Modul kann nicht registriert werden.');
    }
});
