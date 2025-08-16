/**
 * VXORSecurity.js
 * Modul für Security-Benchmark-Visualisierungen im VXOR-Dashboard
 * 
 * Verwaltet Security-spezifische Visualisierungen für:
 * - Sicherheitsanalysen und Audits
 * - Vulnerability-Erkennung und -Prävention
 * - Sicherheitsmatrizen (CVSS, CWE)
 */

const VXORSecurity = (function() {
    'use strict';
    
    // Private Variablen
    let secAnalysisChart;
    let vulnDetectionChart;
    let lastData = null;
    let securityMatrixData = null;
    
    // Chart-Kontexte
    let secAnalysisChartCtx;
    let vulnDetectionChartCtx;
    
    // DOM-Referenz auf Sicherheitsmatrix
    let securityMatrixContainer;
    
    // Sicherheitsdomänen für die Dropdown-Auswahl
    const securityDomains = [
        'Web',
        'Mobile',
        'Cloud',
        'IoT',
        'Netzwerk'
    ];
    
    // Verwundbarkeitstypen für die Analyse
    const vulnerabilityTypes = [
        'Injection',
        'XSS',
        'Access Control',
        'Cryptographic',
        'Configuration',
        'Authentication'
    ];
    
    // Ausgewählte Sicherheitsdomäne
    let selectedDomain = securityDomains[0];
    
    /**
     * Initialisiert das Security-Benchmark-Modul
     */
    function init() {
        console.log('Initialisiere VXORSecurity-Modul...');
        
        // Chart-Kontexte und DOM-Elemente finden
        secAnalysisChartCtx = document.getElementById('sec-analysis-chart')?.getContext('2d');
        vulnDetectionChartCtx = document.getElementById('vuln-detection-chart')?.getContext('2d');
        securityMatrixContainer = document.getElementById('security-matrix');
        
        if (!secAnalysisChartCtx || !vulnDetectionChartCtx) {
            console.warn('Security-Charts nicht gefunden');
        }
        
        if (!securityMatrixContainer) {
            console.warn('Security-Matrix-Container nicht gefunden');
        }
        
        // Charts und Tabellen initialisieren
        initCharts();
        createDomainSelector();
        
        // Event-Listener registrieren
        setupEventListeners();
        
        // Leere Matrix für den Anfang erstellen
        updateSecurityMatrix();
        
        console.log('VXORSecurity-Modul initialisiert');
    }
    
    /**
     * Initialisiert die Security-Charts
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
                            return `${context.dataset.label}: ${context.raw.toFixed(1)}`;
                        }
                    }
                }
            }
        };
        
        // Sicherheitsanalyse-Chart
        if (secAnalysisChartCtx) {
            secAnalysisChart = new Chart(secAnalysisChartCtx, {
                type: 'radar',
                data: {
                    labels: vulnerabilityTypes,
                    datasets: [{
                        label: 'Erkennungsrate',
                        data: [],
                        backgroundColor: 'rgba(54, 162, 235, 0.3)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 2,
                        pointBackgroundColor: 'rgba(54, 162, 235, 1)',
                        pointRadius: 4
                    }, {
                        label: 'Präventionsrate',
                        data: [],
                        backgroundColor: 'rgba(255, 99, 132, 0.3)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 2,
                        pointBackgroundColor: 'rgba(255, 99, 132, 1)',
                        pointRadius: 4
                    }]
                },
                options: {
                    ...chartOptions,
                    scales: {
                        r: {
                            angleLines: {
                                display: true
                            },
                            suggestedMin: 0,
                            suggestedMax: 10,
                            ticks: {
                                stepSize: 2
                            }
                        }
                    }
                }
            });
        }
        
        // Vulnerability-Erkennung-Chart
        if (vulnDetectionChartCtx) {
            vulnDetectionChart = new Chart(vulnDetectionChartCtx, {
                type: 'horizontalBar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Erkennungseffizienz',
                        data: [],
                        backgroundColor: 'rgba(75, 192, 192, 0.7)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }, {
                        label: 'False Positives',
                        data: [],
                        backgroundColor: 'rgba(255, 206, 86, 0.7)',
                        borderColor: 'rgba(255, 206, 86, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    ...chartOptions,
                    indexAxis: 'y',
                    scales: {
                        x: {
                            beginAtZero: true,
                            max: 10,
                            title: {
                                display: true,
                                text: 'Bewertung (0-10)'
                            }
                        }
                    }
                }
            });
        }
    }
    
    /**
     * Erstellt einen Domänen-Selektor
     */
    function createDomainSelector() {
        const securitySection = document.getElementById('security-benchmarks');
        if (!securitySection) return;
        
        // Prüfen, ob der Selektor bereits existiert
        if (document.getElementById('security-domain-selector')) return;
        
        // Container für den Selektor erstellen
        const selectorContainer = document.createElement('div');
        selectorContainer.className = 'domain-selector-container';
        selectorContainer.innerHTML = `
            <label for="security-domain-selector">Sicherheits-Domäne:</label>
            <select id="security-domain-selector" aria-label="Sicherheits-Domäne auswählen">
                ${securityDomains.map(domain => 
                    `<option value="${domain}">${domain}</option>`
                ).join('')}
            </select>
        `;
        
        // Vor dem ersten Chart einfügen
        const firstCard = securitySection.querySelector('.benchmark-card');
        if (firstCard) {
            securitySection.insertBefore(selectorContainer, firstCard);
        } else {
            const cards = securitySection.querySelector('.benchmark-cards');
            if (cards) {
                securitySection.insertBefore(selectorContainer, cards);
            } else {
                securitySection.appendChild(selectorContainer);
            }
        }
        
        // Event-Listener hinzufügen
        const selector = document.getElementById('security-domain-selector');
        if (selector) {
            selector.addEventListener('change', handleDomainChange);
        }
    }
    
    /**
     * Registriert Event-Listener
     */
    function setupEventListeners() {
        // Beim Kernmodul für Security-Daten registrieren
        if (typeof VXORUtils !== 'undefined') {
            VXORUtils.onDataUpdate('security', handleSecurityData);
            VXORUtils.onThemeChange(updateTheme);
        } else {
            console.warn('VXORUtils nicht gefunden, eingeschränkte Funktionalität');
        }
        
        // Event-Listener für den Domänen-Selektor
        document.addEventListener('change', event => {
            if (event.target.id === 'security-domain-selector') {
                handleDomainChange(event);
            }
        });
    }
    
    /**
     * Behandelt Domänen-Änderungen
     * @param {Event} event - Change-Event
     */
    function handleDomainChange(event) {
        selectedDomain = event.target.value;
        
        // Benutzerinteraktion protokollieren
        if (VXORUtils && VXORUtils.EventTypes) {
            document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.USER_INTERACTION, {
                detail: {
                    type: 'security_domain_change',
                    value: selectedDomain
                }
            }));
        }
        
        // Daten aktualisieren, falls vorhanden
        if (lastData) {
            updateSecurityCharts(lastData);
            updateSecurityMatrix(lastData);
        }
    }
    
    /**
     * Verarbeitet Security-Benchmark-Daten und aktualisiert die UI
     * @param {Object} data - Die Benchmark-Daten
     */
    function handleSecurityData(data) {
        if (!data) {
            console.error('Keine gültigen Security-Daten erhalten');
            return;
        }
        
        // Daten für spätere Verwendung speichern
        lastData = data;
        
        // UI-Komponenten aktualisieren
        updateSecurityCharts(data);
        updateSecurityMatrix(data);
    }
    
    /**
     * Aktualisiert die Security-Charts
     * @param {Object} data - Die Benchmark-Daten
     */
    function updateSecurityCharts(data) {
        // Sicherheitsanalyse-Chart aktualisieren
        updateSecAnalysisChart(data);
        
        // Vulnerability-Erkennung-Chart aktualisieren
        updateVulnDetectionChart(data);
    }
    
    /**
     * Aktualisiert das Sicherheitsanalyse-Chart
     * @param {Object} [data] - Die Benchmark-Daten (optional)
     */
    function updateSecAnalysisChart(data) {
        if (!secAnalysisChart) return;
        
        // Erkennungs- und Präventionsdaten
        const detectionData = [];
        const preventionData = [];
        
        // Für jeden Verwundbarkeitstyp Daten finden oder simulieren
        vulnerabilityTypes.forEach(vulnType => {
            let detectionValue;
            let preventionValue;
            
            if (data?.results) {
                // Daten aus Ergebnissen extrahieren
                const detectionResult = data.results.find(r => 
                    r.metric.includes('Erkennungsrate') && 
                    r.metric.includes(selectedDomain) && 
                    r.metric.includes(vulnType)
                );
                
                const preventionResult = data.results.find(r => 
                    r.metric.includes('Präventionsrate') && 
                    r.metric.includes(selectedDomain) && 
                    r.metric.includes(vulnType)
                );
                
                detectionValue = detectionResult ? detectionResult.value : generateSecurityRating(vulnType, 'detection');
                preventionValue = preventionResult ? preventionResult.value : generateSecurityRating(vulnType, 'prevention');
            } else {
                // Daten simulieren
                detectionValue = generateSecurityRating(vulnType, 'detection');
                preventionValue = generateSecurityRating(vulnType, 'prevention');
            }
            
            detectionData.push(detectionValue);
            preventionData.push(preventionValue);
        });
        
        // Chart aktualisieren
        secAnalysisChart.data.datasets[0].data = detectionData;
        secAnalysisChart.data.datasets[0].label = `Erkennungsrate (${selectedDomain})`;
        
        secAnalysisChart.data.datasets[1].data = preventionData;
        secAnalysisChart.data.datasets[1].label = `Präventionsrate (${selectedDomain})`;
        
        secAnalysisChart.update();
    }
    
    /**
     * Aktualisiert das Vulnerability-Erkennung-Chart
     * @param {Object} [data] - Die Benchmark-Daten (optional)
     */
    function updateVulnDetectionChart(data) {
        if (!vulnDetectionChart) return;
        
        // Werkzeug-Typen für die Y-Achse
        const toolTypes = [
            'Statische Analyse',
            'Dynamische Analyse',
            'Fuzzing',
            'Penetration Testing',
            'Compositional Analysis'
        ];
        
        // Effizienz- und False-Positive-Daten
        const efficiencyData = [];
        const falsePositiveData = [];
        
        // Für jeden Werkzeug-Typ Daten finden oder simulieren
        toolTypes.forEach(toolType => {
            let efficiencyValue;
            let falsePositiveValue;
            
            if (data?.results) {
                // Daten aus Ergebnissen extrahieren
                const efficiencyResult = data.results.find(r => 
                    r.metric.includes('Erkennungseffizienz') && 
                    r.metric.includes(selectedDomain) && 
                    r.metric.includes(toolType)
                );
                
                const falsePositiveResult = data.results.find(r => 
                    r.metric.includes('False Positives') && 
                    r.metric.includes(selectedDomain) && 
                    r.metric.includes(toolType)
                );
                
                efficiencyValue = efficiencyResult ? efficiencyResult.value : generateToolRating(toolType, 'efficiency');
                falsePositiveValue = falsePositiveResult ? falsePositiveResult.value : generateToolRating(toolType, 'falsePositive');
            } else {
                // Daten simulieren
                efficiencyValue = generateToolRating(toolType, 'efficiency');
                falsePositiveValue = generateToolRating(toolType, 'falsePositive');
            }
            
            efficiencyData.push(efficiencyValue);
            falsePositiveData.push(falsePositiveValue);
        });
        
        // Chart aktualisieren
        vulnDetectionChart.data.labels = toolTypes;
        vulnDetectionChart.data.datasets[0].data = efficiencyData;
        vulnDetectionChart.data.datasets[0].label = `Erkennungseffizienz (${selectedDomain})`;
        
        vulnDetectionChart.data.datasets[1].data = falsePositiveData;
        vulnDetectionChart.data.datasets[1].label = `False Positives (${selectedDomain})`;
        
        vulnDetectionChart.update();
    }
    
    /**
     * Aktualisiert die Sicherheitsmatrix
     * @param {Object} [data] - Die Benchmark-Daten (optional)
     */
    function updateSecurityMatrix(data) {
        if (!securityMatrixContainer) return;
        
        // Sicherheitsmatrix-Daten extrahieren oder simulieren
        let matrixData;
        
        if (data?.securityMatrix && data.securityMatrix[selectedDomain]) {
            matrixData = data.securityMatrix[selectedDomain];
        } else {
            // Simulierte CVSS-Matrix generieren
            matrixData = generateSecurityMatrix();
        }
        
        // Matrix-Darstellung für das DOM erstellen
        createSecurityMatrixDisplay(matrixData);
    }
    
    /**
     * Erstellt die visuelle Darstellung der Sicherheitsmatrix
     * @param {Object} matrixData - Die Daten für die Sicherheitsmatrix
     */
    function createSecurityMatrixDisplay(matrixData) {
        // Container leeren
        securityMatrixContainer.innerHTML = '';
        
        // Beschreibungstext für Screenreader
        const srDescription = document.createElement('p');
        srDescription.id = 'security-matrix-description';
        srDescription.className = 'sr-only';
        srDescription.textContent = `Diese Matrix zeigt die Anzahl der erkannten Sicherheitsprobleme in ${selectedDomain}-Systemen, 
                                    kategorisiert nach Risikobereich (Zeilen) und Schweregrad (Spalten).`;
        securityMatrixContainer.appendChild(srDescription);
        
        // Tabelle erstellen
        const table = document.createElement('table');
        table.className = 'security-matrix-table';
        table.setAttribute('aria-label', `Sicherheitsmatrix für ${selectedDomain}`);
        table.setAttribute('aria-describedby', 'security-matrix-description');
        
        // Header-Zeile
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        
        // Ecke oben links
        const cornerHeader = document.createElement('th');
        cornerHeader.textContent = 'Risikobereich';
        headerRow.appendChild(cornerHeader);
        
        // Spaltenüberschriften (Schweregrade)
        ['Kritisch', 'Hoch', 'Mittel', 'Niedrig'].forEach(severity => {
            const th = document.createElement('th');
            th.textContent = severity;
            th.scope = 'col';
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // Table Body
        const tbody = document.createElement('tbody');
        
        // Zeilen für jeden Bereich erstellen
        ['Authentifizierung', 'Zugriffskontrolle', 'Datenvalidierung', 'Kryptographie', 'Konfiguration'].forEach((area, index) => {
            const row = document.createElement('tr');
            
            // Zeilenüberschrift
            const rowHeader = document.createElement('th');
            rowHeader.textContent = area;
            rowHeader.scope = 'row';
            row.appendChild(rowHeader);
            
            // Zellen für jeden Schweregrad
            ['Critical', 'High', 'Medium', 'Low'].forEach((severity, severityIndex) => {
                const cell = document.createElement('td');
                const value = matrixData[area]?.[severity] || 0;
                
                cell.textContent = value;
                
                // Erhöhte Barrierefreiheit durch zusätzliches aria-label
                const sevNames = ['Kritisch', 'Hoch', 'Mittel', 'Niedrig'];
                cell.setAttribute('aria-label', `${area}, ${sevNames[severityIndex]}: ${value} Schwachstellen`);
                
                // Farbe je nach Wert
                if (value > 0) {
                    const intensity = Math.min(1, value / 10);
                    const colorClass = getColorClassForSeverity(severity, intensity);
                    cell.className = colorClass;
                }
                
                row.appendChild(cell);
            });
            
            tbody.appendChild(row);
        });
        
        table.appendChild(tbody);
        securityMatrixContainer.appendChild(table);
        
        // Live-Region für dynamische Updates hinzufügen
        const liveUpdate = document.createElement('div');
        liveUpdate.id = 'security-matrix-update';
        liveUpdate.setAttribute('aria-live', 'polite');
        liveUpdate.className = 'sr-only';
        liveUpdate.textContent = `Sicherheitsmatrix für ${selectedDomain} wurde aktualisiert.`;
        securityMatrixContainer.appendChild(liveUpdate);
        
        // Legende hinzufügen
        addMatrixLegend();
    }
    
    /**
     * Fügt eine Legende zur Sicherheitsmatrix hinzu
     */
    function addMatrixLegend() {
        const legend = document.createElement('div');
        legend.className = 'security-matrix-legend';
        legend.innerHTML = `
            <p>Anzahl der erkannten Schwachstellen pro Kategorie:</p>
            <ul class="legend-items">
                <li><span class="legend-color critical-high"></span> Kritisch: Sofortige Maßnahmen erforderlich</li>
                <li><span class="legend-color high-medium"></span> Hoch: Hohe Priorität</li>
                <li><span class="legend-color medium-low"></span> Mittel: Moderate Priorität</li>
                <li><span class="legend-color low-minimal"></span> Niedrig: Niedrige Priorität</li>
            </ul>
        `;
        
        securityMatrixContainer.appendChild(legend);
    }
    
    /**
     * Generiert eine Bewertung für einen Sicherheitsaspekt
     * @param {string} vulnType - Der Verwundbarkeitstyp
     * @param {string} aspect - Der zu bewertende Aspekt ('detection' oder 'prevention')
     * @returns {number} Die generierte Bewertung (0-10)
     */
    function generateSecurityRating(vulnType, aspect) {
        // Basiswerte je nach Verwundbarkeitstyp und Domäne
        let baseValue;
        
        // Verschiedene Domänen haben unterschiedliche Stärken/Schwächen
        switch (selectedDomain) {
            case 'Web':
                baseValue = (vulnType === 'XSS' || vulnType === 'Injection') ? 7.5 : 6.5;
                break;
            case 'Mobile':
                baseValue = (vulnType === 'Authentication' || vulnType === 'Cryptographic') ? 7.0 : 6.0;
                break;
            case 'Cloud':
                baseValue = (vulnType === 'Configuration' || vulnType === 'Access Control') ? 8.0 : 6.5;
                break;
            case 'IoT':
                baseValue = (vulnType === 'Cryptographic' || vulnType === 'Configuration') ? 5.5 : 5.0;
                break;
            case 'Netzwerk':
                baseValue = (vulnType === 'Access Control' || vulnType === 'Injection') ? 7.0 : 6.0;
                break;
            default:
                baseValue = 6.0;
                break;
        }
        
        // Erkennungsrate ist oft besser als Präventionsrate
        if (aspect === 'prevention') {
            baseValue *= 0.85; // Prävention ist schwieriger als Erkennung
        }
        
        // Zufällige Variation hinzufügen
        return Math.min(10, Math.max(1, baseValue * (1 + (Math.random() * 0.2 - 0.1))));
    }
    
    /**
     * Generiert eine Bewertung für ein Sicherheitswerkzeug
     * @param {string} toolType - Der Werkzeug-Typ
     * @param {string} aspect - Der zu bewertende Aspekt ('efficiency' oder 'falsePositive')
     * @returns {number} Die generierte Bewertung (0-10)
     */
    function generateToolRating(toolType, aspect) {
        // Basiswerte je nach Werkzeug und Domäne
        let baseValue;
        
        switch (toolType) {
            case 'Statische Analyse':
                baseValue = 7.5;
                break;
            case 'Dynamische Analyse':
                baseValue = 8.0;
                break;
            case 'Fuzzing':
                baseValue = 6.5;
                break;
            case 'Penetration Testing':
                baseValue = 9.0;
                break;
            case 'Compositional Analysis':
                baseValue = 7.0;
                break;
            default:
                baseValue = 7.0;
                break;
        }
        
        // Domänen-spezifische Anpassungen
        switch (selectedDomain) {
            case 'Web':
                baseValue *= (toolType === 'Dynamische Analyse' || toolType === 'Penetration Testing') ? 1.1 : 1.0;
                break;
            case 'Mobile':
                baseValue *= (toolType === 'Statische Analyse') ? 1.1 : 1.0;
                break;
            case 'Cloud':
                baseValue *= (toolType === 'Compositional Analysis') ? 1.1 : 1.0;
                break;
            case 'IoT':
                baseValue *= (toolType === 'Fuzzing') ? 1.1 : 1.0;
                break;
            case 'Netzwerk':
                baseValue *= (toolType === 'Penetration Testing') ? 1.1 : 1.0;
                break;
        }
        
        // False-Positives sind umgekehrt zur Effizienz (niedriger ist besser)
        if (aspect === 'falsePositive') {
            baseValue = 10 - ((baseValue - 1) * 0.7); // Inverse Beziehung, aber nicht komplett umgekehrt
        }
        
        // Zufällige Variation hinzufügen
        return Math.min(10, Math.max(1, baseValue * (1 + (Math.random() * 0.2 - 0.1))));
    }
    
    /**
     * Generiert eine simulierte Sicherheitsmatrix
     * @returns {Object} Die generierte Sicherheitsmatrix
     */
    function generateSecurityMatrix() {
        const matrix = {};
        
        // Für jeden Bereich eine Unterdaten-Struktur erstellen
        ['Authentifizierung', 'Zugriffskontrolle', 'Datenvalidierung', 'Kryptographie', 'Konfiguration'].forEach(area => {
            matrix[area] = {
                Critical: 0,
                High: 0,
                Medium: 0,
                Low: 0
            };
            
            // Werte basierend auf Domäne und Bereich generieren
            switch (selectedDomain) {
                case 'Web':
                    if (area === 'Datenvalidierung') {
                        matrix[area].Critical = Math.round(Math.random() * 3 + 1);
                        matrix[area].High = Math.round(Math.random() * 5 + 3);
                    } else if (area === 'Authentifizierung') {
                        matrix[area].High = Math.round(Math.random() * 4 + 2);
                        matrix[area].Medium = Math.round(Math.random() * 6 + 2);
                    }
                    break;
                    
                case 'Mobile':
                    if (area === 'Kryptographie') {
                        matrix[area].High = Math.round(Math.random() * 3 + 1);
                        matrix[area].Medium = Math.round(Math.random() * 5 + 2);
                    } else if (area === 'Datenvalidierung') {
                        matrix[area].Medium = Math.round(Math.random() * 4 + 2);
                        matrix[area].Low = Math.round(Math.random() * 7 + 3);
                    }
                    break;
                    
                case 'Cloud':
                    if (area === 'Konfiguration') {
                        matrix[area].Critical = Math.round(Math.random() * 2 + 1);
                        matrix[area].High = Math.round(Math.random() * 4 + 2);
                    } else if (area === 'Zugriffskontrolle') {
                        matrix[area].High = Math.round(Math.random() * 3 + 1);
                        matrix[area].Medium = Math.round(Math.random() * 5 + 2);
                    }
                    break;
                    
                case 'IoT':
                    if (area === 'Kryptographie') {
                        matrix[area].Critical = Math.round(Math.random() * 3 + 2);
                        matrix[area].High = Math.round(Math.random() * 6 + 3);
                    } else if (area === 'Konfiguration') {
                        matrix[area].High = Math.round(Math.random() * 5 + 3);
                        matrix[area].Medium = Math.round(Math.random() * 7 + 4);
                    }
                    break;
                    
                case 'Netzwerk':
                    if (area === 'Zugriffskontrolle') {
                        matrix[area].Critical = Math.round(Math.random() * 2 + 1);
                        matrix[area].High = Math.round(Math.random() * 4 + 2);
                    } else if (area === 'Konfiguration') {
                        matrix[area].High = Math.round(Math.random() * 3 + 2);
                        matrix[area].Medium = Math.round(Math.random() * 5 + 3);
                    }
                    break;
            }
            
            // Grundlegende zufällige Werte für alle anderen Kombinationen
            if (matrix[area].Critical === 0) matrix[area].Critical = Math.round(Math.random());
            if (matrix[area].High === 0) matrix[area].High = Math.round(Math.random() * 2);
            if (matrix[area].Medium === 0) matrix[area].Medium = Math.round(Math.random() * 3 + 1);
            if (matrix[area].Low === 0) matrix[area].Low = Math.round(Math.random() * 4 + 2);
        });
        
        return matrix;
    }
    
    /**
     * Liefert die CSS-Klasse für eine Zelle basierend auf Schweregrad und Intensität
     * @param {string} severity - Der Schweregrad
     * @param {number} intensity - Die Intensität (0-1)
     * @returns {string} Die CSS-Klasse
     */
    function getColorClassForSeverity(severity, intensity) {
        // Intensitäts-Level (niedrig, mittel, hoch)
        const intensityLevel = intensity < 0.33 ? '-low' : 
                             (intensity < 0.66 ? '-medium' : '-high');
        
        // Färbung je nach Schweregrad
        switch (severity) {
            case 'Critical': return 'critical' + intensityLevel;
            case 'High': return 'high' + intensityLevel;
            case 'Medium': return 'medium' + intensityLevel;
            case 'Low': return 'low' + intensityLevel;
            default: return '';
        }
    }
    
    /**
     * Aktualisiert das Theme für Charts und Matrix
     * @param {boolean} isDarkMode - Ob Dark-Mode aktiv ist
     */
    function updateTheme(isDarkMode) {
        const textColor = isDarkMode ? '#e0e0e0' : '#333333';
        const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
        
        const updateChartTheme = (chart) => {
            if (!chart) return;
            
            // Achsenbeschriftungen für normale Charts
            Object.values(chart.options.scales || {}).forEach(scale => {
                if (scale.ticks) scale.ticks.color = textColor;
                if (scale.grid) scale.grid.color = gridColor;
                if (scale.title) scale.title.color = textColor;
            });
            
            // Radar-Chart-spezifische Einstellungen
            if (chart.options.scales && chart.options.scales.r) {
                chart.options.scales.r.pointLabels = chart.options.scales.r.pointLabels || {};
                chart.options.scales.r.pointLabels.color = textColor;
                chart.options.scales.r.angleLines = chart.options.scales.r.angleLines || {};
                chart.options.scales.r.angleLines.color = gridColor;
                chart.options.scales.r.grid = chart.options.scales.r.grid || {};
                chart.options.scales.r.grid.color = gridColor;
            }
            
            // Legende
            if (chart.options.plugins && chart.options.plugins.legend) {
                chart.options.plugins.legend.labels = chart.options.plugins.legend.labels || {};
                chart.options.plugins.legend.labels.color = textColor;
            }
            
            chart.update();
        };
        
        // Beide Charts aktualisieren
        updateChartTheme(secAnalysisChart);
        updateChartTheme(vulnDetectionChart);
        
        // Matrix aktualisieren, falls Daten vorhanden
        if (securityMatrixContainer && securityMatrixContainer.querySelector('.security-matrix-table')) {
            // Matrix-Layout wird über CSS-Variablen gesteuert, kein expliziter Update nötig
        }
    }
    
    /**
     * Öffentliche API des VXORSecurity-Moduls
     */
    return {
        init,
        
        // Erlaubt externen Zugriff auf bestimmte Funktionen
        updateSecurityCharts,
        handleSecurityData,
        
        // Erlaubt das Setzen der Domäne von außen
        setDomain: function(domain) {
            if (securityDomains.includes(domain)) {
                selectedDomain = domain;
                const selector = document.getElementById('security-domain-selector');
                if (selector) selector.value = domain;
                
                if (lastData) {
                    updateSecurityCharts(lastData);
                    updateSecurityMatrix(lastData);
                }
            }
        },
        
        // Nur für Testzwecke
        _getCharts: () => ({ secAnalysisChart, vulnDetectionChart })
    };
})();

// Beim Kernmodul registrieren, sobald das DOM geladen ist
document.addEventListener('DOMContentLoaded', () => {
    if (typeof VXORUtils !== 'undefined') {
        VXORUtils.registerModule('security', VXORSecurity);
    } else {
        console.error('VXORUtils nicht gefunden! Security-Modul kann nicht registriert werden.');
    }
});
