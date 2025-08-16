/**
 * VXORSWEBench.js
 * Modul für Software Engineering Benchmark-Visualisierungen im VXOR-Dashboard
 * 
 * Verwaltet SWE-Bench-spezifische Visualisierungen für:
 * - Code-Generierungsqualität und -geschwindigkeit
 * - Bugfixing-Performance und -erfolgsrate
 * - Leistungsvergleich verschiedener Aufgabentypen
 */

const VXORSWEBench = (function() {
    'use strict';
    
    // Private Variablen
    let codeGenChart;
    let bugfixChart;
    let lastData = null;
    
    // Chart-Kontexte
    let codeGenChartCtx;
    let bugfixChartCtx;
    
    // Referenz auf die Metrik-Tabelle
    let metricsTableBody;
    
    // Task-Kategorien für die Dropdown-Auswahl
    const taskCategories = [
        'Algorithmus-Implementation',
        'Feature-Erweiterung',
        'Bugfixing',
        'Refactoring',
        'Dokumentation'
    ];
    
    // Programmiersprachen für die Auswahl
    const programmingLanguages = [
        'Python',
        'JavaScript',
        'Java',
        'C++',
        'Rust'
    ];
    
    // Aktuell ausgewählte Kategorie und Sprache
    let selectedTaskCategory = taskCategories[0];
    let selectedLanguage = programmingLanguages[0];
    
    /**
     * Initialisiert das SWE-Bench-Modul
     */
    function init() {
        console.log('Initialisiere VXORSWEBench-Modul...');
        
        // Chart-Kontexte und DOM-Elemente finden
        codeGenChartCtx = document.getElementById('code-gen-chart')?.getContext('2d');
        bugfixChartCtx = document.getElementById('bugfix-chart')?.getContext('2d');
        metricsTableBody = document.querySelector('#swe-metrics tbody');
        
        if (!codeGenChartCtx || !bugfixChartCtx) {
            console.warn('SWE-Bench-Charts nicht gefunden');
        }
        
        // Charts und Tabellen initialisieren
        initCharts();
        createSelectors();
        
        // Event-Listener registrieren
        setupEventListeners();
        
        console.log('VXORSWEBench-Modul initialisiert');
    }
    
    /**
     * Initialisiert die SWE-Bench-Charts
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
                            const dataset = context.dataset;
                            const value = context.raw;
                            
                            if (dataset.yAxisID === 'y') {
                                return `Zeit: ${value.toFixed(1)} Sek`;
                            } else if (dataset.yAxisID === 'y1') {
                                return `Erfolgsrate: ${(value * 100).toFixed(1)}%`;
                            } else {
                                return `${dataset.label}: ${value}`;
                            }
                        }
                    }
                }
            }
        };
        
        // Code-Generierungs-Chart
        if (codeGenChartCtx) {
            codeGenChart = new Chart(codeGenChartCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Generierungszeit',
                        data: [],
                        backgroundColor: 'rgba(54, 162, 235, 0.7)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1,
                        yAxisID: 'y'
                    }, {
                        label: 'Code-Qualität (0-10)',
                        data: [],
                        backgroundColor: 'rgba(75, 192, 192, 0.7)',
                        borderColor: 'rgba(75, 192, 192, 1)',
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
                                text: 'Zeit (Sek)'
                            },
                            grid: {
                                display: true
                            }
                        },
                        y1: {
                            beginAtZero: true,
                            position: 'right',
                            min: 0,
                            max: 10,
                            title: {
                                display: true,
                                text: 'Qualität (0-10)'
                            },
                            grid: {
                                display: false
                            }
                        }
                    }
                }
            });
        }
        
        // Bugfixing-Chart
        if (bugfixChartCtx) {
            bugfixChart = new Chart(bugfixChartCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Zeit bis Fix',
                        data: [],
                        backgroundColor: 'rgba(255, 99, 132, 0.7)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1,
                        yAxisID: 'y'
                    }, {
                        label: 'Erfolgsrate',
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
                                text: 'Zeit (Sek)'
                            },
                            grid: {
                                display: true
                            }
                        },
                        y1: {
                            beginAtZero: true,
                            position: 'right',
                            min: 0,
                            max: 1,
                            title: {
                                display: true,
                                text: 'Erfolgsrate'
                            },
                            ticks: {
                                callback: function(value) {
                                    return (value * 100) + '%';
                                }
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
     * Erstellt Selektoren für Aufgabentyp und Programmiersprache mit verbesserter 
     * Tastaturzugänglichkeit und ARIA-Unterstützung
     */
    function createSelectors() {
        const sweSection = document.getElementById('swe-benchmarks');
        if (!sweSection) return;
        
        // Container für Selektoren erstellen, falls nicht vorhanden
        let selectorContainer = sweSection.querySelector('.selectors-container');
        if (!selectorContainer) {
            selectorContainer = document.createElement('div');
            selectorContainer.className = 'selectors-container';
            selectorContainer.setAttribute('role', 'group');
            selectorContainer.setAttribute('aria-label', 'Filter für Software Engineering Benchmarks');
            
            // Hinzufügen einer Beschreibung für Screenreader
            const srDescription = document.createElement('p');
            srDescription.id = 'swe-selectors-description';
            srDescription.className = 'sr-only';
            srDescription.textContent = 'Diese Filteroptionen passen die Benchmark-Anzeige an den gewählten Aufgabentyp und die Programmiersprache an.';
            selectorContainer.appendChild(srDescription);
            
            // HTML für beide Selektoren
            // Erste Selektor-Gruppe: Aufgabentyp
            const taskGroup = document.createElement('div');
            taskGroup.className = 'selector-group';
            
            const taskLabel = document.createElement('label');
            taskLabel.setAttribute('for', 'task-category-selector');
            taskLabel.textContent = 'Aufgabentyp:';
            taskGroup.appendChild(taskLabel);
            
            const taskSelect = document.createElement('select');
            taskSelect.id = 'task-category-selector';
            taskSelect.setAttribute('aria-label', 'Aufgabentyp auswählen');
            taskSelect.setAttribute('aria-describedby', 'swe-selectors-description');
            
            // Optionen hinzufügen
            taskCategories.forEach(task => {
                const option = document.createElement('option');
                option.value = task;
                option.textContent = task;
                taskSelect.appendChild(option);
            });
            
            taskGroup.appendChild(taskSelect);
            selectorContainer.appendChild(taskGroup);
            
            // Zweite Selektor-Gruppe: Programmiersprache
            const langGroup = document.createElement('div');
            langGroup.className = 'selector-group';
            
            const langLabel = document.createElement('label');
            langLabel.setAttribute('for', 'programming-language-selector');
            langLabel.textContent = 'Programmiersprache:';
            langGroup.appendChild(langLabel);
            
            const langSelect = document.createElement('select');
            langSelect.id = 'programming-language-selector';
            langSelect.setAttribute('aria-label', 'Programmiersprache auswählen');
            langSelect.setAttribute('aria-describedby', 'swe-selectors-description');
            
            // Optionen hinzufügen
            programmingLanguages.forEach(lang => {
                const option = document.createElement('option');
                option.value = lang;
                option.textContent = lang;
                langSelect.appendChild(option);
            });
            
            langGroup.appendChild(langSelect);
            selectorContainer.appendChild(langGroup);
            
            // Live-Region für Ankündigungen von Filteränderungen
            const liveUpdate = document.createElement('div');
            liveUpdate.id = 'swe-filter-update';
            liveUpdate.setAttribute('aria-live', 'polite');
            liveUpdate.className = 'sr-only';
            selectorContainer.appendChild(liveUpdate);
            
            // Selektoren vor dem ersten Chart einfügen
            const firstCard = sweSection.querySelector('.benchmark-card');
            if (firstCard) {
                sweSection.insertBefore(selectorContainer, firstCard);
            } else {
                const cards = sweSection.querySelector('.benchmark-cards');
                if (cards) {
                    sweSection.insertBefore(selectorContainer, cards);
                } else {
                    sweSection.appendChild(selectorContainer);
                }
            }
        }
        
        // Event-Listener für die Selektoren hinzufügen
        const categorySelector = document.getElementById('task-category-selector');
        const languageSelector = document.getElementById('programming-language-selector');
        const liveUpdate = document.getElementById('swe-filter-update');
        
        if (categorySelector) {
            // Füge auch Tastaturunterstützung hinzu (Enter/Space)
            categorySelector.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    e.target.focus();
                }
            });
            
            categorySelector.addEventListener('change', (e) => {
                selectedTaskCategory = e.target.value;
                updateCharts();
                
                // Aktualisiere die Live-Region für Screenreader
                if (liveUpdate) {
                    liveUpdate.textContent = `Aufgabentyp geändert zu ${selectedTaskCategory}. Charts werden aktualisiert.`;
                }
                
                // UI-Interaktion protokollieren
                if (VXORUtils && VXORUtils.EventTypes) {
                    document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.USER_INTERACTION, {
                        detail: {
                            type: 'swe_task_category_change',
                            value: selectedTaskCategory
                        }
                    }));
                }
            });
        }
        
        if (languageSelector) {
            // Füge auch Tastaturunterstützung hinzu (Enter/Space)
            languageSelector.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    e.target.focus();
                }
            });
            
            languageSelector.addEventListener('change', (e) => {
                selectedLanguage = e.target.value;
                updateCharts();
                
                // Aktualisiere die Live-Region für Screenreader
                if (liveUpdate) {
                    liveUpdate.textContent = `Programmiersprache geändert zu ${selectedLanguage}. Charts werden aktualisiert.`;
                }
                
                // UI-Interaktion protokollieren
                if (VXORUtils && VXORUtils.EventTypes) {
                    document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.USER_INTERACTION, {
                        detail: {
                            type: 'swe_language_change',
                            value: selectedLanguage
                        }
                    }));
                }
            });
        }
    }
    
    /**
     * Registriert Event-Listener
     */
    function setupEventListeners() {
        // Beim Kernmodul für SWE-Bench-Daten registrieren
        if (typeof VXORUtils !== 'undefined') {
            VXORUtils.onDataUpdate('swe', handleSWEBenchData);
            VXORUtils.onThemeChange(updateTheme);
        } else {
            console.warn('VXORUtils nicht gefunden, eingeschränkte Funktionalität');
        }
    }
    
    /**
     * Verarbeitet SWE-Bench-Daten und aktualisiert die UI
     * @param {Object} data - Die Benchmark-Daten
     */
    function handleSWEBenchData(data) {
        if (!data) {
            console.error('Keine gültigen SWE-Bench-Daten erhalten');
            return;
        }
        
        // Daten für spätere Verwendung speichern
        lastData = data;
        
        // UI-Komponenten aktualisieren
        updateCharts();
    }
    
    /**
     * Aktualisiert alle Charts und Tabellen
     */
    function updateCharts() {
        // Aktualisiere die Charts, wenn Daten vorhanden sind
        if (lastData) {
            updateCodeGenChart(lastData);
            updateBugfixChart(lastData);
            updateMetricsTable(lastData);
        } else {
            // Andernfalls Daten simulieren
            updateCodeGenChart();
            updateBugfixChart();
            updateMetricsTable();
        }
    }
    
    /**
     * Aktualisiert das Code-Generierungs-Chart
     * @param {Object} [data] - Die Benchmark-Daten (optional)
     */
    function updateCodeGenChart(data) {
        if (!codeGenChart) return;
        
        // Komplexitätsstufen für die X-Achse
        const complexityLevels = ['Einfach', 'Mittel', 'Komplex', 'Sehr komplex'];
        
        // Zeit- und Qualitätsdaten
        const generationTimeData = [];
        const codeQualityData = [];
        
        // Für jede Komplexitätsstufe Daten finden oder simulieren
        complexityLevels.forEach(complexity => {
            // Zeit-Daten
            let timeValue;
            let qualityValue;
            
            if (data?.results) {
                // Daten aus Ergebnissen extrahieren
                const timeResult = data.results.find(r => 
                    r.metric.includes('Generierungszeit') && 
                    r.metric.includes(selectedTaskCategory) && 
                    r.metric.includes(selectedLanguage) && 
                    r.metric.includes(complexity)
                );
                
                const qualityResult = data.results.find(r => 
                    r.metric.includes('Code-Qualität') && 
                    r.metric.includes(selectedTaskCategory) && 
                    r.metric.includes(selectedLanguage) && 
                    r.metric.includes(complexity)
                );
                
                timeValue = timeResult ? timeResult.value : generateCodeGenTime(complexity);
                qualityValue = qualityResult ? qualityResult.value : generateCodeQuality(complexity);
            } else {
                // Daten simulieren
                timeValue = generateCodeGenTime(complexity);
                qualityValue = generateCodeQuality(complexity);
            }
            
            generationTimeData.push(timeValue);
            codeQualityData.push(qualityValue);
        });
        
        // Chart aktualisieren
        codeGenChart.data.labels = complexityLevels;
        codeGenChart.data.datasets[0].data = generationTimeData;
        codeGenChart.data.datasets[0].label = `Generierungszeit (${selectedLanguage})`;
        codeGenChart.data.datasets[1].data = codeQualityData;
        codeGenChart.data.datasets[1].label = `Code-Qualität (${selectedLanguage})`;
        codeGenChart.update();
    }
    
    /**
     * Aktualisiert das Bugfixing-Chart
     * @param {Object} [data] - Die Benchmark-Daten (optional)
     */
    function updateBugfixChart(data) {
        if (!bugfixChart) return;
        
        // Verschiedene Bugtypen für die X-Achse
        const bugTypes = ['Syntax', 'Logic', 'Performance', 'Security', 'Edge Case'];
        
        // Zeit- und Erfolgsraten-Daten
        const timeToFixData = [];
        const successRateData = [];
        
        // Für jeden Bugtyp Daten finden oder simulieren
        bugTypes.forEach(bugType => {
            // Zeit- und Erfolgsraten-Daten
            let timeValue;
            let successValue;
            
            if (data?.results) {
                // Daten aus Ergebnissen extrahieren
                const timeResult = data.results.find(r => 
                    r.metric.includes('Zeit bis Fix') && 
                    r.metric.includes(selectedLanguage) && 
                    r.metric.includes(bugType)
                );
                
                const successResult = data.results.find(r => 
                    r.metric.includes('Erfolgsrate') && 
                    r.metric.includes(selectedLanguage) && 
                    r.metric.includes(bugType)
                );
                
                timeValue = timeResult ? timeResult.value : generateTimeToFix(bugType);
                successValue = successResult ? successResult.value : generateSuccessRate(bugType);
            } else {
                // Daten simulieren
                timeValue = generateTimeToFix(bugType);
                successValue = generateSuccessRate(bugType);
            }
            
            timeToFixData.push(timeValue);
            successRateData.push(successValue);
        });
        
        // Chart aktualisieren
        bugfixChart.data.labels = bugTypes;
        bugfixChart.data.datasets[0].data = timeToFixData;
        bugfixChart.data.datasets[0].label = `Zeit bis Fix (${selectedLanguage})`;
        bugfixChart.data.datasets[1].data = successRateData;
        bugfixChart.data.datasets[1].label = `Erfolgsrate (${selectedLanguage})`;
        bugfixChart.update();
    }
    
    /**
     * Aktualisiert die SWE-Metriken-Tabelle
     * @param {Object} [data] - Die Benchmark-Daten (optional)
     */
    function updateMetricsTable(data) {
        if (!metricsTableBody) {
            metricsTableBody = document.querySelector('#swe-metrics tbody');
            if (!metricsTableBody) return;
        }
        
        // Tabelle leeren
        metricsTableBody.innerHTML = '';
        
        // Verschiedene Aufgabentypen für die Tabelle
        const taskTypes = taskCategories.slice(0, 3); // Nehme die ersten 3 Kategorien
        
        // Für jeden Aufgabentyp eine Zeile erstellen
        taskTypes.forEach(taskType => {
            const row = document.createElement('tr');
            
            // Aufgabentyp
            const taskCell = document.createElement('td');
            taskCell.textContent = taskType;
            
            // Erfolgsrate
            const successRateCell = document.createElement('td');
            let successRateValue;
            
            if (data?.results) {
                const successRateResult = data.results.find(r => 
                    r.metric.includes('Erfolgsrate') && 
                    r.metric.includes(taskType) && 
                    r.metric.includes(selectedLanguage)
                );
                
                successRateValue = successRateResult ? 
                    successRateResult.value : 
                    Math.min(0.95, 0.6 + Math.random() * 0.3);
            } else {
                successRateValue = Math.min(0.95, 0.6 + Math.random() * 0.3);
            }
            
            successRateCell.textContent = `${(successRateValue * 100).toFixed(1)}%`;
            
            // Zeit bis Lösung
            const timeCell = document.createElement('td');
            let timeValue;
            
            if (data?.results) {
                const timeResult = data.results.find(r => 
                    r.metric.includes('Zeit bis Lösung') && 
                    r.metric.includes(taskType) && 
                    r.metric.includes(selectedLanguage)
                );
                
                timeValue = timeResult ? 
                    timeResult.value : 
                    (taskType === 'Bugfixing' ? 60 + Math.random() * 60 : 120 + Math.random() * 180);
            } else {
                timeValue = taskType === 'Bugfixing' ? 60 + Math.random() * 60 : 120 + Math.random() * 180;
            }
            
            timeCell.textContent = `${timeValue.toFixed(1)} Sek`;
            
            // Codequalität
            const qualityCell = document.createElement('td');
            let qualityValue;
            
            if (data?.results) {
                const qualityResult = data.results.find(r => 
                    r.metric.includes('Codequalität') && 
                    r.metric.includes(taskType) && 
                    r.metric.includes(selectedLanguage)
                );
                
                qualityValue = qualityResult ? 
                    qualityResult.value : 
                    (6 + Math.random() * 3);
            } else {
                qualityValue = 6 + Math.random() * 3;
            }
            
            qualityCell.textContent = `${qualityValue.toFixed(1)}/10`;
            
            // Zellen zur Zeile hinzufügen
            row.appendChild(taskCell);
            row.appendChild(successRateCell);
            row.appendChild(timeCell);
            row.appendChild(qualityCell);
            
            // Zeile zur Tabelle hinzufügen
            metricsTableBody.appendChild(row);
        });
    }
    
    /**
     * Generiert eine realistische Code-Generierungszeit für die angegebene Komplexität
     * @param {string} complexity - Die Komplexitätsstufe
     * @returns {number} Die generierte Zeit in Sekunden
     */
    function generateCodeGenTime(complexity) {
        const baseTime = getLanguageBaseTime(selectedLanguage);
        const complexityFactor = getComplexityFactor(complexity);
        return baseTime * complexityFactor * (1 + Math.random() * 0.2 - 0.1);
    }
    
    /**
     * Generiert eine realistische Code-Qualitätsbewertung für die angegebene Komplexität
     * @param {string} complexity - Die Komplexitätsstufe
     * @returns {number} Die generierte Qualitätsbewertung (0-10)
     */
    function generateCodeQuality(complexity) {
        const baseQuality = getLanguageBaseQuality(selectedLanguage);
        const complexityPenalty = getComplexityPenalty(complexity);
        return Math.min(10, baseQuality - complexityPenalty * (1 + Math.random() * 0.2 - 0.1));
    }
    
    /**
     * Generiert eine realistische Zeit bis zur Lösung für den angegebenen Bugtyp
     * @param {string} bugType - Der Bugtyp
     * @returns {number} Die generierte Zeit in Sekunden
     */
    function generateTimeToFix(bugType) {
        const baseTime = getLanguageBaseTime(selectedLanguage) * 0.5;
        const bugFactor = getBugTypeFactor(bugType);
        return baseTime * bugFactor * (1 + Math.random() * 0.2 - 0.1);
    }
    
    /**
     * Generiert eine realistische Erfolgsrate für den angegebenen Bugtyp
     * @param {string} bugType - Der Bugtyp
     * @returns {number} Die generierte Erfolgsrate (0-1)
     */
    function generateSuccessRate(bugType) {
        const baseRate = getLanguageBaseSuccessRate(selectedLanguage);
        const bugPenalty = getBugTypePenalty(bugType);
        return Math.min(0.98, Math.max(0.5, baseRate - bugPenalty * (1 + Math.random() * 0.1 - 0.05)));
    }
    
    /**
     * Liefert die Basis-Generierungszeit für die angegebene Programmiersprache
     * @param {string} language - Die Programmiersprache
     * @returns {number} Die Basis-Zeit in Sekunden
     */
    function getLanguageBaseTime(language) {
        switch (language) {
            case 'Python': return 45;
            case 'JavaScript': return 40;
            case 'Java': return 60;
            case 'C++': return 70;
            case 'Rust': return 80;
            default: return 50;
        }
    }
    
    /**
     * Liefert die Basis-Codequalität für die angegebene Programmiersprache
     * @param {string} language - Die Programmiersprache
     * @returns {number} Die Basis-Qualität (0-10)
     */
    function getLanguageBaseQuality(language) {
        switch (language) {
            case 'Python': return 8.5;
            case 'JavaScript': return 8.0;
            case 'Java': return 8.2;
            case 'C++': return 7.8;
            case 'Rust': return 8.7;
            default: return 8.0;
        }
    }
    
    /**
     * Liefert die Basis-Erfolgsrate für die angegebene Programmiersprache
     * @param {string} language - Die Programmiersprache
     * @returns {number} Die Basis-Erfolgsrate (0-1)
     */
    function getLanguageBaseSuccessRate(language) {
        switch (language) {
            case 'Python': return 0.92;
            case 'JavaScript': return 0.88;
            case 'Java': return 0.90;
            case 'C++': return 0.85;
            case 'Rust': return 0.82;  // Niedriger wegen der Komplexität
            default: return 0.88;
        }
    }
    
    /**
     * Liefert den Komplexitätsfaktor für die angegebene Komplexitätsstufe
     * @param {string} complexity - Die Komplexitätsstufe
     * @returns {number} Der Komplexitätsfaktor
     */
    function getComplexityFactor(complexity) {
        switch (complexity) {
            case 'Einfach': return 1.0;
            case 'Mittel': return 2.0;
            case 'Komplex': return 3.5;
            case 'Sehr komplex': return 6.0;
            default: return 1.0;
        }
    }
    
    /**
     * Liefert die Qualitätsreduktion für die angegebene Komplexitätsstufe
     * @param {string} complexity - Die Komplexitätsstufe
     * @returns {number} Die Qualitätsreduktion
     */
    function getComplexityPenalty(complexity) {
        switch (complexity) {
            case 'Einfach': return 0.0;
            case 'Mittel': return 0.5;
            case 'Komplex': return 1.5;
            case 'Sehr komplex': return 3.0;
            default: return 0.0;
        }
    }
    
    /**
     * Liefert den Zeitfaktor für den angegebenen Bugtyp
     * @param {string} bugType - Der Bugtyp
     * @returns {number} Der Zeitfaktor
     */
    function getBugTypeFactor(bugType) {
        switch (bugType) {
            case 'Syntax': return 1.0;
            case 'Logic': return 2.5;
            case 'Performance': return 3.0;
            case 'Security': return 3.5;
            case 'Edge Case': return 4.0;
            default: return 2.0;
        }
    }
    
    /**
     * Liefert die Erfolgsratenreduktion für den angegebenen Bugtyp
     * @param {string} bugType - Der Bugtyp
     * @returns {number} Die Erfolgsratenreduktion
     */
    function getBugTypePenalty(bugType) {
        switch (bugType) {
            case 'Syntax': return 0.02;
            case 'Logic': return 0.08;
            case 'Performance': return 0.12;
            case 'Security': return 0.15;
            case 'Edge Case': return 0.18;
            default: return 0.10;
        }
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
        updateChartTheme(codeGenChart);
        updateChartTheme(bugfixChart);
    }
    
    /**
     * Öffentliche API des VXORSWEBench-Moduls
     */
    return {
        init,
        
        // Erlaubt externen Zugriff auf bestimmte Funktionen
        updateCharts,
        handleSWEBenchData,
        
        // Erlaubt das Setzen der Auswahloptionen von außen
        setTaskCategory: function(category) {
            if (taskCategories.includes(category)) {
                selectedTaskCategory = category;
                const selector = document.getElementById('task-category-selector');
                if (selector) selector.value = category;
                updateCharts();
            }
        },
        
        setLanguage: function(language) {
            if (programmingLanguages.includes(language)) {
                selectedLanguage = language;
                const selector = document.getElementById('programming-language-selector');
                if (selector) selector.value = language;
                updateCharts();
            }
        },
        
        // Nur für Testzwecke
        _getCharts: () => ({ codeGenChart, bugfixChart })
    };
})();

// Beim Kernmodul registrieren, sobald das DOM geladen ist
document.addEventListener('DOMContentLoaded', () => {
    if (typeof VXORUtils !== 'undefined') {
        VXORUtils.registerModule('swe', VXORSWEBench);
    } else {
        console.error('VXORUtils nicht gefunden! SWE-Bench-Modul kann nicht registriert werden.');
    }
});
