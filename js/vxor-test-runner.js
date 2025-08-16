/**
 * VXOR Benchmark Dashboard - Test-Runner
 * Version: 1.0.0
 * Datum: 01.05.2025
 * Phase 7.1: Integrationstests
 *
 * Zweck: Automatisierte Ausführung aller Testmodule mit
 * Reporting, Fehlerbehandlung und Zero-Toleranz-Validierung
 */

'use strict';

const VXORTestRunner = (function() {
    // Konfiguration
    const DEFAULT_CONFIG = {
        timeout: 10000,         // Standard-Timeout für Tests (10 Sekunden)
        retryAttempts: 2,       // Anzahl der Wiederholungen bei Fehlern
        retryDelay: 1000,       // Verzögerung zwischen Wiederholungen (1 Sekunde)
        autoSave: true,         // Automatisches Speichern der Testergebnisse
        logToConsole: true      // Ausgabe der Testergebnisse in die Konsole
    };
    
    // Status-Tracking
    let isRunning = false;
    let currentTest = null;
    let testStartTime = null;
    let testResults = null;
    
    // DOM-Elemente
    let reportContainer = null;
    
    /**
     * Initialisiert den Test-Runner
     * @param {Object} options - Optionale Konfiguration
     */
    function init(options = {}) {
        console.log('🚀 VXOR Test-Runner initialisiert');
        
        // Konfiguration mit den übergebenen Optionen zusammenführen
        const config = { ...DEFAULT_CONFIG, ...options };
        
        // Report-Container erstellen oder abrufen
        reportContainer = document.getElementById('test-report-container');
        
        if (!reportContainer) {
            reportContainer = document.createElement('div');
            reportContainer.id = 'test-report-container';
            reportContainer.className = 'test-report';
            document.body.appendChild(reportContainer);
            
            // Standard-Styling
            reportContainer.style.position = 'fixed';
            reportContainer.style.top = '20px';
            reportContainer.style.right = '20px';
            reportContainer.style.maxWidth = '400px';
            reportContainer.style.maxHeight = '80vh';
            reportContainer.style.overflow = 'auto';
            reportContainer.style.padding = '15px';
            reportContainer.style.backgroundColor = 'rgba(0, 0, 0, 0.85)';
            reportContainer.style.color = '#fff';
            reportContainer.style.fontFamily = 'monospace';
            reportContainer.style.fontSize = '14px';
            reportContainer.style.borderRadius = '5px';
            reportContainer.style.zIndex = '10000';
            reportContainer.style.boxShadow = '0 5px 15px rgba(0, 0, 0, 0.5)';
            reportContainer.style.transition = 'all 0.3s ease';
        }
        
        // Startzustand setzen
        reportContainer.innerHTML = `
            <h2>VXOR Test-Runner</h2>
            <div class="test-controls">
                <button id="run-all-tests" class="run-tests-button">Alle Tests ausführen</button>
                <button id="clear-report" class="clear-report-button">Report leeren</button>
            </div>
            <div id="test-summary" class="test-summary">Bereit für die Testausführung</div>
            <div id="test-results" class="test-results"></div>
        `;
        
        // Event-Listener für Test-Buttons hinzufügen
        const runAllButton = document.getElementById('run-all-tests');
        const clearReportButton = document.getElementById('clear-report');
        
        if (runAllButton) {
            runAllButton.addEventListener('click', function() {
                runAllTests();
            });
        }
        
        if (clearReportButton) {
            clearReportButton.addEventListener('click', function() {
                clearReport();
            });
        }
    }
    
    /**
     * Führt alle Tests aus
     */
    async function runAllTests() {
        if (isRunning) {
            console.warn('Tests laufen bereits, bitte warten...');
            return;
        }
        
        isRunning = true;
        testStartTime = performance.now();
        
        // Test-Report vorbereiten
        const testResultsContainer = document.getElementById('test-results');
        if (testResultsContainer) {
            testResultsContainer.innerHTML = '';
        }
        
        const testSummary = document.getElementById('test-summary');
        if (testSummary) {
            testSummary.innerHTML = '⏳ Tests werden ausgeführt...';
        }
        
        console.log('🔍 Starte alle Integrationstests...');
        
        // Gesamtergebnisse initialisieren
        testResults = {
            success: true,
            startTime: new Date().toISOString(),
            endTime: null,
            duration: 0,
            tests: []
        };
        
        try {
            // 1. Modul-Integrationstests
            const moduleResults = await moduleIntegrationTests();
            testResults.tests.push({
                name: 'Modul-Integrationstests',
                ...moduleResults
            });
            logResults('Modul-Integrationstests', moduleResults);
            
            // 2. Daten-Integritätstests
            const dataResults = await dataIntegrityTests();
            testResults.tests.push({
                name: 'Daten-Integritätstests',
                ...dataResults
            });
            logResults('Daten-Integritätstests', dataResults);
            
            // 3. UI-Konsistenztests
            const uiResults = await uiConsistencyTests();
            testResults.tests.push({
                name: 'UI-Konsistenztests',
                ...uiResults
            });
            logResults('UI-Konsistenztests', uiResults);
            
            // 4. Barrierefreiheitstests
            const accessibilityResults = await accessibilityTests();
            testResults.tests.push({
                name: 'Barrierefreiheitstests',
                ...accessibilityResults
            });
            logResults('Barrierefreiheitstests', accessibilityResults);
            
            // 5. Performance-Tests
            const performanceResults = await performanceTests();
            testResults.tests.push({
                name: 'Performance-Tests',
                ...performanceResults
            });
            logResults('Performance-Tests', performanceResults);
            
            // 6. Edge-Case-Tests
            const edgeCaseResults = await edgeCaseTests();
            testResults.tests.push({
                name: 'Edge-Case-Tests',
                ...edgeCaseResults
            });
            logResults('Edge-Case-Tests', edgeCaseResults);
            
            // Gesamtergebnis berechnen
            testResults.success = testResults.tests.every(test => test.success);
            testResults.endTime = new Date().toISOString();
            testResults.duration = performance.now() - testStartTime;
            
            // Test-Zusammenfassung anzeigen
            updateSummary(testResults);
            
            // Testergebnisse speichern
            if (DEFAULT_CONFIG.autoSave) {
                saveResults(testResults);
            }
            
            console.log(`✅ Alle Tests abgeschlossen (${testResults.duration.toFixed(2)}ms)`);
            console.log(`📊 Gesamtresultat: ${testResults.success ? 'ERFOLGREICH' : 'FEHLGESCHLAGEN'}`);
        } 
        catch (error) {
            console.error('❌ Fehler bei der Testausführung:', error);
            
            if (testSummary) {
                testSummary.innerHTML = `
                    <div class="test-error">
                        <h3>❌ Fehler bei der Testausführung</h3>
                        <pre>${error.message}</pre>
                    </div>
                `;
            }
        }
        finally {
            isRunning = false;
        }
    }
    
    /**
     * Führt einen einzelnen Test aus mit Retry-Mechanismus
     * @param {string} name - Name des Tests
     * @param {Function} testFunction - Die auszuführende Testfunktion
     * @param {Object} config - Testkonfiguration
     * @returns {Promise<Object>} Testergebnis
     */
    async function runTest(name, testFunction, config = {}) {
        // Konfiguration mit Default-Werten
        const testConfig = {
            ...DEFAULT_CONFIG,
            ...config
        };
        
        currentTest = name;
        let attempts = 0;
        let success = false;
        let error = null;
        let startTime = performance.now();
        
        console.log(`▶ Test starten: ${name}`);
        
        while (attempts <= testConfig.retryAttempts && !success) {
            if (attempts > 0) {
                console.log(`🔄 Wiederhole Test: ${name} (Versuch ${attempts}/${testConfig.retryAttempts})`);
                await new Promise(resolve => setTimeout(resolve, testConfig.retryDelay));
            }
            
            try {
                // Test mit Timeout ausführen
                const testPromise = testFunction();
                const timeoutPromise = new Promise((_, reject) => {
                    setTimeout(() => reject(new Error(`Timeout nach ${testConfig.timeout}ms`)), testConfig.timeout);
                });
                
                // Der schnellere der beiden gewinnt (Test oder Timeout)
                success = await Promise.race([testPromise, timeoutPromise]);
                error = null;
            } 
            catch (err) {
                success = false;
                error = err;
                console.error(`❌ Test fehlgeschlagen: ${name} - ${err.message}`);
            }
            
            attempts++;
            
            // Bei Erfolg oder maximaler Anzahl von Versuchen beenden
            if (success || attempts > testConfig.retryAttempts) {
                break;
            }
        }
        
        const duration = performance.now() - startTime;
        
        const result = {
            success,
            duration,
            attempts,
            timestamp: new Date().toISOString()
        };
        
        if (error) {
            result.error = error.message;
            result.stack = error.stack;
        }
        
        console.log(`${success ? '✅' : '❌'} Test abgeschlossen: ${name} (${duration.toFixed(2)}ms)`);
        
        currentTest = null;
        return result;
    }
    
    /**
     * Aktualisiert die Test-Zusammenfassung im UI
     * @param {Object} results - Testergebnisse
     */
    function updateSummary(results) {
        const testSummary = document.getElementById('test-summary');
        
        if (!testSummary) return;
        
        const totalTests = results.tests.length;
        const successfulTests = results.tests.filter(test => test.success).length;
        
        const summaryHTML = `
            <h3 style="margin-bottom: 5px;">${results.success ? '✅ TESTS ERFOLGREICH' : '❌ TESTS FEHLGESCHLAGEN'}</h3>
            <div class="stats">
                <p>Gesamtergebnis: ${successfulTests}/${totalTests} Tests erfolgreich</p>
                <p>Dauer: ${results.duration.toFixed(2)}ms</p>
                <p>Abgeschlossen: ${new Date(results.endTime).toLocaleTimeString()}</p>
            </div>
        `;
        
        testSummary.innerHTML = summaryHTML;
        
        // Farbliche Gestaltung je nach Ergebnis
        if (results.success) {
            testSummary.style.backgroundColor = 'rgba(0, 128, 0, 0.2)';
            testSummary.style.border = '1px solid #00a000';
        } else {
            testSummary.style.backgroundColor = 'rgba(128, 0, 0, 0.2)';
            testSummary.style.border = '1px solid #a00000';
        }
    }
    
    /**
     * Protokolliert Testergebnisse im UI
     * @param {string} testName - Name des Tests
     * @param {Object} results - Testergebnisse
     */
    function logResults(testName, results) {
        const testResultsContainer = document.getElementById('test-results');
        
        if (!testResultsContainer) return;
        
        const resultElement = document.createElement('div');
        resultElement.className = `test-result ${results.success ? 'success' : 'failure'}`;
        
        // Zähle erfolgreiche Subtests
        let successfulTests = 0;
        let totalTests = 0;
        
        if (results.tests && Array.isArray(results.tests)) {
            totalTests = results.tests.length;
            successfulTests = results.tests.filter(test => test.success).length;
        }
        
        // Erstelle HTML für das Ergebnis
        resultElement.innerHTML = `
            <div class="test-header">
                <h3 class="test-name">
                    <span class="test-icon">${results.success ? '✅' : '❌'}</span>
                    ${testName}
                </h3>
                <div class="test-details">
                    <span class="test-count">${successfulTests}/${totalTests} Tests</span>
                    <span class="test-duration">${results.duration ? results.duration.toFixed(2) + 'ms' : 'N/A'}</span>
                </div>
            </div>
        `;
        
        // Füge Details hinzu, wenn vorhanden
        if (results.tests && Array.isArray(results.tests) && results.tests.length > 0) {
            const detailsContainer = document.createElement('div');
            detailsContainer.className = 'subtest-container';
            
            results.tests.forEach(subtest => {
                const subtestElement = document.createElement('div');
                subtestElement.className = `subtest ${subtest.success ? 'success' : 'failure'}`;
                
                subtestElement.innerHTML = `
                    <div class="subtest-header">
                        <span class="subtest-icon">${subtest.success ? '✅' : '❌'}</span>
                        <span class="subtest-name">${subtest.name || 'Unbenannter Test'}</span>
                        <span class="subtest-duration">${subtest.duration ? subtest.duration.toFixed(2) + 'ms' : 'N/A'}</span>
                    </div>
                `;
                
                // Zeige Fehler an, wenn vorhanden
                if (!subtest.success && subtest.error) {
                    const errorElement = document.createElement('div');
                    errorElement.className = 'subtest-error';
                    errorElement.innerHTML = `<pre>${subtest.error}</pre>`;
                    subtestElement.appendChild(errorElement);
                }
                
                detailsContainer.appendChild(subtestElement);
            });
            
            resultElement.appendChild(detailsContainer);
        }
        
        // Zeige Fehler an, wenn vorhanden
        if (!results.success && results.error) {
            const errorElement = document.createElement('div');
            errorElement.className = 'test-error';
            errorElement.innerHTML = `<pre>${results.error}</pre>`;
            resultElement.appendChild(errorElement);
        }
        
        testResultsContainer.appendChild(resultElement);
    }
    
    /**
     * Speichert Testergebnisse im localStorage (wenn verfügbar)
     * @param {Object} results - Testergebnisse
     */
    function saveResults(results) {
        if (!window.localStorage) return;
        
        try {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const key = `vxor-test-results-${timestamp}`;
            localStorage.setItem(key, JSON.stringify(results));
            console.log(`📝 Testergebnisse gespeichert unter: ${key}`);
        } 
        catch (error) {
            console.error('❌ Fehler beim Speichern der Testergebnisse:', error);
        }
    }
    
    /**
     * Löscht den Test-Report
     */
    function clearReport() {
        const testResultsContainer = document.getElementById('test-results');
        const testSummary = document.getElementById('test-summary');
        
        if (testResultsContainer) {
            testResultsContainer.innerHTML = '';
        }
        
        if (testSummary) {
            testSummary.innerHTML = 'Bereit für die Testausführung';
            testSummary.style.backgroundColor = 'transparent';
            testSummary.style.border = 'none';
        }
        
        console.log('🧹 Test-Report gelöscht');
    }
    
    /**
     * Modul-Integrationstests
     */
    async function moduleIntegrationTests() {
        if (typeof VXORModuleIntegrationTests !== 'undefined' && 
            typeof VXORModuleIntegrationTests.runTests === 'function') {
            console.log('✓ Verwende externes Modul-Integrationstest-Modul...');
            return await VXORModuleIntegrationTests.runTests();
        }
        
        console.warn('⚠ Externes Modul-Integrationstest-Modul nicht gefunden, verwende Platzhalter!');
        const results = {
            success: false,
            tests: [],
            error: 'Testmodul nicht geladen. js/test-modules/module-integration-tests.js muss im HTML eingebunden werden.'
        };
        
        return results;
    }
    
    /**
     * Daten-Integritätstests
     */
    async function dataIntegrityTests() {
        if (typeof VXORDataIntegrityTests !== 'undefined' && 
            typeof VXORDataIntegrityTests.runTests === 'function') {
            console.log('✓ Verwende externes Daten-Integritätstest-Modul...');
            return await VXORDataIntegrityTests.runTests();
        }
        
        console.warn('⚠ Externes Daten-Integritätstest-Modul nicht gefunden, verwende Platzhalter!');
        const results = {
            success: false,
            tests: [],
            error: 'Testmodul nicht geladen. js/test-modules/data-integrity-tests.js muss im HTML eingebunden werden.'
        };
        
        return results;
    }
    
    /**
     * UI-Konsistenztests
     */
    async function uiConsistencyTests() {
        if (typeof VXORUIConsistencyTests !== 'undefined' && 
            typeof VXORUIConsistencyTests.runTests === 'function') {
            console.log('✓ Verwende externes UI-Konsistenztest-Modul...');
            return await VXORUIConsistencyTests.runTests();
        }
        
        console.warn('⚠ Externes UI-Konsistenztest-Modul nicht gefunden, verwende Platzhalter!');
        const results = {
            success: false,
            tests: [],
            error: 'Testmodul nicht geladen. js/test-modules/ui-consistency-tests.js muss im HTML eingebunden werden.'
        };
        
        return results;
    }
    
    /**
     * Barrierefreiheitstests
     */
    async function accessibilityTests() {
        if (typeof VXORAccessibilityTests !== 'undefined' && 
            typeof VXORAccessibilityTests.runTests === 'function') {
            console.log('✓ Verwende externes Barrierefreiheitstest-Modul...');
            return await VXORAccessibilityTests.runTests();
        }
        
        console.warn('⚠ Externes Barrierefreiheitstest-Modul nicht gefunden, verwende Platzhalter!');
        const results = {
            success: false,
            tests: [],
            error: 'Testmodul nicht geladen. js/test-modules/accessibility-tests.js muss im HTML eingebunden werden.'
        };
        
        return results;
    }
    
    /**
     * Performance-Tests
     */
    async function performanceTests() {
        if (typeof VXORPerformanceTests !== 'undefined' && 
            typeof VXORPerformanceTests.runTests === 'function') {
            console.log('✓ Verwende externes Performance-Test-Modul...');
            return await VXORPerformanceTests.runTests();
        }
        
        console.warn('⚠ Externes Performance-Test-Modul nicht gefunden, verwende Platzhalter!');
        const results = {
            success: false,
            tests: [],
            error: 'Testmodul nicht geladen. js/test-modules/performance-tests.js muss im HTML eingebunden werden.'
        };
        
        return results;
    }
    
    /**
     * Edge-Case-Tests
     */
    async function edgeCaseTests() {
        if (typeof VXOREdgeCaseTests !== 'undefined' && 
            typeof VXOREdgeCaseTests.runTests === 'function') {
            console.log('✓ Verwende externes Edge-Case-Test-Modul...');
            return await VXOREdgeCaseTests.runTests();
        }
        
        console.warn('⚠ Externes Edge-Case-Test-Modul nicht gefunden, verwende Platzhalter!');
        const results = {
            success: false,
            tests: [],
            error: 'Testmodul nicht geladen. js/test-modules/edge-case-tests.js muss im HTML eingebunden werden.'
        };
        
        return results;
    }
    
    // CSS-Stil für den Test-Report
    function injectStyles() {
        const styleElement = document.createElement('style');
        styleElement.innerHTML = `
            .test-report {
                font-family: monospace;
            }
            
            .test-controls {
                margin-bottom: 15px;
                display: flex;
                gap: 10px;
            }
            
            .run-tests-button, .clear-report-button {
                padding: 5px 10px;
                border: none;
                border-radius: 3px;
                cursor: pointer;
                font-family: monospace;
            }
            
            .run-tests-button {
                background-color: #4CAF50;
                color: white;
            }
            
            .clear-report-button {
                background-color: #f44336;
                color: white;
            }
            
            .test-summary {
                padding: 10px;
                margin-bottom: 15px;
                border-radius: 5px;
            }
            
            .test-result {
                margin-bottom: 10px;
                border-radius: 3px;
                overflow: hidden;
                border-left: 4px solid transparent;
            }
            
            .test-result.success {
                border-left-color: #4CAF50;
                background-color: rgba(76, 175, 80, 0.1);
            }
            
            .test-result.failure {
                border-left-color: #f44336;
                background-color: rgba(244, 67, 54, 0.1);
            }
            
            .test-header {
                padding: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .test-name {
                margin: 0;
                font-size: 16px;
                display: flex;
                align-items: center;
                gap: 5px;
            }
            
            .test-details {
                font-size: 12px;
                display: flex;
                gap: 10px;
            }
            
            .subtest-container {
                padding: 0 10px 10px;
            }
            
            .subtest {
                padding: 5px;
                margin-bottom: 5px;
                border-radius: 3px;
                font-size: 14px;
            }
            
            .subtest.success {
                background-color: rgba(76, 175, 80, 0.05);
            }
            
            .subtest.failure {
                background-color: rgba(244, 67, 54, 0.05);
            }
            
            .subtest-header {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .subtest-name {
                flex-grow: 1;
            }
            
            .subtest-duration {
                font-size: 12px;
                color: #888;
            }
            
            .test-error, .subtest-error {
                padding: 10px;
                margin-top: 5px;
                background-color: rgba(0, 0, 0, 0.1);
                border-radius: 3px;
                overflow: auto;
            }
            
            .test-error pre, .subtest-error pre {
                margin: 0;
                font-family: monospace;
                font-size: 12px;
                color: #f44336;
            }
        `;
        
        document.head.appendChild(styleElement);
    }
    
    // Öffentliche API
    return {
        init,
        runAllTests,
        runTest,
        clearReport,
        moduleIntegrationTests,
        dataIntegrityTests,
        uiConsistencyTests,
        accessibilityTests,
        performanceTests,
        edgeCaseTests,
        injectStyles
    };
})();

// Automatische Initialisierung, wenn das DOM geladen ist
document.addEventListener('DOMContentLoaded', function() {
    if (typeof VXORTestRunner !== 'undefined') {
        VXORTestRunner.init();
        VXORTestRunner.injectStyles();
        console.log('🚀 VXOR Test-Runner bereit für die Testausführung');
    }
});

// Einfache Testfunktion für externe Aufrufe
window.runVXORTests = function() {
    if (typeof VXORTestRunner !== 'undefined') {
        VXORTestRunner.runAllTests();
    } else {
        console.error('❌ VXORTestRunner ist nicht verfügbar');
    }
};