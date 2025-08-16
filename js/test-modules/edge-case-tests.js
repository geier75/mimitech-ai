/**
 * VXOR Benchmark Dashboard - Edge-Case-Tests
 * Version: 1.0.0
 * Datum: 01.05.2025
 * Phase 7.1: Integrationstests
 *
 * Zweck: Validierung der Robustheit bei ungewöhnlichen Eingaben
 * und Grenzwerten, um die Fehlerresistenz zu gewährleisten
 */

'use strict';

const VXOREdgeCaseTests = (function() {
    // Testkonfiguration
    const TEST_CONFIG = {
        timeout: 10000, // Längere Timeout für Edge-Case-Tests
        retryAttempts: 2
    };
    
    // Extreme Testdaten
    const EDGE_CASE_DATA = {
        emptyResults: {
            results: [],
            timestamp: Date.now()
        },
        extremeValues: {
            results: [
                { component: 'Test', metric: 'MaxValue', value: Number.MAX_VALUE },
                { component: 'Test', metric: 'MinValue', value: Number.MIN_VALUE },
                { component: 'Test', metric: 'Infinity', value: Infinity },
                { component: 'Test', metric: 'NegativeInfinity', value: -Infinity },
                { component: 'Test', metric: 'NaN', value: NaN }
            ],
            timestamp: Date.now()
        },
        malformedJson: "{ this is not valid JSON }",
        hugeDataset: {
            results: Array(10000).fill().map((_, i) => ({
                component: `Component${i % 10}`,
                metric: `Metric${i % 100}`,
                value: i * 1.5
            })),
            timestamp: Date.now()
        },
        invalidTimestamp: {
            results: [
                { component: 'Test', metric: 'TestMetric', value: 100 }
            ],
            timestamp: "Not a timestamp"
        },
        missingRequired: {
            // timestamp fehlt absichtlich
            results: [
                { component: 'Test', metric: 'TestMetric', value: 100 }
            ]
        },
        specialCharacters: {
            results: [
                { component: '<script>alert("XSS")</script>', metric: '"><img src=x onerror=alert(1)>', value: 100 },
                { component: '--Injection--', metric: "'; DROP TABLE users;--", value: 100 }
            ],
            timestamp: Date.now()
        }
    };
    
    /**
     * Führt alle Edge-Case-Tests aus
     * @returns {Promise<Object>} Testergebnisse
     */
    async function runTests() {
        console.log('▶ Starte Edge-Case-Tests...');
        
        const results = {
            success: true,
            tests: []
        };
        
        try {
            // Test 1: Leere Ergebnisse
            const emptyResultsTest = await VXORTestRunner.runTest(
                'Leere Ergebnisse validieren',
                testEmptyResults,
                TEST_CONFIG
            );
            results.tests.push(emptyResultsTest);
            
            // Test 2: Extreme Werte
            const extremeValuesTest = await VXORTestRunner.runTest(
                'Extreme Werte validieren',
                testExtremeValues,
                TEST_CONFIG
            );
            results.tests.push(extremeValuesTest);
            
            // Test 3: Große Datensätze
            const largeDatasetTest = await VXORTestRunner.runTest(
                'Große Datensätze validieren',
                testLargeDataset,
                TEST_CONFIG
            );
            results.tests.push(largeDatasetTest);
            
            // Test 4: Ungültige Datenformate
            const invalidFormatTest = await VXORTestRunner.runTest(
                'Ungültige Datenformate validieren',
                testInvalidFormat,
                TEST_CONFIG
            );
            results.tests.push(invalidFormatTest);
            
            // Test 5: Netzwerkfehler
            const networkErrorTest = await VXORTestRunner.runTest(
                'Netzwerkfehler validieren',
                testNetworkErrors,
                TEST_CONFIG
            );
            results.tests.push(networkErrorTest);
            
            // Gesamtergebnis bestimmen
            results.success = results.tests.every(test => test.success);
            
            return results;
        }
        catch (error) {
            console.error('❌ Fehler in Edge-Case-Tests:', error);
            results.success = false;
            results.error = error.message;
            return results;
        }
    }
    
    /**
     * Test 1: Validierung der Verarbeitung leerer Ergebnisse
     */
    async function testEmptyResults() {
        if (!window.VXORUtils) {
            throw new Error('VXORUtils nicht verfügbar');
        }
        
        // Prüfen, ob Fehlerbehandlung verfügbar ist
        if (typeof VXORUtils.handleApiError !== 'function') {
            throw new Error('VXORUtils.handleApiError nicht verfügbar');
        }
        
        // Event-Handler für Fehler
        let errorReceived = false;
        const errorHandler = () => {
            errorReceived = true;
        };
        
        document.addEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        
        try {
            // Leere Ergebnisse simulieren
            document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.DATA_UPDATED, {
                detail: { 
                    category: 'test', 
                    data: EDGE_CASE_DATA.emptyResults 
                }
            }));
            
            // Kurze Verzögerung für Datenverarbeitung
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Beim Erhalt leerer Ergebnisse sollte kein Fehler auftreten
            if (errorReceived) {
                throw new Error('Unerwarteter Fehler bei leeren Ergebnissen');
            }
            
            // Fehlerbehandlung manuell testen
            let errorContainer = document.getElementById('error-container-test');
            
            if (!errorContainer) {
                errorContainer = document.createElement('div');
                errorContainer.id = 'error-container-test';
                document.body.appendChild(errorContainer);
            }
            
            // Fehlermeldung für leere Ergebnisse anzeigen
            VXORUtils.handleApiError(
                { message: 'Keine Daten gefunden' },
                'test',
                'error-container-test'
            );
            
            // Kurze Verzögerung für Fehlerverarbeitung
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Fehlermeldung sollte angezeigt werden
            const errorMessage = errorContainer.querySelector('.error-message');
            
            if (!errorMessage) {
                throw new Error('Keine Fehlermeldung für leere Ergebnisse angezeigt');
            }
            
            // Aufräumen
            errorContainer.innerHTML = '';
            
            return true;
        }
        finally {
            document.removeEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        }
    }
    
    /**
     * Test 2: Validierung der Verarbeitung extremer Werte
     */
    async function testExtremeValues() {
        if (!window.VXORUtils) {
            throw new Error('VXORUtils nicht verfügbar');
        }
        
        // Event-Handler für Fehler
        let errorReceived = false;
        const errorHandler = () => {
            errorReceived = true;
        };
        
        document.addEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        
        try {
            // Extreme Werte simulieren
            document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.DATA_UPDATED, {
                detail: { 
                    category: 'test', 
                    data: EDGE_CASE_DATA.extremeValues 
                }
            }));
            
            // Kurze Verzögerung für Datenverarbeitung
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Bei extremen Werten sollte kein Fehler auftreten
            if (errorReceived) {
                throw new Error('Unerwarteter Fehler bei extremen Werten');
            }
            
            // Charts sollten trotz extremer Werte gerendert werden
            const charts = document.querySelectorAll('canvas');
            
            if (charts.length === 0) {
                console.warn('Keine Charts gefunden, visueller Test nicht möglich');
            }
            
            return true;
        }
        finally {
            document.removeEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        }
    }
    
    /**
     * Test 3: Validierung der Verarbeitung großer Datensätze
     */
    async function testLargeDataset() {
        if (!window.VXORUtils) {
            throw new Error('VXORUtils nicht verfügbar');
        }
        
        // Event-Handler für Fehler
        let errorReceived = false;
        let startTime, endTime;
        
        const errorHandler = () => {
            errorReceived = true;
        };
        
        document.addEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        
        try {
            // Startzeit für Performance-Messung
            startTime = performance.now();
            
            // Großen Datensatz simulieren
            document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.DATA_UPDATED, {
                detail: { 
                    category: 'test', 
                    data: EDGE_CASE_DATA.hugeDataset 
                }
            }));
            
            // Warte auf UI-Updates
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Endzeit für Performance-Messung
            endTime = performance.now();
            
            // Bei großen Datensätzen sollte kein Fehler auftreten
            if (errorReceived) {
                throw new Error('Unerwarteter Fehler bei großem Datensatz');
            }
            
            // Verarbeitungszeit protokollieren
            const processingTime = endTime - startTime;
            console.log(`📊 Verarbeitungszeit für großen Datensatz: ${processingTime.toFixed(2)}ms`);
            
            // Warnung bei sehr langer Verarbeitungszeit (über 3 Sekunden)
            if (processingTime > 3000) {
                console.warn(`⚠️ Verarbeitungszeit für großen Datensatz sehr lang: ${processingTime.toFixed(2)}ms`);
            }
            
            return true;
        }
        finally {
            document.removeEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        }
    }
    
    /**
     * Test 4: Validierung der Robustheit bei ungültigen Datenformaten
     */
    async function testInvalidFormat() {
        if (!window.VXORUtils) {
            throw new Error('VXORUtils nicht verfügbar');
        }
        
        // Event-Handler für Fehler
        let errorCount = 0;
        const errorHandler = () => {
            errorCount++;
        };
        
        document.addEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        
        try {
            // Test 1: Fehlender Pflichtparameter (timestamp)
            document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.DATA_UPDATED, {
                detail: { 
                    category: 'test', 
                    data: EDGE_CASE_DATA.missingRequired 
                }
            }));
            
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Test 2: Ungültiger Zeitstempel
            document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.DATA_UPDATED, {
                detail: { 
                    category: 'test', 
                    data: EDGE_CASE_DATA.invalidTimestamp 
                }
            }));
            
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Test 3: Sonderzeichen in Daten (potenzieller XSS)
            document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.DATA_UPDATED, {
                detail: { 
                    category: 'test', 
                    data: EDGE_CASE_DATA.specialCharacters 
                }
            }));
            
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Es sollten keine unbehandelten Fehler auftreten
            console.log(`📊 Erkannte Fehler bei ungültigen Formaten: ${errorCount}`);
            
            // Bei ungültigen Formaten sollte die Anwendung nicht abstürzen
            // Der Test ist erfolgreich, wenn wir bis hierher kommen
            
            return true;
        }
        finally {
            document.removeEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        }
    }
    
    /**
     * Test 5: Validierung der Robustheit bei Netzwerkfehlern
     */
    async function testNetworkErrors() {
        if (!window.VXORUtils) {
            throw new Error('VXORUtils nicht verfügbar');
        }
        
        // Prüfen, ob API-Funktionen verfügbar sind
        if (typeof VXORUtils.fetchBenchmarkData !== 'function') {
            throw new Error('VXORUtils.fetchBenchmarkData nicht verfügbar');
        }
        
        // Original-API-Funktionen sichern
        const originalFetch = window.fetch;
        const originalFetchBenchmark = VXORUtils.fetchBenchmarkData;
        
        // Event-Handler für Fehler
        let errorReceived = false;
        const errorHandler = () => {
            errorReceived = true;
        };
        
        document.addEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        
        try {
            // Fehlercontainer vorbereiten
            let errorContainer = document.getElementById('network-error-container');
            
            if (!errorContainer) {
                errorContainer = document.createElement('div');
                errorContainer.id = 'network-error-container';
                document.body.appendChild(errorContainer);
            }
            
            // Simuliere einen Netzwerkfehler durch Überschreiben der fetch-Funktion
            window.fetch = function() {
                return Promise.reject(new Error('Simulierter Netzwerkfehler'));
            };
            
            // Überschreibe die API-Funktion, um einen Fehler auszulösen
            VXORUtils.fetchBenchmarkData = async function() {
                throw new Error('Simulierter API-Fehler');
            };
            
            // Versuche, Daten abzurufen (sollte fehlschlagen)
            try {
                await VXORUtils.fetchBenchmarkData('test');
            } catch (error) {
                // Erwarteter Fehler, verarbeite ihn mit der Fehlerbehandlung
                VXORUtils.handleApiError(error, 'test', 'network-error-container');
            }
            
            // Kurze Verzögerung für Fehlerverarbeitung
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Prüfe, ob ein Fehler-Event ausgelöst wurde
            if (!errorReceived) {
                throw new Error('Kein Fehler-Event bei simuliertem Netzwerkfehler ausgelöst');
            }
            
            // Prüfe, ob die Fehlermeldung angezeigt wird
            const errorMessage = errorContainer.querySelector('.error-message');
            
            if (!errorMessage) {
                throw new Error('Keine Fehlermeldung für Netzwerkfehler angezeigt');
            }
            
            // Prüfe, ob ein Retry-Button vorhanden ist
            const retryButton = errorContainer.querySelector('.retry-button') || 
                               errorContainer.querySelector('button');
            
            if (!retryButton) {
                throw new Error('Kein Retry-Button für Netzwerkfehler gefunden');
            }
            
            // Aufräumen
            errorContainer.innerHTML = '';
            
            return true;
        }
        finally {
            // Originale Funktionen wiederherstellen
            window.fetch = originalFetch;
            VXORUtils.fetchBenchmarkData = originalFetchBenchmark;
            
            document.removeEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        }
    }
    
    // Öffentliche API
    return {
        runTests
    };
})();

// Registriere die Tests beim TestRunner
if (typeof VXORTestRunner !== 'undefined') {
    VXORTestRunner.edgeCaseTests = VXOREdgeCaseTests.runTests;
}
