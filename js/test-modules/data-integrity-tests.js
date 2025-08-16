/**
 * VXOR Benchmark Dashboard - Daten-Integritätstests
 * Version: 1.0.0
 * Datum: 01.05.2025
 * Phase 7.1: Integrationstests
 *
 * Zweck: Validierung der korrekten Datenverarbeitung und -darstellung
 * mit besonderem Fokus auf Robustheit bei unvollständigen/ungültigen Daten
 */

'use strict';

const VXORDataIntegrityTests = (function() {
    // Testkonfiguration
    const TEST_CONFIG = {
        timeout: 5000,
        retryAttempts: 2
    };
    
    // Testdatensätze für verschiedene Szenarien
    const TEST_DATA = {
        // Normaler Datensatz
        normal: {
            results: [
                { component: 'Matrix', metric: 'Multiplikation', value: 100 },
                { component: 'Matrix', metric: 'Addition', value: 150 },
                { component: 'Quantum', metric: 'Gates', value: 200 }
            ],
            timestamp: Date.now()
        },
        // Unvollständiger Datensatz
        incomplete: {
            results: [
                { component: 'Matrix', metric: 'Multiplikation' }, // Kein Wert
                { metric: 'Addition', value: 150 },                // Keine Komponente
                { component: 'Quantum', value: 200 }               // Keine Metrik
            ],
            timestamp: Date.now()
        },
        // Datensatz mit NULL-Werten
        nullValues: {
            results: [
                { component: 'Matrix', metric: 'Multiplikation', value: null },
                { component: 'Quantum', metric: 'Gates', value: null }
            ],
            timestamp: Date.now()
        },
        // Datensatz mit extremen Werten
        extremeValues: {
            results: [
                { component: 'Matrix', metric: 'Multiplikation', value: Number.MAX_SAFE_INTEGER },
                { component: 'Matrix', metric: 'Addition', value: -9999999 },
                { component: 'Quantum', metric: 'Gates', value: 0.0000001 }
            ],
            timestamp: Date.now()
        },
        // Fehlerfall
        error: {
            error: 'Simulierter API-Fehler für Tests',
            message: 'Daten konnten nicht geladen werden'
        }
    };
    
    /**
     * Führt alle Daten-Integritätstests aus
     * @returns {Promise<Object>} Testergebnisse
     */
    async function runTests() {
        console.log('▶ Starte Daten-Integritätstests...');
        
        const results = {
            success: true,
            tests: []
        };
        
        try {
            // Test 1: Normale Datenverarbeitung
            const normalDataTest = await VXORTestRunner.runTest(
                'Normale Datenverarbeitung validieren',
                testNormalData,
                TEST_CONFIG
            );
            results.tests.push(normalDataTest);
            
            // Test 2: Unvollständige Daten
            const incompleteDataTest = await VXORTestRunner.runTest(
                'Unvollständige Daten validieren',
                testIncompleteData,
                TEST_CONFIG
            );
            results.tests.push(incompleteDataTest);
            
            // Test 3: NULL-Werte
            const nullValuesTest = await VXORTestRunner.runTest(
                'NULL-Werte validieren',
                testNullValues,
                TEST_CONFIG
            );
            results.tests.push(nullValuesTest);
            
            // Test 4: Extreme Werte
            const extremeValuesTest = await VXORTestRunner.runTest(
                'Extreme Werte validieren',
                testExtremeValues,
                TEST_CONFIG
            );
            results.tests.push(extremeValuesTest);
            
            // Test 5: Fehlerbehandlung
            const errorHandlingTest = await VXORTestRunner.runTest(
                'Fehlerbehandlung validieren',
                testErrorHandling,
                TEST_CONFIG
            );
            results.tests.push(errorHandlingTest);
            
            // Gesamtergebnis bestimmen
            results.success = results.tests.every(test => test.success);
            
            return results;
        }
        catch (error) {
            console.error('❌ Fehler in Daten-Integritätstests:', error);
            results.success = false;
            results.error = error.message;
            return results;
        }
    }
    
    /**
     * Test 1: Validierung der normalen Datenverarbeitung
     */
    async function testNormalData() {
        if (!window.VXORUtils) {
            throw new Error('VXORUtils nicht verfügbar');
        }
        
        // Event für Daten-Update überwachen
        let updateReceived = false;
        let hasErrors = false;
        
        const dataUpdateHandler = () => {
            updateReceived = true;
        };
        
        const errorHandler = () => {
            hasErrors = true;
        };
        
        document.addEventListener(VXORUtils.EventTypes.DATA_UPDATED, dataUpdateHandler);
        document.addEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        
        try {
            // Daten-Update simulieren
            document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.DATA_UPDATED, {
                detail: { 
                    category: 'matrix', 
                    data: TEST_DATA.normal 
                }
            }));
            
            // Kurze Verzögerung für Datenverarbeitung
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Prüfungen durchführen
            if (!updateReceived) {
                throw new Error('Daten-Update-Event wurde nicht empfangen');
            }
            
            if (hasErrors) {
                throw new Error('Fehler bei der Verarbeitung normaler Daten');
            }
            
            // Charts und Tabellen sollten vorhanden sein
            const charts = document.querySelectorAll('canvas');
            if (charts.length === 0) {
                throw new Error('Keine Charts nach Datenaktualisierung gefunden');
            }
            
            return true;
        }
        finally {
            // Event-Handler entfernen
            document.removeEventListener(VXORUtils.EventTypes.DATA_UPDATED, dataUpdateHandler);
            document.removeEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        }
    }
    
    /**
     * Test 2: Validierung der Verarbeitung unvollständiger Daten
     */
    async function testIncompleteData() {
        if (!window.VXORUtils) {
            throw new Error('VXORUtils nicht verfügbar');
        }
        
        // Wir erwarten keinen Fehler, nur eine korrekte Behandlung unvollständiger Daten
        let hasErrors = false;
        let updateReceived = false;
        
        const dataUpdateHandler = () => {
            updateReceived = true;
        };
        
        const errorHandler = () => {
            hasErrors = true;
        };
        
        document.addEventListener(VXORUtils.EventTypes.DATA_UPDATED, dataUpdateHandler);
        document.addEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        
        try {
            // Daten-Update mit unvollständigen Daten simulieren
            document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.DATA_UPDATED, {
                detail: { 
                    category: 'matrix', 
                    data: TEST_DATA.incomplete 
                }
            }));
            
            // Kurze Verzögerung für Datenverarbeitung
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Prüfungen durchführen
            if (!updateReceived) {
                throw new Error('Daten-Update-Event wurde nicht empfangen');
            }
            
            // Wir erwarten keine JavaScript-Fehler
            if (hasErrors) {
                throw new Error('Fehler bei der Verarbeitung unvollständiger Daten');
            }
            
            // Es sollte keine JavaScript-Konsolenfehler geben
            // (Dies kann nur indirekt getestet werden)
            
            return true;
        }
        finally {
            // Event-Handler entfernen
            document.removeEventListener(VXORUtils.EventTypes.DATA_UPDATED, dataUpdateHandler);
            document.removeEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        }
    }
    
    /**
     * Test 3: Validierung der Verarbeitung von NULL-Werten
     */
    async function testNullValues() {
        if (!window.VXORUtils) {
            throw new Error('VXORUtils nicht verfügbar');
        }
        
        // Wir erwarten keinen Fehler, nur eine korrekte Behandlung von NULL-Werten
        let hasErrors = false;
        let updateReceived = false;
        
        const dataUpdateHandler = () => {
            updateReceived = true;
        };
        
        const errorHandler = () => {
            hasErrors = true;
        };
        
        document.addEventListener(VXORUtils.EventTypes.DATA_UPDATED, dataUpdateHandler);
        document.addEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        
        try {
            // Daten-Update mit NULL-Werten simulieren
            document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.DATA_UPDATED, {
                detail: { 
                    category: 'matrix', 
                    data: TEST_DATA.nullValues 
                }
            }));
            
            // Kurze Verzögerung für Datenverarbeitung
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Prüfungen durchführen
            if (!updateReceived) {
                throw new Error('Daten-Update-Event wurde nicht empfangen');
            }
            
            // Wir erwarten keine JavaScript-Fehler
            if (hasErrors) {
                throw new Error('Fehler bei der Verarbeitung von NULL-Werten');
            }
            
            // Prüfen, ob UI-Elemente korrekt aktualisiert wurden
            // (Dies kann nur indirekt getestet werden)
            
            return true;
        }
        finally {
            // Event-Handler entfernen
            document.removeEventListener(VXORUtils.EventTypes.DATA_UPDATED, dataUpdateHandler);
            document.removeEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        }
    }
    
    /**
     * Test 4: Validierung der Verarbeitung extremer Werte
     */
    async function testExtremeValues() {
        if (!window.VXORUtils) {
            throw new Error('VXORUtils nicht verfügbar');
        }
        
        // Wir erwarten keinen Fehler, nur eine korrekte Behandlung extremer Werte
        let hasErrors = false;
        let updateReceived = false;
        
        const dataUpdateHandler = () => {
            updateReceived = true;
        };
        
        const errorHandler = () => {
            hasErrors = true;
        };
        
        document.addEventListener(VXORUtils.EventTypes.DATA_UPDATED, dataUpdateHandler);
        document.addEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        
        try {
            // Daten-Update mit extremen Werten simulieren
            document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.DATA_UPDATED, {
                detail: { 
                    category: 'matrix', 
                    data: TEST_DATA.extremeValues 
                }
            }));
            
            // Kurze Verzögerung für Datenverarbeitung
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Prüfungen durchführen
            if (!updateReceived) {
                throw new Error('Daten-Update-Event wurde nicht empfangen');
            }
            
            // Wir erwarten keine JavaScript-Fehler
            if (hasErrors) {
                throw new Error('Fehler bei der Verarbeitung extremer Werte');
            }
            
            // Charts sollten noch immer angezeigt werden
            const charts = document.querySelectorAll('canvas');
            if (charts.length === 0) {
                throw new Error('Keine Charts nach Datenaktualisierung mit extremen Werten gefunden');
            }
            
            return true;
        }
        finally {
            // Event-Handler entfernen
            document.removeEventListener(VXORUtils.EventTypes.DATA_UPDATED, dataUpdateHandler);
            document.removeEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        }
    }
    
    /**
     * Test 5: Validierung der Fehlerbehandlung
     */
    async function testErrorHandling() {
        if (!window.VXORUtils || typeof VXORUtils.handleApiError !== 'function') {
            throw new Error('VXORUtils.handleApiError nicht verfügbar');
        }
        
        // Wir erwarten einen Fehler und eine korrekte Fehlerbehandlung
        let errorReceived = false;
        
        const errorHandler = () => {
            errorReceived = true;
        };
        
        document.addEventListener(VXORUtils.EventTypes.ERROR, errorHandler);
        
        try {
            // Container für Fehlermeldung erstellen, falls nicht vorhanden
            let testContainer = document.getElementById('test-error-container');
            if (!testContainer) {
                testContainer = document.createElement('div');
                testContainer.id = 'test-error-container';
                document.body.appendChild(testContainer);
            }
            
            // Fehlerfall simulieren
            VXORUtils.handleApiError(
                new Error(TEST_DATA.error.error),
                'test-category',
                'test-error-container'
            );
            
            // Kurze Verzögerung für Fehlerverarbeitung
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Prüfungen durchführen
            if (!errorReceived) {
                throw new Error('Fehler-Event wurde nicht empfangen');
            }
            
            // Prüfen, ob Fehlermeldung angezeigt wird
            const errorMessage = document.querySelector('#test-error-container .error-message');
            if (!errorMessage) {
                throw new Error('Keine Fehlermeldung nach simuliertem API-Fehler angezeigt');
            }
            
            // Prüfen, ob Retry-Button vorhanden ist
            const retryButton = document.querySelector('#test-error-container .retry-button') || 
                               document.querySelector('#test-error-container button');
            if (!retryButton) {
                throw new Error('Kein Retry-Button in der Fehlermeldung gefunden');
            }
            
            // Container aufräumen
            testContainer.innerHTML = '';
            
            return true;
        }
        finally {
            // Event-Handler entfernen
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
    VXORTestRunner.dataIntegrityTests = VXORDataIntegrityTests.runTests;
}
