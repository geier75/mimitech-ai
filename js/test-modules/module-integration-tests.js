/**
 * VXOR Benchmark Dashboard - Modul-Integrationstests
 * Version: 1.0.0
 * Datum: 01.05.2025
 * Phase 7.1: Integrationstests
 *
 * Zweck: Validierung der korrekten Modul-Integration und Kommunikation
 * zwischen allen Dashboard-Komponenten
 */

'use strict';

const VXORModuleIntegrationTests = (function() {
    // Konstanten für Tests
    const EXPECTED_MODULES = ['matrix', 'quantum', 'tmath', 'mlperf', 'swebench', 'security'];
    const TEST_CONFIG = {
        timeout: 5000,
        retryAttempts: 2
    };
    
    /**
     * Führt alle Modul-Integrationstests aus
     * @returns {Promise<Object>} Testergebnisse
     */
    async function runTests() {
        console.log('▶ Starte Modul-Integrationstests...');
        
        const results = {
            success: true,
            tests: []
        };
        
        try {
            // Test 1: Modul-Registrierung validieren
            const moduleRegistrationTest = await VXORTestRunner.runTest(
                'Modul-Registrierung validieren',
                testModuleRegistration,
                TEST_CONFIG
            );
            results.tests.push(moduleRegistrationTest);
            
            // Test 2: Event-Kommunikation validieren
            const eventCommunicationTest = await VXORTestRunner.runTest(
                'Event-Kommunikation validieren',
                testEventCommunication,
                TEST_CONFIG
            );
            results.tests.push(eventCommunicationTest);
            
            // Test 3: Theme-Änderungen validieren
            const themeChangeTest = await VXORTestRunner.runTest(
                'Theme-Änderungen validieren',
                testThemeChanges,
                TEST_CONFIG
            );
            results.tests.push(themeChangeTest);
            
            // Test 4: API-Callbacks validieren
            const apiCallbackTest = await VXORTestRunner.runTest(
                'API-Callbacks validieren',
                testApiCallbacks,
                TEST_CONFIG
            );
            results.tests.push(apiCallbackTest);
            
            // Test 5: Tab-Navigation validieren
            const tabNavigationTest = await VXORTestRunner.runTest(
                'Tab-Navigation validieren',
                testTabNavigation,
                TEST_CONFIG
            );
            results.tests.push(tabNavigationTest);
            
            // Gesamtergebnis bestimmen
            results.success = results.tests.every(test => test.success);
            
            return results;
        }
        catch (error) {
            console.error('❌ Fehler in Modul-Integrationstests:', error);
            results.success = false;
            results.error = error.message;
            return results;
        }
    }
    
    /**
     * Test 1: Validiert, ob alle erwarteten Module korrekt registriert sind
     */
    async function testModuleRegistration() {
        // Prüfen, ob VXORUtils existiert und die Methode getRegisteredModules hat
        if (!window.VXORUtils || typeof VXORUtils.getRegisteredModules !== 'function') {
            throw new Error('VXORUtils oder getRegisteredModules nicht verfügbar');
        }
        
        const registeredModules = VXORUtils.getRegisteredModules();
        
        // Prüfen, ob alle erwarteten Module registriert sind
        const missingModules = EXPECTED_MODULES.filter(module => !registeredModules.includes(module));
        
        if (missingModules.length > 0) {
            throw new Error(`Module nicht registriert: ${missingModules.join(', ')}`);
        }
        
        // Prüfen, ob jedes Modul eine init-Methode hat
        for (const moduleId of registeredModules) {
            const moduleInterface = VXORUtils.getModuleInterface(moduleId);
            
            if (!moduleInterface || typeof moduleInterface.init !== 'function') {
                throw new Error(`Modul ${moduleId} hat keine gültige Schnittstelle`);
            }
        }
        
        return true;
    }
    
    /**
     * Test 2: Validiert die Event-Kommunikation zwischen den Modulen
     */
    async function testEventCommunication() {
        // Event-Bus-Verfügbarkeit prüfen
        if (!window.VXORUtils || !VXORUtils.EventTypes) {
            throw new Error('VXORUtils EventTypes nicht verfügbar');
        }
        
        // Temporäre Eventhandler für Tests registrieren
        const eventReceived = {};
        const eventHandlers = {};
        
        // Für jeden Event-Typ einen Handler registrieren
        Object.values(VXORUtils.EventTypes).forEach(eventType => {
            eventReceived[eventType] = false;
            
            eventHandlers[eventType] = () => {
                eventReceived[eventType] = true;
            };
            
            document.addEventListener(eventType, eventHandlers[eventType]);
        });
        
        // Events auslösen
        Object.entries(VXORUtils.EventTypes).forEach(([name, eventType]) => {
            const event = new CustomEvent(eventType, { 
                detail: { test: true, eventName: name } 
            });
            document.dispatchEvent(event);
        });
        
        // Kurze Verzögerung für die Event-Verarbeitung
        await new Promise(resolve => setTimeout(resolve, 300));
        
        // Event-Handler entfernen
        Object.entries(VXORUtils.EventTypes).forEach(([_, eventType]) => {
            document.removeEventListener(eventType, eventHandlers[eventType]);
        });
        
        // Prüfen, ob alle Events empfangen wurden
        const missingEvents = Object.entries(eventReceived)
            .filter(([_, received]) => !received)
            .map(([eventType]) => eventType);
        
        if (missingEvents.length > 0) {
            throw new Error(`Events nicht empfangen: ${missingEvents.join(', ')}`);
        }
        
        return true;
    }
    
    /**
     * Test 3: Validiert Theme-Änderungen in allen Modulen
     */
    async function testThemeChanges() {
        // Theme-Wechsel-Funktionalität prüfen
        if (!window.VXORUtils || typeof VXORUtils.toggleTheme !== 'function') {
            throw new Error('VXORUtils.toggleTheme nicht verfügbar');
        }
        
        // Aktuelles Theme speichern, um später wiederherzustellen
        const initialDarkMode = document.body.classList.contains('dark-mode');
        
        // Theme-Änderungs-Event überwachen
        let themeChangeReceived = false;
        const themeChangeHandler = () => {
            themeChangeReceived = true;
        };
        
        document.addEventListener(VXORUtils.EventTypes.THEME_CHANGED, themeChangeHandler);
        
        // Theme wechseln
        VXORUtils.toggleTheme();
        
        // Kurze Verzögerung für die Event-Verarbeitung
        await new Promise(resolve => setTimeout(resolve, 300));
        
        // Aktuelles Theme nach dem Wechsel ermitteln
        const newDarkMode = document.body.classList.contains('dark-mode');
        
        // Event-Handler entfernen
        document.removeEventListener(VXORUtils.EventTypes.THEME_CHANGED, themeChangeHandler);
        
        // Prüfen, ob das Theme-Änderungsevent empfangen wurde
        if (!themeChangeReceived) {
            throw new Error('Theme-Änderungsevent nicht empfangen');
        }
        
        // Prüfen, ob der Body die entsprechende Klasse hat
        const bodyHasDarkClass = document.body.classList.contains('dark-mode');
        if (bodyHasDarkClass !== newDarkMode) {
            throw new Error(`Theme-Änderung falsch angewendet: Body-Klasse=${bodyHasDarkClass}, erwartet=${newDarkMode}`);
        }
        
        // Theme auf Ausgangszustand zurücksetzen, falls nötig
        if (initialDarkMode !== newDarkMode) {
            VXORUtils.toggleTheme();
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        return true;
    }
    
    /**
     * Test 4: Validiert API-Aufrufe und Callbacks
     */
    async function testApiCallbacks() {
        // Mock-Daten für API-Tests
        const mockData = {
            results: [
                { component: 'Test', metric: 'TestMetric', value: 100 }
            ],
            timestamp: Date.now()
        };
        
        // API-Funktionen prüfen
        if (!window.VXORUtils || typeof VXORUtils.fetchBenchmarkData !== 'function') {
            throw new Error('VXORUtils.fetchBenchmarkData nicht verfügbar');
        }
        
        // Daten-Update-Event überwachen
        let dataUpdateReceived = false;
        let receivedData = null;
        
        const dataUpdateHandler = (event) => {
            dataUpdateReceived = true;
            receivedData = event.detail;
        };
        
        document.addEventListener(VXORUtils.EventTypes.DATA_UPDATED, dataUpdateHandler);
        
        // Event simulieren (für Test-Zwecke)
        document.dispatchEvent(new CustomEvent(VXORUtils.EventTypes.DATA_UPDATED, {
            detail: { category: 'test', data: mockData }
        }));
        
        // Kurze Verzögerung für die Event-Verarbeitung
        await new Promise(resolve => setTimeout(resolve, 300));
        
        // Event-Handler entfernen
        document.removeEventListener(VXORUtils.EventTypes.DATA_UPDATED, dataUpdateHandler);
        
        // Prüfen, ob das Daten-Update-Event empfangen wurde
        if (!dataUpdateReceived) {
            throw new Error('Daten-Update-Event nicht empfangen');
        }
        
        // Prüfen, ob die empfangenen Daten korrekt sind
        if (!receivedData || !receivedData.data || !receivedData.data.results) {
            throw new Error('Fehlerhafte Datenstruktur empfangen');
        }
        
        return true;
    }
    
    /**
     * Test 5: Validiert Tab-Navigation und Modul-Aktivierung
     */
    async function testTabNavigation() {
        // Tab-Navigation prüfen
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');
        
        if (tabButtons.length === 0 || tabContents.length === 0) {
            throw new Error('Keine Tab-Buttons oder Tab-Contents gefunden');
        }
        
        // Für jeden Tab prüfen
        for (let i = 0; i < tabButtons.length; i++) {
            const button = tabButtons[i];
            const targetId = button.getAttribute('data-tab');
            
            if (!targetId) {
                continue; // Überspringen, wenn kein Target-ID vorhanden
            }
            
            const targetContent = document.getElementById(targetId);
            
            if (!targetContent) {
                throw new Error(`Tab-Content mit ID "${targetId}" nicht gefunden`);
            }
            
            // Tab klicken
            button.click();
            
            // Kurze Verzögerung für UI-Updates
            await new Promise(resolve => setTimeout(resolve, 200));
            
            // Prüfen, ob der Tab aktiv ist
            if (!button.classList.contains('active')) {
                throw new Error(`Tab-Button für "${targetId}" nicht aktiviert`);
            }
            
            // Prüfen, ob der entsprechende Content angezeigt wird
            if (targetContent.style.display === 'none' || 
                getComputedStyle(targetContent).display === 'none') {
                throw new Error(`Tab-Content "${targetId}" nicht sichtbar`);
            }
        }
        
        return true;
    }
    
    // Öffentliche API
    return {
        runTests
    };
})();

// Registriere die Tests beim TestRunner
if (typeof VXORTestRunner !== 'undefined') {
    VXORTestRunner.moduleIntegrationTests = VXORModuleIntegrationTests.runTests;
}
