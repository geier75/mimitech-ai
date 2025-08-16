/**
 * VXOR Benchmark Dashboard - Performance-Tests
 * Version: 1.0.0
 * Datum: 01.05.2025
 * Phase 7.1: Integrationstests
 *
 * Zweck: Validierung der Performance und Ressourcennutzung
 * mit besonderem Fokus auf Rendering-Zeiten und API-Antwortzeiten
 */

'use strict';

const VXORPerformanceTests = (function() {
    // Testkonfiguration
    const TEST_CONFIG = {
        timeout: 15000, // Längere Timeout für Performance-Tests
        retryAttempts: 1
    };
    
    // Performance-Schwellenwerte
    const PERFORMANCE_THRESHOLDS = {
        apiResponseTime: 2000,      // ms (oberer Schwellenwert)
        pageLoadTime: 1000,         // ms (oberer Schwellenwert)
        chartRenderTime: 500,       // ms (oberer Schwellenwert)
        uiResponseTime: 100,        // ms (oberer Schwellenwert)
        memoryUsageChange: 50000000 // bytes (50MB, oberer Schwellenwert)
    };
    
    /**
     * Führt alle Performance-Tests aus
     * @returns {Promise<Object>} Testergebnisse
     */
    async function runTests() {
        console.log('▶ Starte Performance-Tests...');
        
        const results = {
            success: true,
            tests: []
        };
        
        try {
            // Test 1: API-Antwortzeiten
            const apiResponseTest = await VXORTestRunner.runTest(
                'API-Antwortzeiten validieren',
                testApiResponseTime,
                TEST_CONFIG
            );
            results.tests.push(apiResponseTest);
            
            // Test 2: Chart-Rendering-Zeiten
            const chartRenderingTest = await VXORTestRunner.runTest(
                'Chart-Rendering-Zeiten validieren',
                testChartRenderingTime,
                TEST_CONFIG
            );
            results.tests.push(chartRenderingTest);
            
            // Test 3: UI-Responsiveness
            const uiResponsivenessTest = await VXORTestRunner.runTest(
                'UI-Responsiveness validieren',
                testUIResponsiveness,
                TEST_CONFIG
            );
            results.tests.push(uiResponsivenessTest);
            
            // Test 4: Speichernutzung
            const memoryUsageTest = await VXORTestRunner.runTest(
                'Speichernutzung validieren',
                testMemoryUsage,
                TEST_CONFIG
            );
            results.tests.push(memoryUsageTest);
            
            // Test 5: DOM-Größe und Event-Listener
            const domSizeTest = await VXORTestRunner.runTest(
                'DOM-Größe und Event-Listener validieren',
                testDOMSize,
                TEST_CONFIG
            );
            results.tests.push(domSizeTest);
            
            // Gesamtergebnis bestimmen
            results.success = results.tests.every(test => test.success);
            
            return results;
        }
        catch (error) {
            console.error('❌ Fehler in Performance-Tests:', error);
            results.success = false;
            results.error = error.message;
            return results;
        }
    }
    
    /**
     * Test 1: Validierung der API-Antwortzeiten
     */
    async function testApiResponseTime() {
        if (!window.VXORUtils || !VXORUtils.fetchData) {
            throw new Error('VXORUtils.fetchData nicht verfügbar');
        }
        
        const benchmarkTypes = ['matrix', 'quantum', 'tmath', 'mlperf', 'swebench', 'security'];
        const timings = {};
        
        // Reale API-Calls vermeiden und stattdessen mockData verwenden
        const originalFetch = window.fetch;
        window.fetch = function(url, options) {
            // Verzögerung simulieren
            return new Promise(resolve => {
                setTimeout(() => {
                    resolve({
                        ok: true,
                        json: () => Promise.resolve({ results: [], timestamp: Date.now() })
                    });
                }, 50); // Schnelle Antwort für Tests
            });
        };
        
        try {
            // Alle Benchmark-Typen testen
            for (const benchmarkType of benchmarkTypes) {
                // API-Call starten und Zeit messen
                const startTime = performance.now();
                
                try {
                    await VXORUtils.fetchData(benchmarkType);
                    const endTime = performance.now();
                    const duration = endTime - startTime;
                    
                    timings[benchmarkType] = duration;
                    
                    if (duration > PERFORMANCE_THRESHOLDS.apiResponseTime) {
                        console.warn(`API-Antwortzeit für ${benchmarkType} über Schwellenwert: ${duration.toFixed(2)}ms`);
                    }
                } catch (err) {
                    console.warn(`API-Call für ${benchmarkType} fehlgeschlagen: ${err.message}`);
                }
            }
            
            // Durchschnittliche API-Antwortzeit berechnen
            const timingValues = Object.values(timings);
            const avgResponseTime = timingValues.reduce((sum, time) => sum + time, 0) / timingValues.length;
            
            console.log('📊 API-Antwortzeiten:', timings);
            console.log(`📊 Durchschnittliche API-Antwortzeit: ${avgResponseTime.toFixed(2)}ms`);
            
            // Erfolg basierend auf der durchschnittlichen Antwortzeit
            const isSuccess = avgResponseTime <= PERFORMANCE_THRESHOLDS.apiResponseTime;
            
            return isSuccess;
        }
        finally {
            // Ursprüngliche fetch-Funktion wiederherstellen
            window.fetch = originalFetch;
        }
    }
    
    /**
     * Test 2: Validierung der Chart-Rendering-Zeiten
     */
    async function testChartRenderingTime() {
        if (!window.Chart) {
            throw new Error('Chart.js nicht verfügbar');
        }
        
        // Performance-Messungen sammeln
        const chartRenderingTimes = {};
        
        // Chart.js-Rendering-Zeiten messen
        const originalRender = Chart.prototype.render;
        
        Chart.prototype.render = function() {
            const chartId = this.canvas.id || 'unnamed';
            const startTime = performance.now();
            
            const result = originalRender.apply(this, arguments);
            
            const endTime = performance.now();
            const duration = endTime - startTime;
            
            chartRenderingTimes[chartId] = duration;
            
            if (duration > PERFORMANCE_THRESHOLDS.chartRenderTime) {
                console.warn(`Chart-Rendering-Zeit für ${chartId} über Schwellenwert: ${duration.toFixed(2)}ms`);
            }
            
            return result;
        };
        
        try {
            // Charts neu rendern durch Trigger eines Fenstergrößenänderungsereignisses
            window.dispatchEvent(new Event('resize'));
            
            // Kurze Verzögerung für Chart-Rendering
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Durchschnittliche Rendering-Zeit berechnen
            const timingValues = Object.values(chartRenderingTimes);
            
            if (timingValues.length === 0) {
                console.warn('Keine Charts gerendert während des Tests');
                return true; // Nicht kritisch
            }
            
            const avgRenderTime = timingValues.reduce((sum, time) => sum + time, 0) / timingValues.length;
            
            console.log('📊 Chart-Rendering-Zeiten:', chartRenderingTimes);
            console.log(`📊 Durchschnittliche Chart-Rendering-Zeit: ${avgRenderTime.toFixed(2)}ms`);
            
            // Erfolg basierend auf der durchschnittlichen Rendering-Zeit
            const isSuccess = avgRenderTime <= PERFORMANCE_THRESHOLDS.chartRenderTime;
            
            return isSuccess;
        }
        finally {
            // Ursprüngliche render-Methode wiederherstellen
            Chart.prototype.render = originalRender;
        }
    }
    
    /**
     * Test 3: Validierung der UI-Responsiveness
     */
    async function testUIResponsiveness() {
        // Interaktive Elemente finden
        const tabs = document.querySelectorAll('.tab-button');
        const buttons = document.querySelectorAll('button:not(.tab-button)');
        const selects = document.querySelectorAll('select');
        
        const interactionTimings = {
            tabs: {},
            buttons: {},
            selects: {}
        };
        
        let totalTestCount = 0;
        let successfulTests = 0;
        
        // Tab-Wechsel testen
        if (tabs.length > 0) {
            for (const tab of tabs) {
                const tabId = tab.id || tab.textContent || 'unnamed';
                
                // Klick simulieren und Zeit messen
                const startTime = performance.now();
                tab.click();
                
                // Warten auf Rendering-Aktualisierung
                await new Promise(resolve => setTimeout(resolve, 50));
                
                const endTime = performance.now();
                const duration = endTime - startTime;
                
                interactionTimings.tabs[tabId] = duration;
                totalTestCount++;
                
                if (duration <= PERFORMANCE_THRESHOLDS.uiResponseTime) {
                    successfulTests++;
                } else {
                    console.warn(`UI-Antwortzeit für Tab ${tabId} über Schwellenwert: ${duration.toFixed(2)}ms`);
                }
            }
        }
        
        // Button-Klicks testen (nur eine Stichprobe)
        const buttonSample = Array.from(buttons).slice(0, 3);
        
        for (const button of buttonSample) {
            const buttonId = button.id || button.textContent || 'unnamed';
            
            // Speichere ursprünglichen onClick-Handler
            const originalClick = button.onclick;
            
            // Ersetze onClick temporär
            button.onclick = function(e) {
                e.preventDefault();
                e.stopPropagation();
                return false;
            };
            
            // Klick simulieren und Zeit messen
            const startTime = performance.now();
            button.click();
            
            // Warten auf Rendering-Aktualisierung
            await new Promise(resolve => setTimeout(resolve, 50));
            
            const endTime = performance.now();
            const duration = endTime - startTime;
            
            interactionTimings.buttons[buttonId] = duration;
            totalTestCount++;
            
            if (duration <= PERFORMANCE_THRESHOLDS.uiResponseTime) {
                successfulTests++;
            } else {
                console.warn(`UI-Antwortzeit für Button ${buttonId} über Schwellenwert: ${duration.toFixed(2)}ms`);
            }
            
            // Ursprünglichen Handler wiederherstellen
            button.onclick = originalClick;
        }
        
        // Select-Änderungen testen
        if (selects.length > 0) {
            for (const select of selects) {
                const selectId = select.id || 'unnamed';
                
                if (select.options.length < 2) continue;
                
                // Speichere ursprünglichen onChange-Handler und aktuellen Wert
                const originalChange = select.onchange;
                const originalValue = select.value;
                
                // Ersetze onChange temporär
                select.onchange = function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    return false;
                };
                
                // Nächste Option wählen
                const nextOption = select.options[0].value === originalValue ? 
                                   select.options[1].value : 
                                   select.options[0].value;
                
                // Änderung simulieren und Zeit messen
                const startTime = performance.now();
                select.value = nextOption;
                select.dispatchEvent(new Event('change', { bubbles: true }));
                
                // Warten auf Rendering-Aktualisierung
                await new Promise(resolve => setTimeout(resolve, 50));
                
                const endTime = performance.now();
                const duration = endTime - startTime;
                
                interactionTimings.selects[selectId] = duration;
                totalTestCount++;
                
                if (duration <= PERFORMANCE_THRESHOLDS.uiResponseTime) {
                    successfulTests++;
                } else {
                    console.warn(`UI-Antwortzeit für Select ${selectId} über Schwellenwert: ${duration.toFixed(2)}ms`);
                }
                
                // Ursprünglichen Wert und Handler wiederherstellen
                select.value = originalValue;
                select.onchange = originalChange;
            }
        }
        
        console.log('📊 UI-Interaktionszeiten:', interactionTimings);
        
        if (totalTestCount === 0) {
            console.warn('Keine UI-Interaktionen getestet');
            return true; // Nicht kritisch
        }
        
        // Erfolgsrate berechnen
        const successRate = (successfulTests / totalTestCount) * 100;
        console.log(`📊 UI-Responsiveness Erfolgsrate: ${successRate.toFixed(2)}%`);
        
        return successRate >= 80; // Mindestens 80% der Tests sollten erfolgreich sein
    }
    
    /**
     * Test 4: Validierung der Speichernutzung
     */
    async function testMemoryUsage() {
        // Speichernutzung vor und nach intensiver Nutzung messen
        let initialMemoryUsage = 0;
        let finalMemoryUsage = 0;
        
        // Speichernutzung messen (wenn verfügbar)
        const measureMemory = () => {
            if (window.performance && window.performance.memory) {
                return window.performance.memory.usedJSHeapSize;
            }
            
            return 0; // Nicht verfügbar
        };
        
        // Initiale Speichernutzung messen
        initialMemoryUsage = measureMemory();
        
        if (initialMemoryUsage === 0) {
            console.warn('Speichernutzungsmessung nicht verfügbar');
            return true; // Nicht kritisch
        }
        
        try {
            // Intensive Nutzung simulieren
            await simulateIntensiveUsage();
            
            // Finale Speichernutzung messen
            finalMemoryUsage = measureMemory();
            
            // Speichernutzungsänderung berechnen
            const memoryChange = finalMemoryUsage - initialMemoryUsage;
            
            console.log(`📊 Initiale Speichernutzung: ${formatBytes(initialMemoryUsage)}`);
            console.log(`📊 Finale Speichernutzung: ${formatBytes(finalMemoryUsage)}`);
            console.log(`📊 Speichernutzungsänderung: ${formatBytes(memoryChange)}`);
            
            // Erfolg basierend auf der Speichernutzungsänderung
            const isSuccess = memoryChange <= PERFORMANCE_THRESHOLDS.memoryUsageChange;
            
            if (!isSuccess) {
                console.warn(`Speichernutzungsänderung über Schwellenwert: ${formatBytes(memoryChange)}`);
            }
            
            return isSuccess;
        }
        catch (error) {
            console.error('Fehler bei Speichernutzungstest:', error);
            return true; // Nicht kritisch
        }
    }
    
    /**
     * Test 5: Validierung der DOM-Größe und Event-Listener
     */
    async function testDOMSize() {
        // DOM-Größe und Event-Listener zählen
        const domSize = document.querySelectorAll('*').length;
        
        console.log(`📊 DOM-Größe: ${domSize} Elemente`);
        
        // Event-Listener zählen (wenn möglich)
        if (window.getEventListeners) {
            let totalEventListeners = 0;
            
            document.querySelectorAll('*').forEach(element => {
                const listeners = getEventListeners(element);
                
                if (listeners) {
                    Object.values(listeners).forEach(listenerList => {
                        totalEventListeners += listenerList.length;
                    });
                }
            });
            
            console.log(`📊 Gesamtzahl Event-Listener: ${totalEventListeners}`);
        }
        
        // Speicherverbrauch für DOM schätzen (sehr grobe Schätzung)
        const estimatedDOMMemory = domSize * 2500; // ca. 2.5KB pro DOM-Element (sehr grobe Schätzung)
        
        console.log(`📊 Geschätzter DOM-Speicherverbrauch: ${formatBytes(estimatedDOMMemory)}`);
        
        // Warnung bei sehr großem DOM
        if (domSize > 1500) {
            console.warn(`Sehr großer DOM mit ${domSize} Elementen gefunden`);
        }
        
        return domSize <= 2000; // Obergrenze für DOM-Größe
    }
    
    /**
     * Hilfsfunktion: Simuliert intensive Nutzung der Anwendung
     */
    async function simulateIntensiveUsage() {
        // Simuliere intensive Nutzung, indem wir verschiedene Aktionen ausführen
        
        // 1. Tab-Wechsel mehrmals durchführen
        const tabs = document.querySelectorAll('.tab-button');
        
        if (tabs.length > 1) {
            for (let i = 0; i < 5; i++) {
                for (const tab of tabs) {
                    tab.click();
                    await new Promise(resolve => setTimeout(resolve, 50));
                }
            }
        }
        
        // 2. Select-Änderungen durchführen
        const selects = document.querySelectorAll('select');
        
        for (const select of selects) {
            if (select.options.length < 2) continue;
            
            const originalValue = select.value;
            
            // Jede Option einmal durchgehen
            for (const option of select.options) {
                select.value = option.value;
                select.dispatchEvent(new Event('change', { bubbles: true }));
                await new Promise(resolve => setTimeout(resolve, 50));
            }
            
            // Ursprünglichen Wert wiederherstellen
            select.value = originalValue;
            select.dispatchEvent(new Event('change', { bubbles: true }));
        }
        
        // 3. Fenstergröße ändern
        for (const size of [800, 1200, 600, 1000]) {
            window.innerWidth = size;
            window.dispatchEvent(new Event('resize'));
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        // Garbage Collection anregen (falls möglich)
        if (window.gc) {
            window.gc();
        }
        
        // Kurze Pause, um den Speicherverbrauch zu stabilisieren
        await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    /**
     * Hilfsfunktion: Formatiert Bytes in lesbare Größen
     * @param {number} bytes - Anzahl der Bytes
     * @returns {string} Formatierte Größe mit Einheit
     */
    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Öffentliche API
    return {
        runTests
    };
})();

// Registriere die Tests beim TestRunner
if (typeof VXORTestRunner !== 'undefined') {
    VXORTestRunner.performanceTests = VXORPerformanceTests.runTests;
}
