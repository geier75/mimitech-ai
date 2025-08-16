/**
 * VXOR Benchmark Dashboard - UI-Konsistenztests
 * Version: 1.0.0
 * Datum: 01.05.2025
 * Phase 7.1: Integrationstests
 *
 * Zweck: Validierung der korrekten und konsistenten Benutzeroberfläche
 * mit besonderem Fokus auf Responsiveness und visuelle Konsistenz
 */

'use strict';

const VXORUIConsistencyTests = (function() {
    // Testkonfiguration
    const TEST_CONFIG = {
        timeout: 8000, // Längere Timeout für UI-Tests
        retryAttempts: 2
    };
    
    // Responsive Breakpoints für Tests
    const VIEWPORT_SIZES = [
        { width: 320, height: 568, name: 'Mobile S' },
        { width: 375, height: 667, name: 'Mobile M' },
        { width: 414, height: 736, name: 'Mobile L' },
        { width: 768, height: 1024, name: 'Tablet' },
        { width: 1024, height: 768, name: 'Laptop' },
        { width: 1440, height: 900, name: 'Desktop' }
    ];
    
    /**
     * Führt alle UI-Konsistenztests aus
     * @returns {Promise<Object>} Testergebnisse
     */
    async function runTests() {
        console.log('▶ Starte UI-Konsistenztests...');
        
        const results = {
            success: true,
            tests: []
        };
        
        try {
            // Test 1: Tab-Navigation und Content-Anzeige
            const tabNavigationTest = await VXORTestRunner.runTest(
                'Tab-Navigation und Content-Anzeige validieren',
                testTabNavigation,
                TEST_CONFIG
            );
            results.tests.push(tabNavigationTest);
            
            // Test 2: Responsive Layout
            const responsiveLayoutTest = await VXORTestRunner.runTest(
                'Responsive Layout validieren',
                testResponsiveLayout,
                TEST_CONFIG
            );
            results.tests.push(responsiveLayoutTest);
            
            // Test 3: Chart-Darstellung
            const chartRenderingTest = await VXORTestRunner.runTest(
                'Chart-Darstellung validieren',
                testChartRendering,
                TEST_CONFIG
            );
            results.tests.push(chartRenderingTest);
            
            // Test 4: Selektoren und Filter
            const selectorsAndFiltersTest = await VXORTestRunner.runTest(
                'Selektoren und Filter validieren',
                testSelectorsAndFilters,
                TEST_CONFIG
            );
            results.tests.push(selectorsAndFiltersTest);
            
            // Test 5: Dark Mode / Light Mode
            const themingTest = await VXORTestRunner.runTest(
                'Theming validieren',
                testTheming,
                TEST_CONFIG
            );
            results.tests.push(themingTest);
            
            // Gesamtergebnis bestimmen
            results.success = results.tests.every(test => test.success);
            
            return results;
        }
        catch (error) {
            console.error('❌ Fehler in UI-Konsistenztests:', error);
            results.success = false;
            results.error = error.message;
            return results;
        }
    }
    
    /**
     * Test 1: Validierung der Tab-Navigation und Content-Anzeige
     */
    async function testTabNavigation() {
        // Tab-Navigation prüfen
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');
        
        if (tabButtons.length === 0) {
            throw new Error('Keine Tab-Buttons gefunden');
        }
        
        if (tabContents.length === 0) {
            throw new Error('Keine Tab-Contents gefunden');
        }
        
        // Speichere den ursprünglich aktiven Tab
        const originalActiveTab = Array.from(tabButtons).find(tab => tab.classList.contains('active'));
        let originalActiveTabId = null;
        
        if (originalActiveTab) {
            originalActiveTabId = originalActiveTab.getAttribute('data-tab');
        }
        
        // Prüfe jeden Tab einzeln
        for (const tabButton of tabButtons) {
            const targetId = tabButton.getAttribute('data-tab');
            
            if (!targetId) {
                console.warn(`Tab-Button ohne data-tab-Attribut gefunden: ${tabButton.textContent}`);
                continue;
            }
            
            const targetContent = document.getElementById(targetId);
            
            if (!targetContent) {
                throw new Error(`Tab-Content mit ID "${targetId}" nicht gefunden`);
            }
            
            // Tab aktivieren
            tabButton.click();
            
            // Kurze Verzögerung für UI-Updates
            await new Promise(resolve => setTimeout(resolve, 100));
            
            // Prüfungen
            if (!tabButton.classList.contains('active')) {
                throw new Error(`Tab-Button für "${targetId}" wurde nicht als aktiv markiert`);
            }
            
            // Prüfe, ob der Content angezeigt wird
            const contentStyle = getComputedStyle(targetContent);
            const isVisible = contentStyle.display !== 'none' && contentStyle.visibility !== 'hidden';
            
            if (!isVisible) {
                throw new Error(`Tab-Content "${targetId}" ist nicht sichtbar nach Aktivierung`);
            }
            
            // Prüfe, ob andere Tabs deaktiviert sind
            const otherButtons = Array.from(tabButtons).filter(btn => btn !== tabButton);
            for (const otherButton of otherButtons) {
                if (otherButton.classList.contains('active')) {
                    throw new Error('Mehrere Tab-Buttons gleichzeitig aktiv');
                }
            }
            
            // Prüfe, ob andere Contents ausgeblendet sind
            const otherContents = Array.from(tabContents).filter(content => content.id !== targetId);
            for (const otherContent of otherContents) {
                const otherStyle = getComputedStyle(otherContent);
                const isOtherVisible = otherStyle.display !== 'none' && otherStyle.visibility !== 'hidden';
                
                if (isOtherVisible) {
                    throw new Error(`Inaktiver Tab-Content "${otherContent.id}" ist sichtbar`);
                }
            }
        }
        
        // Stelle den ursprünglichen Tab wieder her
        if (originalActiveTabId) {
            const originalTab = Array.from(tabButtons).find(
                tab => tab.getAttribute('data-tab') === originalActiveTabId
            );
            
            if (originalTab) {
                originalTab.click();
                await new Promise(resolve => setTimeout(resolve, 100));
            }
        }
        
        return true;
    }
    
    /**
     * Test 2: Validierung des responsiven Layouts
     */
    async function testResponsiveLayout() {
        if (!window.VXORUtils) {
            throw new Error('VXORUtils nicht verfügbar');
        }
        
        // Ursprüngliche Fenstergröße speichern
        const originalWidth = window.innerWidth;
        const originalHeight = window.innerHeight;
        
        try {
            // Test für verschiedene Viewport-Größen
            for (const viewport of VIEWPORT_SIZES) {
                // Fenstergröße ändern
                await resizeWindow(viewport.width, viewport.height);
                
                // Kurze Verzögerung für UI-Updates
                await new Promise(resolve => setTimeout(resolve, 300));
                
                // Basiselemente prüfen
                const header = document.querySelector('.header');
                const mainContent = document.querySelector('.main-content');
                const tabs = document.querySelector('.tabs');
                
                if (!header || !mainContent || !tabs) {
                    throw new Error(`Basiselement(e) nicht gefunden bei Viewport ${viewport.name}`);
                }
                
                // Prüfen, ob das Header-Logo angezeigt wird
                const logo = document.querySelector('.header-logo, .logo');
                if (logo) {
                    const logoStyle = getComputedStyle(logo);
                    if (logoStyle.display === 'none' && viewport.width >= 768) {
                        throw new Error(`Logo sollte auf ${viewport.name} angezeigt werden`);
                    }
                }
                
                // Prüfen, ob alle Tabs angezeigt werden
                const tabButtons = tabs.querySelectorAll('.tab-button');
                if (tabButtons.length === 0) {
                    throw new Error(`Keine Tab-Buttons gefunden bei Viewport ${viewport.name}`);
                }
                
                // Prüfen, ob Charts responsive sind
                const charts = document.querySelectorAll('canvas');
                for (const chart of charts) {
                    const chartStyle = getComputedStyle(chart);
                    if (chartStyle.display === 'none') {
                        throw new Error(`Chart nicht angezeigt bei Viewport ${viewport.name}`);
                    }
                    
                    // Chart sollte eine vernünftige Größe haben
                    const chartWidth = chart.width;
                    const chartHeight = chart.height;
                    
                    if (chartWidth === 0 || chartHeight === 0) {
                        throw new Error(`Chart hat ungültige Größe (${chartWidth}x${chartHeight}) bei Viewport ${viewport.name}`);
                    }
                    
                    // Chart sollte nicht über den Viewport hinausragen
                    if (chartWidth > viewport.width) {
                        throw new Error(`Chart zu breit (${chartWidth}px) für Viewport ${viewport.name} (${viewport.width}px)`);
                    }
                }
            }
            
            return true;
        }
        finally {
            // Ursprüngliche Fenstergröße wiederherstellen
            await resizeWindow(originalWidth, originalHeight);
        }
    }
    
    /**
     * Test 3: Validierung der Chart-Darstellung
     */
    async function testChartRendering() {
        // Suche Chart-Elemente
        const canvases = document.querySelectorAll('canvas');
        
        if (canvases.length === 0) {
            throw new Error('Keine Canvas-Elemente gefunden');
        }
        
        // Prüfe jedes Canvas-Element
        for (const canvas of canvases) {
            // Prüfe, ob das Canvas gerendert wurde
            const width = canvas.width;
            const height = canvas.height;
            
            if (width === 0 || height === 0) {
                throw new Error(`Canvas mit ungültiger Größe gefunden: ${width}x${height}`);
            }
            
            // Prüfe, ob der Canvas einen Chart.js-Kontext hat
            const hasChartContext = canvas._chart || 
                                   canvas.__chartjs || 
                                   canvas.dataset.chartjs;
            
            if (!hasChartContext) {
                console.warn(`Canvas ohne Chart.js-Kontext gefunden: ${canvas.id || 'Unnamed'}`);
            }
            
            // Prüfe, ob das Canvas sichtbar ist
            const canvasStyle = getComputedStyle(canvas);
            if (canvasStyle.display === 'none' || canvasStyle.visibility === 'hidden') {
                throw new Error(`Verstecktes Canvas gefunden: ${canvas.id || 'Unnamed'}`);
            }
        }
        
        // Prüfe, ob die Diagramm-Container Labels haben
        const chartContainers = document.querySelectorAll('.chart-container, .benchmark-card');
        
        if (chartContainers.length === 0) {
            console.warn('Keine Chart-Container gefunden');
        } else {
            for (const container of chartContainers) {
                const headings = container.querySelectorAll('h1, h2, h3, h4, h5, h6');
                
                if (headings.length === 0) {
                    console.warn(`Chart-Container ohne Überschrift gefunden: ${container.id || 'Unnamed'}`);
                }
            }
        }
        
        return true;
    }
    
    /**
     * Test 4: Validierung der Selektoren und Filter
     */
    async function testSelectorsAndFilters() {
        // Suche Select-Elemente und Buttons, die als Filter dienen
        const selects = document.querySelectorAll('select');
        const filterButtons = document.querySelectorAll('button[data-filter], [role="button"][data-filter]');
        
        if (selects.length === 0 && filterButtons.length === 0) {
            console.warn('Keine Selektoren oder Filter-Buttons gefunden');
            return true; // Nicht kritisch
        }
        
        // Teste die Select-Elemente
        for (const select of selects) {
            // Prüfe, ob das Select Options hat
            const options = select.querySelectorAll('option');
            
            if (options.length === 0) {
                throw new Error(`Select ohne Optionen gefunden: ${select.id || 'Unnamed'}`);
            }
            
            // Prüfe, ob das Select ein Label hat
            const hasLabel = document.querySelector(`label[for="${select.id}"]`);
            
            if (!hasLabel && !select.getAttribute('aria-label')) {
                console.warn(`Select ohne Label oder ARIA-Label gefunden: ${select.id || 'Unnamed'}`);
            }
            
            // Prüfe, ob das Select einen Change-Event-Handler hat
            const hasChangeListener = select.onchange || 
                                     select.__handlers?.change?.length > 0;
            
            if (!hasChangeListener) {
                // Test für Event-Listener, die mit addEventListener hinzugefügt wurden
                const originalSelect = select.onchange;
                let changeDetected = false;
                
                select.onchange = function() {
                    changeDetected = true;
                    if (originalSelect) originalSelect.apply(this, arguments);
                };
                
                // Auswahl ändern
                const initialValue = select.value;
                const options = Array.from(select.options);
                
                if (options.length > 1) {
                    const newOption = options.find(opt => opt.value !== initialValue) || options[0];
                    select.value = newOption.value;
                    
                    // Change-Event auslösen
                    const changeEvent = new Event('change', { bubbles: true });
                    select.dispatchEvent(changeEvent);
                    
                    // Kurze Verzögerung für Event-Verarbeitung
                    await new Promise(resolve => setTimeout(resolve, 100));
                    
                    // Ursprünglichen Wert wiederherstellen
                    select.value = initialValue;
                    select.dispatchEvent(new Event('change', { bubbles: true }));
                    
                    // Handler zurücksetzen
                    select.onchange = originalSelect;
                    
                    if (!changeDetected) {
                        console.warn(`Select ohne funktionierenden Change-Handler gefunden: ${select.id || 'Unnamed'}`);
                    }
                }
            }
        }
        
        // Teste die Filter-Buttons
        for (const button of filterButtons) {
            // Prüfe, ob der Button einen Click-Event-Handler hat
            const hasClickListener = button.onclick || 
                                    button.__handlers?.click?.length > 0;
            
            if (!hasClickListener) {
                console.warn(`Filter-Button ohne Click-Handler gefunden: ${button.id || button.textContent || 'Unnamed'}`);
            }
            
            // Prüfe, ob der Button ein Label oder ARIA-Label hat
            if (!button.textContent.trim() && !button.getAttribute('aria-label')) {
                console.warn(`Filter-Button ohne Text oder ARIA-Label gefunden: ${button.id || 'Unnamed'}`);
            }
        }
        
        return true;
    }
    
    /**
     * Test 5: Validierung des Themings (Dark Mode / Light Mode)
     */
    async function testTheming() {
        if (!window.VXORUtils || typeof VXORUtils.toggleTheme !== 'function') {
            throw new Error('VXORUtils.toggleTheme nicht verfügbar');
        }
        
        // Ursprüngliches Theme speichern
        const originalDarkMode = document.body.classList.contains('dark-mode');
        
        try {
            // Zum Light-Mode wechseln (falls nicht bereits aktiv)
            if (originalDarkMode) {
                VXORUtils.toggleTheme();
                await new Promise(resolve => setTimeout(resolve, 200));
            }
            
            // Light-Mode-Prüfung
            await validateThemeConsistency('light');
            
            // Zum Dark-Mode wechseln
            VXORUtils.toggleTheme();
            await new Promise(resolve => setTimeout(resolve, 200));
            
            // Dark-Mode-Prüfung
            await validateThemeConsistency('dark');
            
            return true;
        }
        finally {
            // Ursprüngliches Theme wiederherstellen falls nötig
            const currentDarkMode = document.body.classList.contains('dark-mode');
            
            if (currentDarkMode !== originalDarkMode) {
                VXORUtils.toggleTheme();
            }
        }
    }
    
    /**
     * Prüft die Konsistenz eines Themes
     * @param {string} mode - 'light' oder 'dark'
     */
    async function validateThemeConsistency(mode) {
        const isDarkMode = mode === 'dark';
        const darkModeActive = document.body.classList.contains('dark-mode');
        
        // Prüfe, ob der Body die korrekte Klasse hat
        if (isDarkMode !== darkModeActive) {
            throw new Error(`Theme-Inkonsistenz: Body hat ${darkModeActive ? 'dark-mode' : 'nicht dark-mode'} für ${mode}-Modus`);
        }
        
        // Prüfe, ob die CSS-Variablen korrekt gesetzt sind
        const computedStyle = getComputedStyle(document.documentElement);
        const bgColor = computedStyle.getPropertyValue('--bg-color').trim();
        const textColor = computedStyle.getPropertyValue('--text-color').trim();
        
        // Implizierte Prüfung: Im Dark-Mode sollte der Hintergrund dunkel und der Text hell sein
        const isDarkBg = isColorDark(bgColor);
        const isDarkText = isColorDark(textColor);
        
        if (isDarkMode && !isDarkBg) {
            throw new Error(`Theme-Inkonsistenz: Hintergrundfarbe ${bgColor} ist nicht dunkel im Dark-Mode`);
        }
        
        if (isDarkMode && isDarkText) {
            throw new Error(`Theme-Inkonsistenz: Textfarbe ${textColor} ist nicht hell im Dark-Mode`);
        }
        
        if (!isDarkMode && isDarkBg) {
            throw new Error(`Theme-Inkonsistenz: Hintergrundfarbe ${bgColor} ist nicht hell im Light-Mode`);
        }
        
        if (!isDarkMode && !isDarkText) {
            throw new Error(`Theme-Inkonsistenz: Textfarbe ${textColor} ist nicht dunkel im Light-Mode`);
        }
        
        // Prüfe, ob Charts korrekt gerendert werden
        const canvases = document.querySelectorAll('canvas');
        
        if (canvases.length === 0) {
            console.warn('Keine Canvas-Elemente für Theme-Validierung gefunden');
        }
    }
    
    /**
     * Hilfsfunktion: Fenstergröße ändern
     * @param {number} width - Fensterbreite
     * @param {number} height - Fensterhöhe
     */
    async function resizeWindow(width, height) {
        // Fenstergröße ändern
        window.innerWidth = width;
        window.innerHeight = height;
        
        // Resize-Event auslösen
        window.dispatchEvent(new Event('resize'));
        
        // Kurze Verzögerung für UI-Updates
        await new Promise(resolve => setTimeout(resolve, 300));
    }
    
    /**
     * Hilfsfunktion: Prüft, ob eine Farbe dunkel ist
     * @param {string} color - CSS-Farbwert (#RRGGBB oder rgb(r, g, b))
     * @returns {boolean} True, wenn die Farbe dunkel ist
     */
    function isColorDark(color) {
        // Hex-Farbe zu RGB konvertieren
        let r, g, b;
        
        if (color.startsWith('#')) {
            // Hex-Format
            if (color.length === 4) {
                // Kurzformat #RGB
                r = parseInt(color[1] + color[1], 16);
                g = parseInt(color[2] + color[2], 16);
                b = parseInt(color[3] + color[3], 16);
            } else {
                // Normalformat #RRGGBB
                r = parseInt(color.substring(1, 3), 16);
                g = parseInt(color.substring(3, 5), 16);
                b = parseInt(color.substring(5, 7), 16);
            }
        } else if (color.startsWith('rgb')) {
            // RGB-Format
            const match = color.match(/rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/i);
            if (match) {
                r = parseInt(match[1]);
                g = parseInt(match[2]);
                b = parseInt(match[3]);
            } else {
                return false; // Unbekanntes Format
            }
        } else {
            return false; // Unbekanntes Format
        }
        
        // Relative Luminanz berechnen (vereinfacht)
        const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
        
        // Farbe gilt als dunkel, wenn Luminanz < 0.5
        return luminance < 0.5;
    }
    
    // Öffentliche API
    return {
        runTests
    };
})();

// Registriere die Tests beim TestRunner
if (typeof VXORTestRunner !== 'undefined') {
    VXORTestRunner.uiConsistencyTests = VXORUIConsistencyTests.runTests;
}
