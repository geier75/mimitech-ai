/**
 * VXOR Benchmark Dashboard - Barrierefreiheitstests
 * Version: 1.0.0
 * Datum: 01.05.2025
 * Phase 7.1: Integrationstests
 *
 * Zweck: Validierung der WCAG-Konformität und Screenreader-Unterstützung
 * mit besonderem Fokus auf ARIA-Attribute und Tastaturzugänglichkeit
 */

'use strict';

const VXORAccessibilityTests = (function() {
    // Testkonfiguration
    const TEST_CONFIG = {
        timeout: 8000, // Längere Timeout für Barrierefreiheitstests
        retryAttempts: 2
    };
    
    /**
     * Führt alle Barrierefreiheitstests aus
     * @returns {Promise<Object>} Testergebnisse
     */
    async function runTests() {
        console.log('▶ Starte Barrierefreiheitstests...');
        
        const results = {
            success: true,
            tests: []
        };
        
        try {
            // Test 1: ARIA-Landmarks
            const ariaLandmarksTest = await VXORTestRunner.runTest(
                'ARIA-Landmarks validieren',
                testAriaLandmarks,
                TEST_CONFIG
            );
            results.tests.push(ariaLandmarksTest);
            
            // Test 2: Semantische Struktur
            const semanticStructureTest = await VXORTestRunner.runTest(
                'Semantische Struktur validieren',
                testSemanticStructure,
                TEST_CONFIG
            );
            results.tests.push(semanticStructureTest);
            
            // Test 3: Tastaturzugänglichkeit
            const keyboardAccessibilityTest = await VXORTestRunner.runTest(
                'Tastaturzugänglichkeit validieren',
                testKeyboardAccessibility,
                TEST_CONFIG
            );
            results.tests.push(keyboardAccessibilityTest);
            
            // Test 4: Kontrastprüfung
            const contrastRatioTest = await VXORTestRunner.runTest(
                'Kontrastverhältnisse validieren',
                testContrastRatio,
                TEST_CONFIG
            );
            results.tests.push(contrastRatioTest);
            
            // Test 5: Screenreader-Support
            const screenReaderTest = await VXORTestRunner.runTest(
                'Screenreader-Support validieren',
                testScreenReaderSupport,
                TEST_CONFIG
            );
            results.tests.push(screenReaderTest);
            
            // Gesamtergebnis bestimmen
            results.success = results.tests.every(test => test.success);
            
            return results;
        }
        catch (error) {
            console.error('❌ Fehler in Barrierefreiheitstests:', error);
            results.success = false;
            results.error = error.message;
            return results;
        }
    }
    
    /**
     * Test 1: Validierung der ARIA-Landmarks
     */
    async function testAriaLandmarks() {
        // Prüfen, ob wichtige Landmark-Rollen vorhanden sind
        const mainContent = document.querySelector('[role="main"], main');
        const navigation = document.querySelector('[role="navigation"], nav');
        const banner = document.querySelector('[role="banner"], header');
        
        if (!mainContent) {
            throw new Error('Kein main-Element oder Element mit role="main" gefunden');
        }
        
        if (!navigation) {
            console.warn('Kein nav-Element oder Element mit role="navigation" gefunden');
        }
        
        if (!banner) {
            console.warn('Kein header-Element oder Element mit role="banner" gefunden');
        }
        
        // Prüfen, ob Regions korrekt benannt sind
        const regions = document.querySelectorAll('[role="region"], section, [role="tabpanel"]');
        
        for (const region of regions) {
            const hasLabel = region.getAttribute('aria-label') || 
                            region.getAttribute('aria-labelledby');
            
            if (!hasLabel) {
                const headings = region.querySelectorAll('h1, h2, h3, h4, h5, h6');
                
                if (headings.length === 0) {
                    console.warn(`Region ohne Label oder Überschrift gefunden: ${region.id || 'Unnamed'}`);
                }
            }
        }
        
        // Prüfen, ob es genau eine main-Landmark gibt
        const mainElements = document.querySelectorAll('[role="main"], main');
        
        if (mainElements.length > 1) {
            throw new Error(`Mehrere main-Landmarks gefunden (${mainElements.length}), nur eine ist erlaubt`);
        }
        
        return true;
    }
    
    /**
     * Test 2: Validierung der semantischen Struktur
     */
    async function testSemanticStructure() {
        // Prüfen, ob Überschriften korrekt strukturiert sind
        const h1Elements = document.querySelectorAll('h1');
        const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
        
        if (h1Elements.length === 0) {
            console.warn('Keine h1-Überschrift gefunden');
        }
        
        if (h1Elements.length > 1) {
            console.warn(`Mehrere h1-Überschriften gefunden (${h1Elements.length})`);
        }
        
        // Prüfen auf übersprungene Überschriftenebenen
        const headingLevels = Array.from(headings).map(h => parseInt(h.tagName.substring(1)));
        const uniqueLevels = [...new Set(headingLevels)].sort();
        
        for (let i = 1; i < uniqueLevels.length; i++) {
            if (uniqueLevels[i] - uniqueLevels[i-1] > 1) {
                console.warn(`Übersprungene Überschriftenebene: von h${uniqueLevels[i-1]} zu h${uniqueLevels[i]}`);
            }
        }
        
        // Prüfen, ob Listen semantisch korrekt sind
        const lists = document.querySelectorAll('ul, ol');
        
        for (const list of lists) {
            const listItems = list.querySelectorAll('li');
            
            if (listItems.length === 0) {
                console.warn(`Liste ohne Listenelemente gefunden: ${list.id || 'Unnamed'}`);
            }
            
            // Prüfen, ob direkte Kinder ausschließlich li-Elemente sind
            for (const child of list.children) {
                if (child.tagName.toLowerCase() !== 'li') {
                    throw new Error(`Liste mit ungültigem Kindknoten gefunden: ${child.tagName}`);
                }
            }
        }
        
        // Prüfen, ob Tabellen semantisch korrekt sind
        const tables = document.querySelectorAll('table');
        
        for (const table of tables) {
            // Prüfen, ob Tabelle Überschriften hat
            const headers = table.querySelectorAll('th');
            
            if (headers.length === 0) {
                console.warn(`Tabelle ohne <th>-Elemente gefunden: ${table.id || 'Unnamed'}`);
            }
            
            // Prüfen, ob Zellen mit Spalten/Zeilen verknüpft sind
            const cells = table.querySelectorAll('td');
            for (const cell of cells) {
                const headers = cell.getAttribute('headers');
                const scope = cell.getAttribute('scope');
                
                if (!headers && !scope) {
                    console.warn(`Tabellenzelle ohne headers oder scope gefunden: ${table.id || 'Unnamed'}`);
                    break; // Nur einmal pro Tabelle warnen
                }
            }
        }
        
        return true;
    }
    
    /**
     * Test 3: Validierung der Tastaturzugänglichkeit
     */
    async function testKeyboardAccessibility() {
        // Prüfe, ob alle interaktiven Elemente tabulierbar sind
        const interactiveElements = document.querySelectorAll('a, button, input, select, textarea, [role="button"], [role="link"], [role="checkbox"], [role="radio"], [role="combobox"], [role="tab"]');
        
        for (const element of interactiveElements) {
            const tabindex = element.getAttribute('tabindex');
            const isDisabled = element.disabled || element.getAttribute('aria-disabled') === 'true';
            
            // Prüfe, ob das Element tabulierbar ist oder explizit deaktiviert wurde
            if (tabindex === '-1' && !isDisabled) {
                console.warn(`Interaktives Element mit tabindex="-1" gefunden: ${element.tagName} ${element.id || element.textContent || 'Unnamed'}`);
            }
            
            // Prüfe, ob Tab-Reihenfolge nicht manipuliert wird
            if (tabindex && parseInt(tabindex) > 0) {
                console.warn(`Element mit positivem tabindex="${tabindex}" gefunden: ${element.tagName} ${element.id || element.textContent || 'Unnamed'}`);
            }
        }
        
        // Prüfe, ob alle Formulareingaben Labels haben
        const formControls = document.querySelectorAll('input, select, textarea');
        
        for (const control of formControls) {
            if (control.type === 'hidden') continue;
            
            const id = control.id;
            const label = id ? document.querySelector(`label[for="${id}"]`) : null;
            const ariaLabel = control.getAttribute('aria-label');
            const ariaLabelledBy = control.getAttribute('aria-labelledby');
            
            if (!label && !ariaLabel && !ariaLabelledBy) {
                throw new Error(`Formularelement ohne Label, aria-label oder aria-labelledby gefunden: ${control.id || control.name || 'Unnamed'}`);
            }
        }
        
        // Prüfe, ob alle Buttons Beschriftungen haben
        const buttons = document.querySelectorAll('button, [role="button"]');
        
        for (const button of buttons) {
            const hasContent = button.textContent.trim() !== '';
            const hasAriaLabel = button.getAttribute('aria-label');
            const hasAriaLabelledBy = button.getAttribute('aria-labelledby');
            
            if (!hasContent && !hasAriaLabel && !hasAriaLabelledBy) {
                throw new Error(`Button ohne Beschriftung, aria-label oder aria-labelledby gefunden: ${button.id || 'Unnamed'}`);
            }
        }
        
        // Prüfe Fokus-Sichtbarkeit für interaktive Elemente
        for (const element of interactiveElements) {
            // Dies kann nur eine Warnung sein, da wir den Fokus-Stil nicht programmatorisch prüfen können
            const style = getComputedStyle(element);
            
            if (style.outline === '0px none' || style.outline === 'none') {
                const hasFocusStyles = false; // Vereinfachte Annahme
                
                if (!hasFocusStyles) {
                    console.warn(`Element könnte keinen sichtbaren Fokus haben: ${element.tagName} ${element.id || 'Unnamed'}`);
                }
            }
        }
        
        return true;
    }
    
    /**
     * Test 4: Validierung der Kontrastverhältnisse
     */
    async function testContrastRatio() {
        // Hinweis: Eine vollständige Kontrastprüfung benötigt eine spezielle Bibliothek
        // Dies ist eine vereinfachte Version, die nur bestimmte Elemente prüft
        
        // Testfunktion für das Kontrastverhältnis (vereinfacht)
        const checkElementContrast = (element, description) => {
            const elementStyle = getComputedStyle(element);
            const foreground = elementStyle.color;
            
            // Finde das erste übergeordnete Element mit definiertem Hintergrund
            let parent = element;
            let background = 'transparent';
            
            while (parent && background === 'transparent') {
                const parentStyle = getComputedStyle(parent);
                background = parentStyle.backgroundColor;
                
                if (background === 'transparent' || background === 'rgba(0, 0, 0, 0)') {
                    parent = parent.parentElement;
                }
            }
            
            // Fallback auf Body-Hintergrund
            if (background === 'transparent' || background === 'rgba(0, 0, 0, 0)') {
                background = getComputedStyle(document.body).backgroundColor;
            }
            
            // Berechne Kontrastverhältnis (vereinfacht)
            const ratio = calculateContrastRatio(foreground, background);
            
            if (ratio < 4.5) {
                console.warn(`Potenziell niedriges Kontrastverhältnis (${ratio.toFixed(2)}) für ${description}`);
            }
            
            return ratio >= 4.5;
        };
        
        // Prüfe Text-Kontrast für wichtige Elemente
        const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
        const paragraphs = document.querySelectorAll('p');
        const labels = document.querySelectorAll('label');
        
        let allPassed = true;
        
        for (const heading of headings) {
            allPassed = checkElementContrast(heading, `Überschrift "${heading.textContent.trim()}"`) && allPassed;
        }
        
        // Überprüfe eine Stichprobe von Absätzen
        const sampleParagraphs = Array.from(paragraphs).slice(0, 5);
        for (const paragraph of sampleParagraphs) {
            allPassed = checkElementContrast(paragraph, `Absatz "${paragraph.textContent.substring(0, 20)}..."`) && allPassed;
        }
        
        for (const label of labels) {
            allPassed = checkElementContrast(label, `Label "${label.textContent.trim()}"`) && allPassed;
        }
        
        return allPassed;
    }
    
    /**
     * Test 5: Validierung des Screenreader-Supports
     */
    async function testScreenReaderSupport() {
        // Prüfe, ob Live-Regions korrekt konfiguriert sind
        const liveRegions = document.querySelectorAll('[aria-live]');
        
        if (liveRegions.length === 0) {
            console.warn('Keine Live-Regions gefunden, dynamische Updates könnten für Screenreader nicht angekündigt werden');
        } else {
            for (const region of liveRegions) {
                const politeness = region.getAttribute('aria-live');
                
                if (politeness !== 'polite' && politeness !== 'assertive' && politeness !== 'off') {
                    throw new Error(`Ungültiger aria-live-Wert "${politeness}" gefunden`);
                }
            }
        }
        
        // Prüfe, ob dynamische Inhalte aria-relevant haben
        const dynamicContent = document.querySelectorAll('[aria-live]');
        
        for (const content of dynamicContent) {
            const relevant = content.getAttribute('aria-relevant');
            
            if (!relevant) {
                console.info(`Live-Region ohne aria-relevant gefunden: ${content.id || 'Unnamed'}`);
            }
        }
        
        // Prüfe alt-Attribute für Bilder
        const images = document.querySelectorAll('img');
        
        for (const image of images) {
            const alt = image.getAttribute('alt');
            
            if (alt === null) {
                throw new Error(`Bild ohne alt-Attribut gefunden: ${image.src}`);
            }
        }
        
        // Prüfe, ob versteckte Elemente für Screenreader auch wirklich versteckt sind
        const hiddenElements = document.querySelectorAll('[aria-hidden="true"]');
        
        for (const element of hiddenElements) {
            // Prüfe, ob interaktive Elemente mit aria-hidden="true" aus dem Tab-Fokus entfernt wurden
            const focusableDescendants = element.querySelectorAll('a[href], button, input, select, textarea, [tabindex]');
            
            if (focusableDescendants.length > 0) {
                throw new Error(`Element mit aria-hidden="true" enthält fokussierbare Elemente: ${element.id || 'Unnamed'}`);
            }
        }
        
        // Prüfe, ob HTML-Sprache definiert ist
        const html = document.querySelector('html');
        const langAttribute = html.getAttribute('lang');
        
        if (!langAttribute) {
            console.warn('Kein lang-Attribut im html-Element gefunden');
        }
        
        return true;
    }
    
    /**
     * Hilfsfunktion: Berechnet vereinfacht das Kontrastverhältnis zweier Farben
     * @param {string} foreground - CSS-Farbwert für Vordergrund
     * @param {string} background - CSS-Farbwert für Hintergrund
     * @returns {number} Kontrastverhältnis
     */
    function calculateContrastRatio(foreground, background) {
        // Farben in RGB umwandeln
        const fgRgb = parseColor(foreground);
        const bgRgb = parseColor(background);
        
        if (!fgRgb || !bgRgb) return 4.5; // Fallback bei Parsing-Fehlern
        
        // Relative Luminanz berechnen
        const fgLuminance = calculateLuminance(fgRgb.r, fgRgb.g, fgRgb.b);
        const bgLuminance = calculateLuminance(bgRgb.r, bgRgb.g, bgRgb.b);
        
        // Kontrastverhältnis berechnen
        const lighter = Math.max(fgLuminance, bgLuminance);
        const darker = Math.min(fgLuminance, bgLuminance);
        
        return (lighter + 0.05) / (darker + 0.05);
    }
    
    /**
     * Hilfsfunktion: Berechnet die relative Luminanz einer Farbe
     * @param {number} r - Rot-Wert (0-255)
     * @param {number} g - Grün-Wert (0-255)
     * @param {number} b - Blau-Wert (0-255)
     * @returns {number} Relative Luminanz
     */
    function calculateLuminance(r, g, b) {
        // Werte normalisieren
        r /= 255;
        g /= 255;
        b /= 255;
        
        // Gamma-Korrektur
        r = r <= 0.03928 ? r / 12.92 : Math.pow((r + 0.055) / 1.055, 2.4);
        g = g <= 0.03928 ? g / 12.92 : Math.pow((g + 0.055) / 1.055, 2.4);
        b = b <= 0.03928 ? b / 12.92 : Math.pow((b + 0.055) / 1.055, 2.4);
        
        // Relative Luminanz berechnen
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
    
    /**
     * Hilfsfunktion: Parst CSS-Farbwerte in RGB-Objekt
     * @param {string} color - CSS-Farbwert
     * @returns {Object|null} RGB-Objekt oder null bei Parsing-Fehler
     */
    function parseColor(color) {
        if (color.startsWith('#')) {
            // Hex-Format
            let r, g, b;
            
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
            
            return { r, g, b };
        } else if (color.startsWith('rgb')) {
            // RGB-Format
            const match = color.match(/rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/i);
            
            if (match) {
                return {
                    r: parseInt(match[1]),
                    g: parseInt(match[2]),
                    b: parseInt(match[3])
                };
            }
        }
        
        return null; // Unbekanntes Format
    }
    
    // Öffentliche API
    return {
        runTests
    };
})();

// Registriere die Tests beim TestRunner
if (typeof VXORTestRunner !== 'undefined') {
    VXORTestRunner.accessibilityTests = VXORAccessibilityTests.runTests;
}
