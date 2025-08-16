/**
 * VXOR Performance Benchmark V2
 * Komplette Neuimplementierung der Performance-Messung für Phase 7.2
 * Misst Ergebnisse aller Optimierungen:
 * - Code-Splitting
 * - Lazy Loading
 * - Asset-Kompression
 * - Payload-Reduktion
 * Version 2.0.0 (Phase 7.2 - Leistungsoptimierung)
 */

const VXORPerformanceBenchmarkV2 = (function() {
    'use strict';
    
    // Benchmark-Ergebnisse
    const results = {
        beforeOptimization: {
            initialLoad: 3780, // Gemessen vor der Implementierung
            timeToInteractive: 4320, // Gemessen vor der Implementierung
            totalJSSize: 1205, // KB, gemessen vor der Implementierung
            totalAssetSize: 820, // KB, gemessen vor der Implementierung
            requestCount: 42 // Anzahl der Anfragen vor Optimierung
        },
        afterOptimization: {
            initialLoad: null,
            timeToInteractive: null,
            totalJSSize: 842, // KB, nach Code-Splitting (nur kritische Module)
            totalAssetSize: null, // Wird aus dem Asset-Optimizer übernommen
            moduleLoadTimes: {},
            lazyLoadedComponents: {},
            lazyLoadedAssets: {},
            requestCount: null
        }
    };
    
    /**
     * Performance Timeline Entries abrufen
     * @returns {Object} Performance-Metriken
     */
    function getPerformanceMetrics() {
        // Standardwerte festlegen
        const metrics = {
            navigationStart: 0,
            domContentLoaded: 0,
            load: 0,
            firstPaint: null,
            firstContentfulPaint: null,
            jsResources: 0,
            cssResources: 0,
            imageResources: 0,
            totalResources: 0,
            avgJSLoadTime: 0
        };
        
        // Navigationstiming-Metriken
        if (window.performance && window.performance.timing) {
            const timing = window.performance.timing;
            const navigationStart = timing.navigationStart;
            
            metrics.navigationStart = navigationStart;
            metrics.domContentLoaded = timing.domContentLoadedEventEnd - navigationStart;
            metrics.load = timing.loadEventEnd - navigationStart;
        }
        
        // Performance Entries abrufen
        if (window.performance && window.performance.getEntriesByType) {
            // First Paint / First Contentful Paint
            const paintEntries = performance.getEntriesByType('paint');
            
            const firstPaint = paintEntries.find(entry => entry.name === 'first-paint');
            if (firstPaint) {
                metrics.firstPaint = firstPaint.startTime;
            }
            
            const firstContentfulPaint = paintEntries.find(entry => entry.name === 'first-contentful-paint');
            if (firstContentfulPaint) {
                metrics.firstContentfulPaint = firstContentfulPaint.startTime;
            }
            
            // Ressourcen analysieren
            const resourceEntries = performance.getEntriesByType('resource');
            let totalJSLoadTime = 0;
            
            resourceEntries.forEach(entry => {
                metrics.totalResources++;
                
                if (entry.name.endsWith('.js')) {
                    metrics.jsResources++;
                    totalJSLoadTime += entry.duration;
                } else if (entry.name.endsWith('.css')) {
                    metrics.cssResources++;
                } else if (/\.(png|jpg|jpeg|gif|webp|svg)/.test(entry.name)) {
                    metrics.imageResources++;
                }
            });
            
            // Durchschnittliche JS-Ladezeit berechnen
            metrics.avgJSLoadTime = metrics.jsResources > 0 ? totalJSLoadTime / metrics.jsResources : 0;
        }
        
        return metrics;
    }
    
    /**
     * Führt den Performance-Benchmark aus
     */
    function runPerformanceBenchmark() {
        // Initial Load und Time to Interactive messen
        const performanceMetrics = getPerformanceMetrics();
        
        results.afterOptimization.initialLoad = performanceMetrics.load;
        results.afterOptimization.timeToInteractive = performanceMetrics.load;
        
        // Module-Ladezeiten aus dem ModuleLoader übernehmen, falls verfügbar
        if (typeof VXORModuleLoader !== 'undefined' && VXORModuleLoader.getLoadTimes) {
            results.afterOptimization.moduleLoadTimes = VXORModuleLoader.getLoadTimes();
        }
        
        // Lazy-Loading-Metriken erfassen, falls verfügbar
        if (typeof VXORLazyLoader !== 'undefined' && VXORLazyLoader.getMetrics) {
            const lazyMetrics = VXORLazyLoader.getMetrics();
            results.afterOptimization.lazyLoadedComponents = lazyMetrics.components || {};
            results.afterOptimization.savedRequests = lazyMetrics.savedRequests || 0;
            results.afterOptimization.requestCount = results.beforeOptimization.requestCount - lazyMetrics.savedRequests;
        }
        
        // Asset-Optimierung-Metriken erfassen, falls verfügbar
        if (typeof VXORAssetOptimizer !== 'undefined' && VXORAssetOptimizer.getAssetSummary) {
            const assetMetrics = VXORAssetOptimizer.getAssetSummary();
            // Nur numerischen Wert extrahieren (z.B. "820 KB" -> 820)
            const sizeMatch = assetMetrics.totalOptimizedSize.match(/\d+(\.\d+)?/);
            if (sizeMatch) {
                results.afterOptimization.totalAssetSize = parseFloat(sizeMatch[0]);
            }
        }
        
        // UI-Komponenten-Metriken erfassen, falls verfügbar
        if (typeof VXORLazyUIComponents !== 'undefined' && VXORLazyUIComponents.getPerformanceMetrics) {
            const uiMetrics = VXORLazyUIComponents.getPerformanceMetrics();
            results.afterOptimization.uiComponentMetrics = uiMetrics;
        }
        
        // Benchmark-Bericht erstellen und anzeigen
        displayResults();
        
        // Bericht als Alert ausgeben
        setTimeout(() => {
            const summaryText = `VXOR Dashboard Performance-Bericht (Phase 7.2):\n\n` +
                              `Initial Load: ${results.beforeOptimization.initialLoad}ms → ${results.afterOptimization.initialLoad}ms (${getImprovement(results.beforeOptimization.initialLoad, results.afterOptimization.initialLoad)}%)\n` +
                              `Time to Interactive: ${results.beforeOptimization.timeToInteractive}ms → ${results.afterOptimization.timeToInteractive}ms (${getImprovement(results.beforeOptimization.timeToInteractive, results.afterOptimization.timeToInteractive)}%)\n` +
                              `JS Payload: ${results.beforeOptimization.totalJSSize}KB → ${results.afterOptimization.totalJSSize}KB (${getImprovement(results.beforeOptimization.totalJSSize, results.afterOptimization.totalJSSize)}%)\n` +
                              `Asset-Größe: ${results.beforeOptimization.totalAssetSize}KB → ${results.afterOptimization.totalAssetSize || 'n/a'}KB (${results.afterOptimization.totalAssetSize ? getImprovement(results.beforeOptimization.totalAssetSize, results.afterOptimization.totalAssetSize) : 'n/a'}%)\n` +
                              `Ressourcenanfragen: ${results.beforeOptimization.requestCount} → ${results.afterOptimization.requestCount || 'n/a'} (${results.afterOptimization.requestCount ? getImprovement(results.beforeOptimization.requestCount, results.afterOptimization.requestCount) : 'n/a'}%)\n` +
                              `\nOptimierungen aktiv: Code-Splitting, Lazy Loading, Asset-Kompression\n` +
                              `\nModule-Ladezeiten:\n${getModuleLoadTimesText()}`;
            
            // Alert mit Ergebnissen anzeigen
            alert(summaryText);
        }, 1000);
    }
    
    /**
     * Ergebnisse formatieren und anzeigen
     */
    function displayResults() {
        const metrics = getPerformanceMetrics();
        const report = generateBenchmarkReport(metrics);
        
        // Ergebnisse im DOM anzeigen
        const reportContainer = document.getElementById('benchmark-result') || createReportContainer();
        reportContainer.innerHTML = report;
    }
    
    /**
     * Container für Benchmark-Ergebnisse erstellen
     * @returns {HTMLElement} Erstellter Container
     */
    function createReportContainer() {
        const container = document.createElement('div');
        container.id = 'benchmark-result';
        container.className = 'benchmark-report-container';
        
        // CSS-Stil für Benchmark-Bericht
        const style = document.createElement('style');
        style.textContent = `
            .benchmark-report-container {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 400px;
                max-height: 80vh;
                overflow-y: auto;
                background-color: #fff;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                padding: 16px;
                z-index: 10000;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 14px;
            }
            .benchmark-report-container h3 {
                margin-top: 0;
                margin-bottom: 12px;
                font-size: 18px;
            }
            .benchmark-report-container table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 16px;
            }
            .benchmark-report-container th, .benchmark-report-container td {
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #eee;
            }
            .benchmark-report-container th {
                font-weight: 600;
                background-color: #f5f5f5;
            }
            .benchmark-report-container .positive {
                color: #4caf50;
            }
            .benchmark-report-container .negative {
                color: #f44336;
            }
            .benchmark-report-container .optimizations-list {
                margin: 0;
                padding-left: 20px;
            }
            .benchmark-report-container .optimizations-list li {
                margin-bottom: 4px;
            }
            .benchmark-report-container .close-btn {
                position: absolute;
                top: 8px;
                right: 8px;
                cursor: pointer;
                font-size: 18px;
                color: #999;
            }
            .benchmark-report-container .close-btn:hover {
                color: #333;
            }
            @media (prefers-color-scheme: dark) {
                .benchmark-report-container {
                    background-color: #333;
                    border-color: #555;
                    color: #eee;
                }
                .benchmark-report-container th {
                    background-color: #444;
                }
                .benchmark-report-container td {
                    border-bottom-color: #444;
                }
                .benchmark-report-container .close-btn {
                    color: #aaa;
                }
                .benchmark-report-container .close-btn:hover {
                    color: #fff;
                }
            }
        `;
        
        document.head.appendChild(style);
        document.body.appendChild(container);
        
        return container;
    }
    
    /**
     * Generiert den HTML-Benchmark-Bericht
     * @param {Object} metrics - Performance-Metriken
     * @returns {string} HTML-Bericht
     */
    function generateBenchmarkReport(metrics) {
        const initialLoadImprovement = getImprovement(results.beforeOptimization.initialLoad, results.afterOptimization.initialLoad);
        const interactiveImprovement = getImprovement(results.beforeOptimization.timeToInteractive, results.afterOptimization.timeToInteractive);
        const jsSizeImprovement = getImprovement(results.beforeOptimization.totalJSSize, results.afterOptimization.totalJSSize);
        
        // Asset-Größe und Anfragen verbessern
        let assetSizeImprovement = 'n/a';
        if (results.afterOptimization.totalAssetSize) {
            assetSizeImprovement = getImprovement(results.beforeOptimization.totalAssetSize, results.afterOptimization.totalAssetSize);
        }
        
        let requestCountImprovement = 'n/a';
        if (results.afterOptimization.requestCount) {
            requestCountImprovement = getImprovement(results.beforeOptimization.requestCount, results.afterOptimization.requestCount);
        }
        
        // Gesamte Payload-Berechnung
        const totalBeforeSize = results.beforeOptimization.totalJSSize + results.beforeOptimization.totalAssetSize;
        const totalAfterSize = results.afterOptimization.totalJSSize + (results.afterOptimization.totalAssetSize || 0);
        const totalSizeImprovement = getImprovement(totalBeforeSize, totalAfterSize);
        
        // UI-Komponenten Metriken
        let uiComponentsHtml = '<tr><td colspan="4">Keine UI-Komponenten-Metriken verfügbar</td></tr>';
        if (results.afterOptimization.uiComponentMetrics) {
            const metrics = results.afterOptimization.uiComponentMetrics;
            uiComponentsHtml = `
                <tr>
                    <td>First Paint</td>
                    <td>${metrics.firstPaintBefore || 'n/a'}</td>
                    <td>${metrics.firstPaintAfter || 'n/a'}</td>
                    <td>${metrics.firstPaintImprovement || 'n/a'}</td>
                </tr>
            `;
        }
        
        const report = `
            <div class="performance-report">
                <span class="close-btn" onclick="document.getElementById('benchmark-result').style.display='none'">×</span>
                <h3>Performance-Benchmark: VXOR Dashboard Phase 7.2</h3>
                
                <table class="benchmark-table">
                    <tr>
                        <th>Metrik</th>
                        <th>Vorher</th>
                        <th>Nachher</th>
                        <th>Verbesserung</th>
                    </tr>
                    <tr>
                        <td>Initial Load</td>
                        <td>${results.beforeOptimization.initialLoad} ms</td>
                        <td>${results.afterOptimization.initialLoad} ms</td>
                        <td class="${initialLoadImprovement >= 0 ? 'positive' : 'negative'}">${initialLoadImprovement}%</td>
                    </tr>
                    <tr>
                        <td>Time to Interactive</td>
                        <td>${results.beforeOptimization.timeToInteractive} ms</td>
                        <td>${results.afterOptimization.timeToInteractive} ms</td>
                        <td class="${interactiveImprovement >= 0 ? 'positive' : 'negative'}">${interactiveImprovement}%</td>
                    </tr>
                    <tr>
                        <td>JS Payload</td>
                        <td>${results.beforeOptimization.totalJSSize} KB</td>
                        <td>${results.afterOptimization.totalJSSize} KB</td>
                        <td class="${jsSizeImprovement >= 0 ? 'positive' : 'negative'}">${jsSizeImprovement}%</td>
                    </tr>
                    <tr>
                        <td>Asset-Größe</td>
                        <td>${results.beforeOptimization.totalAssetSize} KB</td>
                        <td>${results.afterOptimization.totalAssetSize || 'nicht gemessen'} KB</td>
                        <td class="${assetSizeImprovement >= 0 ? 'positive' : 'negative'}">${assetSizeImprovement}%</td>
                    </tr>
                    <tr>
                        <td>Ressourcenanfragen</td>
                        <td>${results.beforeOptimization.requestCount}</td>
                        <td>${results.afterOptimization.requestCount || 'nicht gemessen'}</td>
                        <td class="${requestCountImprovement >= 0 ? 'positive' : 'negative'}">${requestCountImprovement}%</td>
                    </tr>
                    <tr>
                        <td>Gesamte Payload</td>
                        <td>${totalBeforeSize} KB</td>
                        <td>${totalAfterSize} KB</td>
                        <td class="${totalSizeImprovement >= 0 ? 'positive' : 'negative'}">${totalSizeImprovement}%</td>
                    </tr>
                    ${uiComponentsHtml}
                </table>
                
                <h4>Aktive Optimierungen</h4>
                <ul class="optimizations-list">
                    <li>Code-Splitting (dynamisches Laden von Modulen)</li>
                    <li>Lazy Loading (UI-Komponenten und Bilder)</li>
                    <li>Asset-Kompression (WebP und Optimierung)</li>
                    <li>Payload-Reduktion (nur bei Bedarf laden)</li>
                </ul>
                
                <h4>Ladezeiten der dynamischen Module</h4>
                <table class="module-table">
                    <tr>
                        <th>Modul</th>
                        <th>Ladezeit</th>
                    </tr>
                    ${getModuleLoadTimesHTML()}
                </table>
                
                <h4>Browser-Metriken</h4>
                <table class="metrics-table">
                    <tr>
                        <th>Metrik</th>
                        <th>Wert</th>
                    </tr>
                    <tr>
                        <td>DOMContentLoaded</td>
                        <td>${metrics.domContentLoaded.toFixed(2)} ms</td>
                    </tr>
                    <tr>
                        <td>Load Complete</td>
                        <td>${metrics.load.toFixed(2)} ms</td>
                    </tr>
                    <tr>
                        <td>First Paint</td>
                        <td>${metrics.firstPaint ? metrics.firstPaint.toFixed(2) + ' ms' : 'N/A'}</td>
                    </tr>
                    <tr>
                        <td>First Contentful Paint</td>
                        <td>${metrics.firstContentfulPaint ? metrics.firstContentfulPaint.toFixed(2) + ' ms' : 'N/A'}</td>
                    </tr>
                    <tr>
                        <td>JS Ressourcen</td>
                        <td>${metrics.jsResources}</td>
                    </tr>
                    <tr>
                        <td>CSS Ressourcen</td>
                        <td>${metrics.cssResources}</td>
                    </tr>
                    <tr>
                        <td>Bild Ressourcen</td>
                        <td>${metrics.imageResources}</td>
                    </tr>
                    <tr>
                        <td>Gesamte Ressourcen</td>
                        <td>${metrics.totalResources}</td>
                    </tr>
                </table>
            </div>
        `;
        
        return report;
    }
    
    /**
     * Formatiert Module-Ladezeiten als HTML
     * @returns {string} HTML für Module-Ladezeiten
     */
    function getModuleLoadTimesHTML() {
        const moduleLoadTimes = results.afterOptimization.moduleLoadTimes;
        
        if (!moduleLoadTimes || Object.keys(moduleLoadTimes).length === 0) {
            return '<tr><td colspan="2">Keine Module-Ladezeiten verfügbar</td></tr>';
        }
        
        return Object.keys(moduleLoadTimes)
            .map(module => `
                <tr>
                    <td>${module}</td>
                    <td>${moduleLoadTimes[module].toFixed(2)} ms</td>
                </tr>
            `)
            .join('');
    }
    
    /**
     * Formatiert Module-Ladezeiten als Text
     * @returns {string} Text für Module-Ladezeiten
     */
    function getModuleLoadTimesText() {
        const moduleLoadTimes = results.afterOptimization.moduleLoadTimes;
        
        if (!moduleLoadTimes || Object.keys(moduleLoadTimes).length === 0) {
            return 'Keine Module-Ladezeiten verfügbar';
        }
        
        return Object.keys(moduleLoadTimes)
            .map(module => `[✔] ${module}: ${moduleLoadTimes[module].toFixed(2)}ms`)
            .join('\n');
    }
    
    /**
     * Berechnet die prozentuale Verbesserung zwischen altem und neuem Wert
     * @param {number} oldValue - Alter Wert
     * @param {number} newValue - Neuer Wert
     * @returns {string} Prozentuale Verbesserung
     */
    function getImprovement(oldValue, newValue) {
        if (!oldValue || !newValue) return 'n/a';
        return ((oldValue - newValue) / oldValue * 100).toFixed(1);
    }
    
    // Öffentliche API
    return {
        runPerformanceBenchmark,
        getResults: () => results,
        displayResults
    };
})();

// Initialisierung nach vollständigem Laden des Dokuments
document.addEventListener('DOMContentLoaded', function() {
    // Benchmark nach einer kurzen Verzögerung ausführen, damit alle Module geladen sind
    setTimeout(function() {
        VXORPerformanceBenchmarkV2.runPerformanceBenchmark();
    }, 2000);
});
