/**
 * VXORModuleLoader.js
 * Dynamischer Modul-Loader für VXOR Benchmark Dashboard
 * 
 * Implementiert Code-Splitting für verbesserte Performance durch:
 * - Asynchrones Laden von Modulen nur bei Bedarf
 * - Paralleles Laden kritischer Module
 * - Priorisierung von Kernfunktionalität
 * - Deferred Loading nicht-kritischer Module
 */

const VXORModuleLoader = (function() {
    'use strict';
    
    // Module und ihre Ladestatus
    const modules = {
        utils: { path: 'js/VXORUtils.js', loaded: false, instance: null, critical: true },
        matrix: { path: 'js/VXORMatrix.js', loaded: false, instance: null, critical: false },
        quantum: { path: 'js/VXORQuantum.js', loaded: false, instance: null, critical: false },
        tmath: { path: 'js/VXORTMath.js', loaded: false, instance: null, critical: false },
        mlperf: { path: 'js/VXORMLPerf.js', loaded: false, instance: null, critical: false },
        swebench: { path: 'js/VXORSWEBench.js', loaded: false, instance: null, critical: false },
        security: { path: 'js/VXORSecurity.js', loaded: false, instance: null, critical: false }
    };
    
    // Performance-Messungen speichern
    const performance = {
        startTime: null,
        moduleLoadTimes: {},
        getTotalLoadTime: function() {
            return window.performance.now() - this.startTime;
        }
    };
    
    // Modul-Integritäts-Hashes (in der Produktion würden diese aus einer Konfigurationsdatei kommen)
    const moduleIntegrity = {
        'utils': 'sha384-abcdef1234567890', // Beispielwerte
        'matrix': 'sha384-fedcba0987654321',
        'quantum': 'sha384-123456abcdef7890',
        'tmath': 'sha384-098765fedcba4321',
        'mlperf': 'sha384-abcdef0987654321',
        'swebench': 'sha384-fedcba1234567890',
        'security': 'sha384-123456fedcba7890'
    };
    
    /**
     * Prüft, ob eine URL gegen die Whitelist validiert
     * @param {string} url - Zu prüfende URL
     * @returns {boolean} Ist die URL erlaubt
     * @private
     */
    function isUrlWhitelisted(url) {
        // Wenn VXORAuth vorhanden ist, dessen Whitelist-Prüfung verwenden
        if (window.VXORAuth && typeof VXORAuth.isUrlWhitelisted === 'function') {
            return VXORAuth.isUrlWhitelisted(url);
        }
        
        // Fallback: Einfache Domänenprüfung
        try {
            const urlObj = new URL(url, window.location.origin);
            // Nur lokale Skripte oder vertrauenswürdige CDNs zulassen
            return urlObj.origin === window.location.origin ||
                   urlObj.hostname === 'cdn.jsdelivr.net' ||
                   urlObj.hostname === 'trusted.cdn.com';
        } catch (error) {
            console.error('Ungültige URL:', url);
            return false;
        }
    }
    
    /**
     * Dynamisch ein JavaScript-Modul laden mit Sicherheitsmaßnahmen
     * @param {string} moduleId - ID des zu ladenden Moduls
     * @returns {Promise<Object>} Das geladene Modul
     */
    async function loadModule(moduleId) {
        if (!modules[moduleId]) {
            throw new Error(`Unbekanntes Modul: ${moduleId}`);
        }
        
        // Wenn bereits geladen, direkt zurückgeben
        if (modules[moduleId].loaded) {
            return modules[moduleId].instance;
        }
        
        const startTime = window.performance.now();
        const modulePath = modules[moduleId].path;
        
        // SICHERHEIT: URL-Validierung gegen Whitelist
        if (!isUrlWhitelisted(modulePath)) {
            throw new Error(`Sicherheitswarnung: Skript-URL nicht auf der Whitelist: ${modulePath}`);
        }
        
        try {
            // Dynamischer Import des Moduls
            const script = document.createElement('script');
            script.src = modulePath;
            script.async = true;
            
            // SICHERHEIT: crossOrigin für externe Skripte setzen
            if (!modulePath.startsWith(window.location.origin)) {
                script.crossOrigin = 'anonymous';
            }
            
            // SICHERHEIT: Integritätsprüfung hinzufügen, falls vorhanden
            if (moduleIntegrity[moduleId]) {
                script.integrity = moduleIntegrity[moduleId];
            }
            
            // Promise für das Laden des Scripts
            const moduleLoaded = new Promise((resolve, reject) => {
                script.onload = resolve;
                script.onerror = (event) => {
                    // Detaillierte Fehlermeldung bei Integritätsprüfungen
                    if (script.integrity && event.type === 'error') {
                        reject(new Error(`Integritätsprüfung fehlgeschlagen für ${moduleId}: Hash stimmt nicht überein`));
                    } else {
                        reject(new Error(`Fehler beim Laden von ${moduleId}`));
                    }
                };
            });
            
            // Script zum DOM hinzufügen und warten bis es geladen ist
            document.head.appendChild(script);
            await moduleLoaded;
            
            // Referenz auf globales Modul-Objekt erhalten
            const moduleGlobal = window[`VXOR${moduleId.charAt(0).toUpperCase() + moduleId.slice(1)}`];
            if (!moduleGlobal) {
                throw new Error(`Modul ${moduleId} nicht gefunden nach dem Laden`);
            }
            
            // SICHERHEIT: Grundlegende Prüfung der Modulschnittstelle
            if (typeof moduleGlobal !== 'object' || moduleGlobal === null) {
                throw new Error(`Modul ${moduleId} hat ungültige Schnittstelle (kein Objekt)`);
            }
            
            modules[moduleId].instance = moduleGlobal;
            modules[moduleId].loaded = true;
            
            // Ladezeit speichern
            const loadTime = window.performance.now() - startTime;
            performance.moduleLoadTimes[moduleId] = loadTime;
            console.log(`Modul ${moduleId} in ${loadTime.toFixed(2)}ms geladen`);
            
            return moduleGlobal;
        } catch (error) {
            console.error(`Fehler beim Laden von Modul ${moduleId}:`, error);
            throw error;
        }
    }
    
    /**
     * Lädt kritische Module parallel
     * @returns {Promise<Object>} Objekt mit allen kritischen Modulen
     */
    async function loadCriticalModules() {
        performance.startTime = window.performance.now();
        
        const criticalModuleIds = Object.keys(modules).filter(id => modules[id].critical);
        console.log(`Lade kritische Module: ${criticalModuleIds.join(', ')}`);
        
        try {
            const loadedModules = await Promise.all(
                criticalModuleIds.map(moduleId => loadModule(moduleId))
            );
            
            // Kritische Module als Objekt zurückgeben
            const result = {};
            criticalModuleIds.forEach((id, index) => {
                result[id] = loadedModules[index];
            });
            
            return result;
        } catch (error) {
            console.error('Fehler beim Laden kritischer Module:', error);
            throw error;
        }
    }
    
    /**
     * Lädt Module nach Bedarf
     * @param {string} moduleId - ID des zu ladenden Moduls
     * @returns {Promise<Object>} Das geladene Modul
     */
    async function loadModuleOnDemand(moduleId) {
        console.log(`Bedarfsgesteuertes Laden von Modul: ${moduleId}`);
        return loadModule(moduleId);
    }
    
    /**
     * Lädt die Module für eine Benchmark-Kategorie
     * @param {string} category - Benchmark-Kategorie
     * @returns {Promise<Object>} Das entsprechende Modul
     */
    async function loadBenchmarkCategory(category) {
        // Mapping von Kategorien zu Modul-IDs
        const categoryModuleMap = {
            'matrix': 'matrix',
            'quantum': 'quantum',
            'tmath': 'tmath',
            'mlperf': 'mlperf',
            'swebench': 'swebench',
            'security': 'security'
        };
        
        const moduleId = categoryModuleMap[category];
        if (!moduleId) {
            throw new Error(`Unbekannte Kategorie: ${category}`);
        }
        
        return loadModuleOnDemand(moduleId);
    }
    
    /**
     * Gibt Leistungsstatistiken zurück
     * @returns {Object} Performance-Daten
     */
    function getPerformanceStats() {
        return {
            totalLoadTime: performance.getTotalLoadTime(),
            moduleLoadTimes: { ...performance.moduleLoadTimes }
        };
    }
    
    /**
     * Prüft ob ein Modul bereits geladen wurde
     * @param {string} moduleId - Modul-ID
     * @returns {boolean} Ist das Modul geladen
     */
    function isModuleLoaded(moduleId) {
        return modules[moduleId] && modules[moduleId].loaded;
    }
    
    // Öffentliche API
    return {
        loadCriticalModules,
        loadModuleOnDemand,
        loadBenchmarkCategory,
        getPerformanceStats,
        isModuleLoaded
    };
})();
