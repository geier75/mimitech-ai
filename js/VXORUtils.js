/**
 * VXORUtils.js
 * Kernmodul für das VXOR Benchmark Dashboard
 * 
 * Stellt zentrale Funktionen und Dienste für alle Dashboard-Module bereit:
 * - API-Kommunikation
 * - Fehlerbehandlung
 * - Event-System für Modul-Kommunikation
 * - Theme-Management
 * - Modulregistrierung
 */

const VXORUtils = (function() {
    'use strict';
    
    // Konstanten
    const API_BASE_URL = window.location.origin;
    const API_TIMEOUT = 10000; // 10 Sekunden Timeout für API-Anfragen
    
    // Interner Zustand
    const state = {
        initializing: true,
        selectedComponent: null,
        activeCategory: 'all',
        isDarkMode: window.matchMedia('(prefers-color-scheme: dark)').matches,
        registeredModules: new Map(),
        themeListeners: new Set(),
        componentChangeListeners: new Set(),
        categoryChangeListeners: new Set(),
        dataUpdateListeners: new Map()
    };
    
    // Cache für Benchmark-Daten
    const benchmarkCache = {
        all: null,
        matrix: null,
        quantum: null,
        tmaths: null,
        mlperf: null,
        swe: null,
        security: null
    };
    
    /**
     * API-Hilfsfunktionen
     */
    
    /**
     * Sendet einen Fehler an das Backend-Logging
     * @param {string} message - Fehlermeldung
     * @param {string} category - Fehlerkategorie
     * @param {Error|null} error - Optionales Error-Objekt
     * @returns {Promise<void>}
     */
    async function logError(message, category, error = null) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT);
            
            await fetch(`${API_BASE_URL}/api/log_error`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message,
                    category,
                    url: window.location.href,
                    timestamp: new Date().toISOString(),
                    error: error ? error.toString() : null,
                    userAgent: navigator.userAgent
                }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
        } catch (e) {
            console.error('Fehler beim Senden des Fehlerberichts:', e);
        }
    }
    
    /**
     * Behandelt API-Fehler und zeigt Fallback-UI mit verbesserter Barrierefreiheit
     * @param {Error} error - Der aufgetretene Fehler
     * @param {string} category - Betroffene Kategorie
     * @param {string|null} containerId - Optional: ID des Container-Elements für Fehlermeldung
     */
    function handleApiError(error, category, containerId = null) {
        console.error(`Fehler in ${category}-Benchmarks:`, error);
        logError(`API-Fehler beim Laden von ${category}-Daten`, category, error);
        
        // Eindeutige ID für den Error-Container erstellen
        const errorId = `error-${category}-${Date.now()}`;
        
        // Fallback-UI anzeigen
        if (containerId) {
            const container = document.getElementById(containerId);
            if (container) {
                // Bestehenden Inhalt entfernen
                container.innerHTML = '';
                
                // Fehlercontainer erstellen
                const errorContainer = document.createElement('div');
                errorContainer.className = 'error-message';
                errorContainer.id = errorId;
                errorContainer.setAttribute('role', 'alert');
                errorContainer.setAttribute('aria-live', 'assertive');
                
                // Struktur für bessere Barrierefreiheit
                const header = document.createElement('h3');
                header.textContent = 'Daten konnten nicht geladen werden';
                header.id = `${errorId}-heading`;
                errorContainer.appendChild(header);
                
                // Fehlerinhalt
                const message = document.createElement('p');
                message.textContent = `Beim Abrufen der ${category}-Benchmark-Daten ist ein Fehler aufgetreten.`;
                errorContainer.appendChild(message);
                
                // Technische Details für Entwickler (mit Toggle)
                if (error.message) {
                    const detailsContainer = document.createElement('details');
                    detailsContainer.className = 'error-details';
                    
                    const summary = document.createElement('summary');
                    summary.textContent = 'Technische Details';
                    detailsContainer.appendChild(summary);
                    
                    const techDetails = document.createElement('pre');
                    techDetails.className = 'error-tech-details';
                    techDetails.textContent = error.message;
                    detailsContainer.appendChild(techDetails);
                    
                    errorContainer.appendChild(detailsContainer);
                }
                
                // Retry-Button mit besserer Barrierefreiheit
                const buttonContainer = document.createElement('div');
                buttonContainer.className = 'error-actions';
                
                const retryButton = document.createElement('button');
                retryButton.className = 'retry-button';
                retryButton.textContent = 'Erneut versuchen';
                retryButton.setAttribute('aria-label', `${category}-Daten neu laden`);
                retryButton.setAttribute('aria-describedby', `${errorId}-heading`);
                retryButton.addEventListener('click', (e) => {
                    e.preventDefault();
                    retryFetch(category);
                    
                    // Feedback für Screenreader
                    const statusUpdate = document.createElement('div');
                    statusUpdate.setAttribute('aria-live', 'polite');
                    statusUpdate.className = 'sr-only';
                    statusUpdate.textContent = `Lade ${category}-Daten neu...`;
                    document.body.appendChild(statusUpdate);
                    
                    // Nach 5 Sekunden entfernen
                    setTimeout(() => {
                        if (document.body.contains(statusUpdate)) {
                            document.body.removeChild(statusUpdate);
                        }
                    }, 5000);
                });
                
                buttonContainer.appendChild(retryButton);
                errorContainer.appendChild(buttonContainer);
                
                container.appendChild(errorContainer);
            }
        }
        
        // Event auslösen, um Module über den Fehler zu informieren
        const event = new CustomEvent('vxor:error', {
            detail: { id: errorId, category, message: error.message, error }
        });
        document.dispatchEvent(event);
    }
    
    /**
     * Lädt Komponenten-Statusdaten vom Backend
     * @returns {Promise<Object|null>} Komponenten-Daten oder null bei Fehler
     */
    async function fetchComponentStatus() {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT);
            
            const response = await fetch(`${API_BASE_URL}/api/component_status`, {
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Fehler beim Laden der Komponenten:', error);
            handleApiError(error, 'components', 'component-container');
            return null;
        }
    }
    
    /**
     * Lädt Hardware-Metrikdaten vom Backend
     * @returns {Promise<Object|null>} Hardware-Metriken oder null bei Fehler
     */
    async function fetchHardwareMetrics() {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT);
            
            const response = await fetch(`${API_BASE_URL}/api/hardware_metrics`, {
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Fehler beim Laden der Hardware-Metriken:', error);
            handleApiError(error, 'hardware', 'system-stats');
            return null;
        }
    }
    
    /**
     * Initiiert einen erneuten Abrufversuch für eine Kategorie
     * @param {string} category - Benchmark-Kategorie oder 'components', 'hardware'
     * @returns {Promise<void>}
     */
    async function retryFetch(category) {
        if (category === 'components') {
            const data = await fetchComponentStatus();
            if (data) {
                // Event auslösen
                const event = new CustomEvent('vxor:componentsUpdated', { detail: { data } });
                document.dispatchEvent(event);
            }
        } else if (category === 'hardware') {
            const data = await fetchHardwareMetrics();
            if (data) {
                // Event auslösen
                const event = new CustomEvent('vxor:hardwareUpdated', { detail: { data } });
                document.dispatchEvent(event);
            }
        } else {
            await fetchBenchmarkData(state.selectedComponent, category);
        }
    }
    
    /**
     * Lädt Benchmark-Daten für eine bestimmte Kategorie
     * @param {string|null} component - Optional: Komponenten-ID für gefilterte Daten
     * @param {string} category - Benchmark-Kategorie ('all', 'matrix', 'quantum', etc.)
     * @returns {Promise<Object|null>} Benchmark-Daten oder null bei Fehler
     */
    async function fetchBenchmarkData(component = null, category = 'all') {
        // Bei vorhandenen Cache-Daten, diese verwenden
        if (benchmarkCache[category] && !component) {
            console.log(`Verwende gecachte Daten für ${category}-Benchmarks`);
            
            // Event auslösen
            notifyDataUpdateListeners(category, benchmarkCache[category]);
            
            return benchmarkCache[category];
        }
        
        // API-Endpunkt entsprechend der Kategorie wählen
        let url;
        switch (category) {
            case 'matrix':
                url = `${API_BASE_URL}/api/benchmark/matrix`;
                break;
            case 'quantum':
                url = `${API_BASE_URL}/api/benchmark/quantum`;
                break;
            case 'tmaths':
                url = `${API_BASE_URL}/api/benchmark/tmath`;
                break;
            case 'mlperf':
                url = `${API_BASE_URL}/api/benchmark/mlperf`;
                break;
            case 'swe':
                url = `${API_BASE_URL}/api/benchmark/swe`;
                break;
            case 'security':
                url = `${API_BASE_URL}/api/benchmark/security`;
                break;
            case 'all':
            default:
                url = `${API_BASE_URL}/api/benchmark_data`;
                break;
        }
        
        // Komponenten-Parameter hinzufügen, falls vorhanden
        if (component) {
            url += url.includes('?') ? '&' : '?';
            url += `component=${encodeURIComponent(component)}`;
        }
        
        try {
            console.log(`Lade Benchmark-Daten von ${url}`);
            
            // SICHERHEIT: Prüfen, ob VXORAuth vorhanden ist
            if (!window.VXORAuth) {
                throw new Error('Sicherheitsmodul nicht verfügbar');
            }
            
            // SICHERHEIT: Prüfen, ob die URL auf der Whitelist steht
            if (!VXORAuth.isUrlWhitelisted(url)) {
                throw new Error('Unerlaubte API-URL: ' + url);
            }
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT);
            
            // SICHERHEIT: Auth- und CSRF-Token hinzufügen
            const headers = new Headers({
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${VXORAuth.getToken() || 'guest'}`,
                'X-CSRF-Token': VXORAuth.getCSRFToken() || ''
            });
            
            const response = await fetch(url, {
                method: 'GET',
                headers: headers,
                signal: controller.signal,
                credentials: 'same-origin' // Cookies für Cross-Origin-Anfragen einschließen
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                // 401/403 speziell behandeln
                if (response.status === 401 || response.status === 403) {
                    throw new Error('Authentifizierungsfehler: Keine Berechtigung für diese Daten.');
                }
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // SICHERHEIT: Validierung der API-Antwort gegen Schema
            if (!VXORAuth.validateApiResponse(data, 'benchmarkData')) {
                throw new Error(`Ungültiges Datenformat für ${category}-Benchmark-Daten`);
            }
            
            // Speichern im Cache
            benchmarkCache[category] = data;
            
            // Event auslösen
            notifyDataUpdateListeners(category, data);
            
            return data;
        } catch (error) {
            if (error.name === 'AbortError') {
                console.error(`Timeout beim Laden der ${category}-Benchmark-Daten`);
                handleApiError(new Error('Die Anfrage hat zu lange gedauert. Bitte später erneut versuchen.'), category, 'charts-container');
            } else {
                console.error(`Fehler beim Laden der ${category}-Benchmark-Daten:`, error);
                handleApiError(error, category, 'charts-container');
            }
            
            return null;
        }
    }
    
    /**
     * Cache-Verwaltung
     */
    
    /**
     * Löscht den Cache für eine bestimmte Kategorie oder alle Kategorien
     * @param {string|null} category - Zu löschende Kategorie oder null für alle
     */
    function clearCache(category = null) {
        if (category === null) {
            // Gesamten Cache leeren
            Object.keys(benchmarkCache).forEach(key => {
                benchmarkCache[key] = null;
            });
            console.log('Gesamter Benchmark-Cache geleert');
        } else if (benchmarkCache.hasOwnProperty(category)) {
            // Nur die angegebene Kategorie leeren
            benchmarkCache[category] = null;
            console.log(`Cache für ${category}-Benchmarks geleert`);
        }
    }
    
    /**
     * Event-System
     */
    
    // Definierte Event-Typen
    const EventTypes = {
        DATA_UPDATED: 'vxor:dataUpdated',
        COMPONENT_CHANGED: 'vxor:componentChanged',
        CATEGORY_CHANGED: 'vxor:categoryChanged',
        THEME_CHANGED: 'vxor:themeChanged',
        ERROR: 'vxor:error',
        COMPONENTS_UPDATED: 'vxor:componentsUpdated',
        HARDWARE_UPDATED: 'vxor:hardwareUpdated',
        MODULE_REGISTERED: 'vxor:moduleRegistered',
        DASHBOARD_INITIALIZED: 'vxor:dashboardInitialized',
        USER_INTERACTION: 'vxor:userInteraction'
    };
    
    /**
     * Hilfsklasse für Event-Handling
     */
    class EventBus {
        constructor() {
            this.listeners = new Map();
        }
        
        /**
         * Registriert einen Event-Listener
         * @param {string} eventType - Typ des Events
         * @param {Function} callback - Callback-Funktion
         * @param {Object} options - Optionen für addEventListener
         */
        on(eventType, callback, options = {}) {
            if (!this.listeners.has(eventType)) {
                this.listeners.set(eventType, new Set());
            }
            this.listeners.get(eventType).add({ callback, options });
            
            document.addEventListener(eventType, callback, options);
            return this;
        }
        
        /**
         * Entfernt einen Event-Listener
         * @param {string} eventType - Typ des Events
         * @param {Function} callback - Callback-Funktion
         */
        off(eventType, callback) {
            if (this.listeners.has(eventType)) {
                const listeners = this.listeners.get(eventType);
                for (const listener of listeners) {
                    if (listener.callback === callback) {
                        document.removeEventListener(eventType, callback, listener.options);
                        listeners.delete(listener);
                        break;
                    }
                }
                
                if (listeners.size === 0) {
                    this.listeners.delete(eventType);
                }
            }
            return this;
        }
        
        /**
         * Löst ein Event aus
         * @param {string} eventType - Typ des Events
         * @param {Object} detail - Event-Details
         */
        emit(eventType, detail = {}) {
            // SICHERHEIT: Typprüfung und Validierung des detail-Objekts
            if (typeof eventType !== 'string' || !eventType) {
                console.error('Event-Typ muss ein nicht-leerer String sein');
                return this;
            }
            
            // Sicherstellen, dass detail ein Objekt ist
            if (detail === null || typeof detail !== 'object' || Array.isArray(detail)) {
                console.error('Event-Detail muss ein Objekt sein, kein null, Array oder primitiver Wert');
                return this;
            }
            
            // SICHERHEIT: Event-Detail durch VXORAuth validieren, wenn verfügbar
            if (window.VXORAuth && typeof VXORAuth.validateEventDetail === 'function') {
                if (!VXORAuth.validateEventDetail(detail)) {
                    console.error('Event-Detail enthält potenziell gefährliche Werte und wurde blockiert', {
                        eventType,
                        blockedDetail: JSON.stringify(detail).substring(0, 100) + '...'
                    });
                    return this;
                }
            }
            
            // Deep-Copy des detail-Objekts um Manipulation zu verhindern
            const safeCopy = JSON.parse(JSON.stringify(detail));
            
            const event = new CustomEvent(eventType, { detail: safeCopy });
            document.dispatchEvent(event);
            return this;
        }
        
        /**
         * Registriert einen einmaligen Event-Listener
         * @param {string} eventType - Typ des Events
         * @param {Function} callback - Callback-Funktion
         */
        once(eventType, callback) {
            const onceCallback = (event) => {
                this.off(eventType, onceCallback);
                callback(event);
            };
            return this.on(eventType, onceCallback);
        }
        
        /**
         * Entfernt alle Event-Listener
         * @param {string} eventType - Optional: Typ der zu entfernenden Events
         */
        removeAllListeners(eventType) {
            if (eventType) {
                if (this.listeners.has(eventType)) {
                    const listeners = this.listeners.get(eventType);
                    for (const { callback, options } of listeners) {
                        document.removeEventListener(eventType, callback, options);
                    }
                    this.listeners.delete(eventType);
                }
            } else {
                for (const [type, listeners] of this.listeners.entries()) {
                    for (const { callback, options } of listeners) {
                        document.removeEventListener(type, callback, options);
                    }
                }
                this.listeners.clear();
            }
            return this;
        }
    }
    
    // Event-Bus-Instanz erstellen
    const eventBus = new EventBus();
    
    /**
     * Benachrichtigt alle registrierten Listener über neue Daten
     * @param {string} category - Daten-Kategorie
     * @param {Object} data - Die neuen Daten
     * @private
     */
    function notifyDataUpdateListeners(category, data) {
        // Module benachrichtigen, die sich für diese Kategorie registriert haben
        if (state.dataUpdateListeners.has(category)) {
            state.dataUpdateListeners.get(category).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Fehler beim Ausführen des Callbacks für ${category}:`, error);
                    logError(`Event-Callback-Fehler für ${category}`, 'event', error);
                }
            });
        }
        
        // Event auslösen
        eventBus.emit(EventTypes.DATA_UPDATED, { category, data });
    }
    
    /**
     * Registriert einen neuen Listener für Daten-Updates
     * @param {string} category - Daten-Kategorie ('all', 'matrix', 'quantum', etc.)
     * @param {Function} callback - Callback-Funktion, die bei Updates aufgerufen wird
     */
    function onDataUpdate(category, callback) {
        // Legacy-Support für direkte Callbacks
        if (!state.dataUpdateListeners.has(category)) {
            state.dataUpdateListeners.set(category, new Set());
        }
        
        state.dataUpdateListeners.get(category).add(callback);
        
        // EventBus-Integration
        const eventHandler = (event) => {
            if (event.detail.category === category) {
                callback(event.detail.data);
            }
        };
        
        // Speichern des Event-Handlers in einer WeakMap zur späteren Referenz
        if (!state.eventHandlers) {
            state.eventHandlers = new WeakMap();
        }
        state.eventHandlers.set(callback, eventHandler);
        
        // Auf Event-Bus registrieren
        eventBus.on(EventTypes.DATA_UPDATED, eventHandler);
    }
    
    /**
     * Entfernt einen Listener für Daten-Updates
     * @param {string} category - Daten-Kategorie
     * @param {Function} callback - Zu entfernender Callback
     */
    function offDataUpdate(category, callback) {
        // Legacy-Support entfernen
        if (state.dataUpdateListeners.has(category)) {
            state.dataUpdateListeners.get(category).delete(callback);
        }
        
        // EventBus-Handler entfernen
        if (state.eventHandlers && state.eventHandlers.has(callback)) {
            const eventHandler = state.eventHandlers.get(callback);
            eventBus.off(EventTypes.DATA_UPDATED, eventHandler);
            state.eventHandlers.delete(callback);
        }
    }
    
    /**
     * Registriert einen Listener für Theme-Änderungen
     * @param {Function} callback - Funktion, die bei Theme-Änderungen aufgerufen wird
     */
    function onThemeChange(callback) {
        // Legacy-Support beibehalten
        state.themeListeners.add(callback);
        
        // EventBus-Integration
        eventBus.on(EventTypes.THEME_CHANGED, (event) => {
            callback(event.detail.isDarkMode);
        });
        
        // Sofort den aktuellen Theme-Status mitteilen
        callback(state.isDarkMode);
    }
    
    /**
     * Entfernt einen Listener für Theme-Änderungen
     * @param {Function} callback - Zu entfernender Callback
     */
    function offThemeChange(callback) {
        // Legacy-Support entfernen
        state.themeListeners.delete(callback);
        
        // Vom EventBus kann nicht direkt entfernt werden, da die Wrapper-Funktion
        // nicht mehr verfügbar ist. Bei einer vollständigen Migration sollte dies
        // wie bei onDataUpdate implementiert werden.
    }
    
    /**
     * Registriert einen Listener für Komponenten-Änderungen
     * @param {Function} callback - Funktion, die bei Komponentenwechsel aufgerufen wird
     */
    function onComponentChange(callback) {
        // Legacy-Support beibehalten
        state.componentChangeListeners.add(callback);
        
        // EventBus-Integration
        eventBus.on(EventTypes.COMPONENT_CHANGED, (event) => {
            callback(event.detail.componentId);
        });
        
        // Sofort den aktuellen Komponenten-Status mitteilen
        callback(state.selectedComponent);
    }
    
    /**
     * Entfernt einen Listener für Komponenten-Änderungen
     * @param {Function} callback - Zu entfernender Callback
     */
    function offComponentChange(callback) {
        // Legacy-Support entfernen
        state.componentChangeListeners.delete(callback);
    }
    
    /**
     * Registriert einen Listener für Kategorie-Änderungen
     * @param {Function} callback - Funktion, die bei Kategoriewechsel aufgerufen wird
     */
    function onCategoryChange(callback) {
        // Legacy-Support beibehalten
        state.categoryChangeListeners.add(callback);
        
        // EventBus-Integration
        eventBus.on(EventTypes.CATEGORY_CHANGED, (event) => {
            callback(event.detail.category);
        });
        
        // Sofort die aktuelle Kategorie mitteilen
        callback(state.activeCategory);
    }
    
    /**
     * Entfernt einen Listener für Kategorie-Änderungen
     * @param {Function} callback - Zu entfernender Callback
     */
    function offCategoryChange(callback) {
        // Legacy-Support entfernen
        state.categoryChangeListeners.delete(callback);
    }
    
    /**
     * Registriert einen einmaligen Listener für ein bestimmtes Event
     * @param {string} eventType - Typ des Events (aus EventTypes)
     * @param {Function} callback - Funktion, die beim Event aufgerufen wird
     */
    function once(eventType, callback) {
        return eventBus.once(eventType, callback);
    }
    
    /**
     * Registriert einen Listener für ein bestimmtes Event
     * @param {string} eventType - Typ des Events (aus EventTypes)
     * @param {Function} callback - Funktion, die beim Event aufgerufen wird
     * @param {Object} options - Optionen für addEventListener
     */
    function on(eventType, callback, options = {}) {
        return eventBus.on(eventType, callback, options);
    }
    
    /**
     * Entfernt einen Listener für ein bestimmtes Event
     * @param {string} eventType - Typ des Events (aus EventTypes)
     * @param {Function} callback - Zu entfernender Callback
     */
    function off(eventType, callback) {
        return eventBus.off(eventType, callback);
    }
    
    /**
     * Theme-Management
     */
    
    /**
     * Wechselt zwischen Light- und Dark-Mode
     * @returns {boolean} Neuer Dark-Mode-Status
     */
    function toggleTheme() {
        state.isDarkMode = !state.isDarkMode;
        
        // Legacy-Support: Alle direkten Theme-Listener benachrichtigen
        state.themeListeners.forEach(callback => {
            try {
                callback(state.isDarkMode);
            } catch (error) {
                console.error('Fehler beim Ausführen des Theme-Callbacks:', error);
                logError('Theme-Callback-Fehler', 'theme', error);
            }
        });
        
        // Event mit dem neuen EventBus auslösen
        eventBus.emit(EventTypes.THEME_CHANGED, { isDarkMode: state.isDarkMode });
        
        // DOM-Attribut setzen für CSS-Selektoren
        document.documentElement.setAttribute('data-theme', state.isDarkMode ? 'dark' : 'light');
        
        // Barrierefreiheit: ARIA-Attribut für Screenreader setzen
        document.documentElement.setAttribute('aria-theme', state.isDarkMode ? 'dark' : 'light');
        
        return state.isDarkMode;
    }
    
    /**
     * Modul-Registrierung und -Verwaltung
     */
    
    /**
     * Definiert die erwartete Schnittstelle eines Dashboard-Moduls
     * @typedef {Object} ModuleInterface
     * @property {Function} init - Initialisiert das Modul
     * @property {Function} [update] - Aktualisiert das Modul mit neuen Daten (optional)
     * @property {Function} [destroy] - Bereinigt Ressourcen beim Entfernen des Moduls (optional)
     * @property {Object} [config] - Konfigurationsoptionen für das Modul (optional)
     */
    
    /**
     * Registriert ein neues Dashboard-Modul
     * @param {string} moduleId - Eindeutige Kennung des Moduls
     * @param {ModuleInterface} moduleInterface - Schnittstelle des Moduls
     * @returns {boolean} True bei erfolgreicher Registrierung, sonst false
     */
    function registerModule(moduleId, moduleInterface) {
        if (state.registeredModules.has(moduleId)) {
            console.warn(`Modul ${moduleId} wurde bereits registriert`);
            return false;
        }
        
        // Modul-Schnittstelle validieren
        if (!moduleInterface || typeof moduleInterface.init !== 'function') {
            console.error(`Modul ${moduleId} hat keine gültige Schnittstelle`);
            return false;
        }
        
        // Modul mit Version und Registrierungszeitstempel speichern
        const moduleInfo = {
            id: moduleId,
            interface: moduleInterface,
            registeredAt: new Date().toISOString(),
            version: moduleInterface.config?.version || '1.0.0',
            initialized: false,
            dependencies: moduleInterface.config?.dependencies || []
        };
        
        state.registeredModules.set(moduleId, moduleInfo);
        console.log(`Modul ${moduleId} (v${moduleInfo.version}) erfolgreich registriert`);
        
        // Event über Modul-Registrierung auslösen
        eventBus.emit(EventTypes.MODULE_REGISTERED, { moduleId, moduleInfo });
        
        // Falls Dashboard bereits initialisiert wurde und keine abhängigen Module ausstehen,
        // Modul sofort initialisieren
        if (!state.initializing && canInitializeModule(moduleInfo)) {
            initializeModule(moduleInfo);
        }
        
        return true;
    }
    
    /**
     * Überprüft, ob ein Modul initialisiert werden kann (alle Abhängigkeiten erfüllt)
     * @param {Object} moduleInfo - Informationen zum Modul
     * @returns {boolean} True, wenn alle Abhängigkeiten erfüllt sind
     * @private
     */
    function canInitializeModule(moduleInfo) {
        // Wenn keine Abhängigkeiten definiert sind, kann sofort initialisiert werden
        if (!moduleInfo.dependencies || moduleInfo.dependencies.length === 0) {
            return true;
        }
        
        // Prüfen, ob alle abhängigen Module registriert und initialisiert sind
        return moduleInfo.dependencies.every(depId => {
            if (!state.registeredModules.has(depId)) {
                console.warn(`Abhängigkeit ${depId} für Modul ${moduleInfo.id} nicht registriert`);
                return false;
            }
            
            const depModule = state.registeredModules.get(depId);
            if (!depModule.initialized) {
                console.warn(`Abhängigkeit ${depId} für Modul ${moduleInfo.id} nicht initialisiert`);
                return false;
            }
            
            return true;
        });
    }
    
    /**
     * Initialisiert ein einzelnes Modul
     * @param {Object} moduleInfo - Informationen zum Modul
     * @private
     */
    function initializeModule(moduleInfo) {
        if (moduleInfo.initialized) {
            return;
        }
        
        try {
            moduleInfo.interface.init();
            moduleInfo.initialized = true;
            console.log(`Modul ${moduleInfo.id} initialisiert`);
            
            // Andere Module benachrichtigen, die auf dieses Modul warten könnten
            checkDependentModules(moduleInfo.id);
        } catch (error) {
            console.error(`Fehler beim Initialisieren von Modul ${moduleInfo.id}:`, error);
            logError(`Modul-Initialisierungsfehler: ${moduleInfo.id}`, 'module', error);
        }
    }
    
    /**
     * Prüft, ob andere Module nun initialisiert werden können, nachdem ein Modul initialisiert wurde
     * @param {string} initializedModuleId - ID des gerade initialisierten Moduls
     * @private
     */
    function checkDependentModules(initializedModuleId) {
        state.registeredModules.forEach((moduleInfo, moduleId) => {
            if (!moduleInfo.initialized && 
                moduleInfo.dependencies && 
                moduleInfo.dependencies.includes(initializedModuleId)) {
                
                if (canInitializeModule(moduleInfo)) {
                    initializeModule(moduleInfo);
                }
            }
        });
    }
    
    /**
     * Initialisierung aller registrierten Module
     */
    function initModules() {
        // Zuerst Module ohne Abhängigkeiten initialisieren
        const independentModules = [];
        const dependentModules = [];
        
        state.registeredModules.forEach(moduleInfo => {
            if (!moduleInfo.dependencies || moduleInfo.dependencies.length === 0) {
                independentModules.push(moduleInfo);
            } else {
                dependentModules.push(moduleInfo);
            }
        });
        
        // Unabhängige Module initialisieren
        independentModules.forEach(moduleInfo => {
            initializeModule(moduleInfo);
        });
        
        // Abhängige Module initialisieren (sofern ihre Abhängigkeiten erfüllt sind)
        let remainingModules = dependentModules.length;
        let lastRemainingCount = -1;
        
        // Versuchen, alle abhängigen Module zu initialisieren,
        // bis keine Änderung mehr stattfindet
        while (remainingModules > 0 && remainingModules !== lastRemainingCount) {
            lastRemainingCount = remainingModules;
            
            for (let i = dependentModules.length - 1; i >= 0; i--) {
                const moduleInfo = dependentModules[i];
                
                if (canInitializeModule(moduleInfo)) {
                    initializeModule(moduleInfo);
                    dependentModules.splice(i, 1);
                }
            }
            
            remainingModules = dependentModules.length;
        }
        
        // Falls noch Module übrig sind, warnen
        if (dependentModules.length > 0) {
            console.warn(`${dependentModules.length} Module konnten wegen nicht erfüllter Abhängigkeiten nicht initialisiert werden:`, 
                dependentModules.map(m => m.id).join(', '));
        }
        
        state.initializing = false;
        
        // Dashboard-Initialisierungsevent auslösen
        eventBus.emit(EventTypes.DASHBOARD_INITIALIZED, { 
            initializedModules: Array.from(state.registeredModules.values())
                .filter(m => m.initialized)
                .map(m => m.id),
            uninitializedModules: dependentModules.map(m => m.id)
        });
    }
    
    /**
     * Komponenten- und Kategorie-Verwaltung
     */
    
    /**
     * Setzt die aktive Benchmark-Kategorie
     * @param {string} category - Die neue aktive Kategorie
     */
    function setActiveCategory(category) {
        if (state.activeCategory === category) return;
        
        // Überprüfen, ob die Kategorie unterstützt wird
        if (!isSupportedCategory(category)) {
            console.warn(`Kategorie '${category}' wird nicht unterstützt.`);
            return;
        }
        
        const previousCategory = state.activeCategory;
        state.activeCategory = category;
        
        // Legacy-Support: Alle direkten Kategorie-Listener benachrichtigen
        state.categoryChangeListeners.forEach(callback => {
            try {
                callback(category);
            } catch (error) {
                console.error('Fehler beim Ausführen des Kategorie-Callbacks:', error);
                logError('Kategorie-Callback-Fehler', 'category', error);
            }
        });
        
        // Event mit dem EventBus auslösen
        eventBus.emit(EventTypes.CATEGORY_CHANGED, { 
            category,
            previousCategory
        });
        
        // ARIA-Live-Region aktualisieren für Barrierefreiheit
        const liveRegion = document.getElementById('vxor-live-region') || createLiveRegion();
        liveRegion.textContent = `Kategorie ${category} aktiviert`;
        
        // Daten für die neue Kategorie laden
        fetchBenchmarkData(state.selectedComponent, category);
    }
    
    /**
     * Erstellt eine ARIA-Live-Region, falls noch nicht vorhanden
     * @private
     * @returns {HTMLElement} Die Live-Region
     */
    function createLiveRegion() {
        const liveRegion = document.createElement('div');
        liveRegion.id = 'vxor-live-region';
        liveRegion.className = 'visually-hidden';
        liveRegion.setAttribute('aria-live', 'polite');
        liveRegion.setAttribute('aria-atomic', 'true');
        document.body.appendChild(liveRegion);
        return liveRegion;
    }
    
    /**
     * Wählt eine Komponente aus
     * @param {string|null} componentId - ID der Komponente oder null für alle
     */
    function selectComponent(componentId) {
        if (state.selectedComponent === componentId) return;
        
        const previousComponent = state.selectedComponent;
        state.selectedComponent = componentId;
        
        // Legacy-Support: Alle direkten Komponenten-Listener benachrichtigen
        state.componentChangeListeners.forEach(callback => {
            try {
                callback(componentId);
            } catch (error) {
                console.error('Fehler beim Ausführen des Komponenten-Callbacks:', error);
                logError('Komponenten-Callback-Fehler', 'component', error);
            }
        });
        
        // Event mit dem EventBus auslösen
        eventBus.emit(EventTypes.COMPONENT_CHANGED, { 
            componentId,
            previousComponent
        });
        
        // ARIA-Live-Region aktualisieren für Barrierefreiheit
        const liveRegion = document.getElementById('vxor-live-region') || createLiveRegion();
        liveRegion.textContent = componentId ? 
            `Komponente ${componentId} ausgewählt` : 
            'Alle Komponenten ausgewählt';
        
        // Cache leeren und Daten neu laden
        clearCache();
        fetchBenchmarkData(componentId, state.activeCategory);
    }
    
    /**
     * Formatierung und Hilfsfunktionen
     */
    
    /**
     * Formatiert eine Zeitdauer in Millisekunden als lesbare Zeichenkette
     * @param {number} seconds - Zeitdauer in Sekunden
     * @returns {string} Formatierte Zeitdauer
     */
    function formatDuration(seconds) {
        if (seconds < 0.001) {
            return `${(seconds * 1000000).toFixed(2)} µs`;
        } else if (seconds < 1) {
            return `${(seconds * 1000).toFixed(2)} ms`;
        } else {
            return `${seconds.toFixed(2)} s`;
        }
    }
    
    /**
     * Überprüft, ob die angegebene Benchmark-Kategorie unterstützt wird
     * @param {string} category - Zu prüfende Kategorie
     * @returns {boolean} True, wenn die Kategorie unterstützt wird
     */
    function isSupportedCategory(category) {
        return [
            'all', 'matrix', 'quantum', 'tmaths',
            'mlperf', 'swe', 'security'
        ].includes(category);
    }
    
    /**
     * Gibt den aktuellen Zustand des gesamten Dashboards zurück
     * @returns {Object} Aktueller Dashboard-Zustand
     */
    function getCurrentState() {
        return {
            selectedComponent: state.selectedComponent,
            activeCategory: state.activeCategory,
            isDarkMode: state.isDarkMode,
            initializing: state.initializing
        };
    }
    
    /**
     * UI-Management-Funktionen
     */
    
    /**
     * Initialisiert die Tab-Navigation mit Barrierefreiheitsunterstützung
     * @param {Object} options - Konfigurationsoptionen
     * @param {string} [options.tabsSelector='.category-tabs'] - Selector für den Tab-Container
     * @param {string} [options.tabButtonSelector='.tab-button'] - Selector für die Tab-Buttons
     * @param {string} [options.tabContentSelector='.benchmark-section'] - Selector für die Tab-Inhalte
     * @param {string} [options.activeTabClass='active'] - CSS-Klasse für aktive Tabs
     */
    function setupTabNavigation(options = {}) {
        const config = Object.assign({
            tabsSelector: '.category-tabs',
            tabButtonSelector: '.tab-button',
            tabContentSelector: '.benchmark-section',
            activeTabClass: 'active'
        }, options);
        
        const tabsContainer = document.querySelector(config.tabsSelector);
        if (!tabsContainer) {
            console.error('Tab-Container nicht gefunden:', config.tabsSelector);
            return;
        }
        
        // ARIA-Attribute für Barrierefreiheit setzen
        tabsContainer.setAttribute('role', 'tablist');
        tabsContainer.setAttribute('aria-orientation', 'horizontal');
        
        const tabButtons = tabsContainer.querySelectorAll(config.tabButtonSelector);
        const tabPanels = document.querySelectorAll(config.tabContentSelector);
        
        // Tab-Buttons mit ARIA-Attributen konfigurieren
        tabButtons.forEach((button, index) => {
            const tabId = button.getAttribute('data-category') || `tab-${index}`;
            const panelId = `panel-${tabId}`;
            
            // ARIA-Attribute für Tabs
            button.setAttribute('role', 'tab');
            button.setAttribute('aria-selected', button.classList.contains(config.activeTabClass) ? 'true' : 'false');
            button.setAttribute('id', tabId);
            button.setAttribute('aria-controls', panelId);
            button.setAttribute('tabindex', button.classList.contains(config.activeTabClass) ? '0' : '-1');
            
            // Entsprechendes Panel finden und ARIA-Attribute setzen
            const panel = document.getElementById(tabId + '-benchmarks') || 
                          (tabPanels[index] ? tabPanels[index] : null);
            
            if (panel) {
                panel.setAttribute('role', 'tabpanel');
                panel.setAttribute('id', panelId);
                panel.setAttribute('aria-labelledby', tabId);
                panel.setAttribute('tabindex', '0');
                panel.setAttribute('hidden', !button.classList.contains(config.activeTabClass));
            }
            
            // Click-Event-Handler
            button.addEventListener('click', (event) => {
                event.preventDefault();
                activateTab(button, tabButtons, tabPanels, config);
                
                // Kategorie aktualisieren und Daten laden
                const category = button.getAttribute('data-category');
                if (category && isSupportedCategory(category)) {
                    setActiveCategory(category);
                }
                
                // Benutzereingabe-Event protokollieren
                eventBus.emit(EventTypes.USER_INTERACTION, { 
                    type: 'tab_change', 
                    category,
                    elementId: tabId
                });
            });
            
            // Keydown-Event-Handler für Tastaturnavigation
            button.addEventListener('keydown', (event) => {
                handleTabKeyNavigation(event, button, tabButtons, tabPanels, config);
            });
        });
    }
    
    /**
     * Aktiviert einen Tab und zeigt den entsprechenden Inhaltsbereich an
     * @param {HTMLElement} selectedTab - Der ausgewählte Tab-Button
     * @param {NodeList} allTabs - Alle Tab-Buttons
     * @param {NodeList} allPanels - Alle Tab-Panels
     * @param {Object} config - Konfigurationsoptionen
     * @private
     */
    function activateTab(selectedTab, allTabs, allPanels, config) {
        // Alle Tabs deaktivieren
        allTabs.forEach(tab => {
            tab.classList.remove(config.activeTabClass);
            tab.setAttribute('aria-selected', 'false');
            tab.setAttribute('tabindex', '-1');
        });
        
        // Alle Panels ausblenden
        allPanels.forEach(panel => {
            panel.setAttribute('hidden', 'true');
        });
        
        // Ausgewählten Tab aktivieren
        selectedTab.classList.add(config.activeTabClass);
        selectedTab.setAttribute('aria-selected', 'true');
        selectedTab.setAttribute('tabindex', '0');
        
        // Zugehöriges Panel anzeigen
        const tabCategory = selectedTab.getAttribute('data-category');
        const panelId = selectedTab.getAttribute('aria-controls');
        const panel = document.getElementById(panelId) || 
                      document.getElementById(tabCategory + '-benchmarks');
        
        if (panel) {
            panel.removeAttribute('hidden');
        }
    }
    
    /**
     * Behandelt Tastatur-Navigation für Tabs nach WAI-ARIA Best Practices
     * @param {KeyboardEvent} event - Tastatur-Event
     * @param {HTMLElement} currentTab - Aktueller Tab-Button
     * @param {NodeList} allTabs - Alle Tab-Buttons
     * @param {NodeList} allPanels - Alle Tab-Panels
     * @param {Object} config - Konfigurationsoptionen
     * @private
     */
    function handleTabKeyNavigation(event, currentTab, allTabs, allPanels, config) {
        const tabsArray = Array.from(allTabs);
        const currentIndex = tabsArray.indexOf(currentTab);
        let nextTab = null;
        
        switch (event.key) {
            case 'ArrowRight':
            case 'ArrowDown':
                event.preventDefault();
                nextTab = tabsArray[currentIndex + 1] || tabsArray[0];
                break;
                
            case 'ArrowLeft':
            case 'ArrowUp':
                event.preventDefault();
                nextTab = tabsArray[currentIndex - 1] || tabsArray[tabsArray.length - 1];
                break;
                
            case 'Home':
                event.preventDefault();
                nextTab = tabsArray[0];
                break;
                
            case 'End':
                event.preventDefault();
                nextTab = tabsArray[tabsArray.length - 1];
                break;
                
            case 'Enter':
            case ' ':
                event.preventDefault();
                activateTab(currentTab, allTabs, allPanels, config);
                
                // Kategorie aktualisieren und Daten laden
                const category = currentTab.getAttribute('data-category');
                if (category && isSupportedCategory(category)) {
                    setActiveCategory(category);
                }
                return;
        }
        
        if (nextTab) {
            nextTab.focus();
            // Bei automatischer Aktivierung beim Fokussieren
            // Uncomment diese Zeile:
            // activateTab(nextTab, allTabs, allPanels, config);
        }
    }
    
    /**
     * Initialisiert einen Theme-Toggle mit Icon und Barrierefreiheitsunterstützung
     * @param {string} toggleSelector - CSS-Selector für den Theme-Toggle-Button
     */
    function setupThemeToggle(toggleSelector = '#theme-toggle') {
        const themeToggle = document.querySelector(toggleSelector);
        if (!themeToggle) {
            console.log('Theme-Toggle nicht gefunden. Erstelle neuen Button...');
            createThemeToggle();
            return;
        }
        
        // ARIA-Attribute für Barrierefreiheit setzen
        themeToggle.setAttribute('role', 'switch');
        themeToggle.setAttribute('aria-checked', state.isDarkMode ? 'true' : 'false');
        themeToggle.setAttribute('aria-label', state.isDarkMode ? 'Zum Light-Mode wechseln' : 'Zum Dark-Mode wechseln');
        
        // Initialen Zustand setzen
        updateThemeToggle(themeToggle, state.isDarkMode);
        
        // Event-Listener für Klicks
        themeToggle.addEventListener('click', (event) => {
            event.preventDefault();
            const newDarkMode = toggleTheme();
            updateThemeToggle(themeToggle, newDarkMode);
            
            // Benutzereingabe-Event protokollieren
            eventBus.emit(EventTypes.USER_INTERACTION, { 
                type: 'theme_toggle', 
                newTheme: newDarkMode ? 'dark' : 'light'
            });
        });
        
        // Event-Listener für Tastatureingaben
        themeToggle.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                const newDarkMode = toggleTheme();
                updateThemeToggle(themeToggle, newDarkMode);
            }
        });
        
        // Click-Event registrieren
        themeToggle.addEventListener('click', toggleTheme);
    }
    
    /**
     * Erstellt einen Theme-Toggle-Button, falls keiner im DOM vorhanden ist
     * @private
     */
    function createThemeToggle() {
        const header = document.querySelector('.header');
        if (!header) return;
        
        const themeToggle = document.createElement('button');
        themeToggle.id = 'theme-toggle';
        themeToggle.className = 'theme-toggle-button';
        themeToggle.innerHTML = `
            <span class="theme-toggle-icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" class="light-icon">
                    <path fill="currentColor" d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zM2 13h2c.55 0 1-.45 1-1s-.45-1-1-1H2c-.55 0-1 .45-1 1s.45 1 1 1zm18 0h2c.55 0 1-.45 1-1s-.45-1-1-1h-2c-.55 0-1 .45-1 1s.45 1 1 1zM11 2v2c0 .55.45 1 1 1s1-.45 1-1V2c0-.55-.45-1-1-1s-1 .45-1 1zm0 18v2c0 .55.45 1 1 1s1-.45 1-1v-2c0-.55-.45-1-1-1s-1 .45-1 1zM5.99 4.58c-.39-.39-1.03-.39-1.41 0-.39.39-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0 .39-.39.39-1.03 0-1.41L5.99 4.58zm12.37 12.37c-.39-.39-1.03-.39-1.41 0-.39.39-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0 .39-.39.39-1.03 0-1.41l-1.06-1.06zm1.06-10.96c.39-.39.39-1.03 0-1.41-.39-.39-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06zM7.05 18.36c.39-.39.39-1.03 0-1.41-.39-.39-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06z"></path>
                </svg>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" class="dark-icon">
                    <path fill="currentColor" d="M9.37 5.51c-.18.64-.27 1.31-.27 1.99 0 4.08 3.32 7.4 7.4 7.4.68 0 1.35-.09 1.99-.27C17.45 17.19 14.93 19 12 19c-3.86 0-7-3.14-7-7 0-2.93 1.81-5.45 4.37-6.49zM12 3c-4.97 0-9 4.03-9 9s4.03 9 9 9 9-4.03 9-9c0-.46-.04-.92-.1-1.36-.98 1.37-2.58 2.26-4.4 2.26-2.98 0-5.4-2.42-5.4-5.4 0-1.81.89-3.42 2.26-4.4-.44-.06-.9-.1-1.36-.1z"></path>
                </svg>
            </span>
            <span class="visually-hidden">Theme wechseln</span>
        `;
        
        // CSS-Styles hinzufügen, falls noch nicht in externer CSS-Datei
        if (!document.querySelector('style#theme-toggle-styles')) {
            const style = document.createElement('style');
            style.id = 'theme-toggle-styles';
            style.textContent = `
                .theme-toggle-button {
                    background: transparent;
                    border: none;
                    color: white;
                    cursor: pointer;
                    padding: 8px;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    position: absolute;
                    right: 20px;
                    top: 20px;
                    transition: background-color 0.3s;
                }
                .theme-toggle-button:hover {
                    background-color: rgba(255, 255, 255, 0.1);
                }
                .theme-toggle-button:focus {
                    outline: 2px solid white;
                    outline-offset: 2px;
                }
                .theme-toggle-icon {
                    position: relative;
                    width: 24px;
                    height: 24px;
                }
                .light-icon, .dark-icon {
                    position: absolute;
                    top: 0;
                    left: 0;
                    transition: opacity 0.3s, transform 0.3s;
                }
                [data-theme="dark"] .light-icon,
                :root:not([data-theme="dark"]) .dark-icon {
                    opacity: 1;
                    transform: rotate(0);
                }
                [data-theme="dark"] .dark-icon,
                :root:not([data-theme="dark"]) .light-icon {
                    opacity: 0;
                    transform: rotate(90deg);
                }
            `;
            document.head.appendChild(style);
        }
        
        // ARIA-Attribute für Barrierefreiheit setzen
        themeToggle.setAttribute('role', 'switch');
        themeToggle.setAttribute('aria-checked', state.isDarkMode ? 'true' : 'false');
        themeToggle.setAttribute('aria-label', state.isDarkMode ? 'Zum Light-Mode wechseln' : 'Zum Dark-Mode wechseln');
        
        // Initialen Theme-Zustand setzen
        updateThemeToggle(themeToggle, state.isDarkMode);
        
        // Theme-Toggle zum Header hinzufügen
        header.appendChild(themeToggle);
        
        // Event-Listener für Klicks
        themeToggle.addEventListener('click', (event) => {
            event.preventDefault();
            const newDarkMode = toggleTheme();
            updateThemeToggle(themeToggle, newDarkMode);
        });
    }
    
    /**
     * Aktualisiert das Aussehen des Theme-Toggle-Buttons
     * @param {HTMLElement} toggleButton - Der Theme-Toggle-Button
     * @param {boolean} isDarkMode - Ob der Dark-Mode aktiv ist
     * @private
     */
    function updateThemeToggle(toggleButton, isDarkMode) {
        toggleButton.setAttribute('aria-checked', isDarkMode ? 'true' : 'false');
        toggleButton.setAttribute('aria-label', isDarkMode ? 'Zum Light-Mode wechseln' : 'Zum Dark-Mode wechseln');
        
        // Optional: Visuelles Feedback
        const textEl = toggleButton.querySelector('.visually-hidden');
        if (textEl) {
            textEl.textContent = isDarkMode ? 'Zum Light-Mode wechseln' : 'Zum Dark-Mode wechseln';
        }
    }
    
    /**
     * Initialisiert den Fokus-Ring für bessere Tastatur-Navigation
     */
    function setupFocusRing() {
        // Klasse hinzufügen, wenn Tastatur zur Navigation verwendet wird
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Tab') {
                document.body.classList.add('keyboard-navigation');
            }
        });
        
        // Klasse entfernen, wenn Maus zur Navigation verwendet wird
        document.addEventListener('mousedown', () => {
            document.body.classList.remove('keyboard-navigation');
        });
        
        // CSS-Styles hinzufügen, falls noch nicht in externer CSS-Datei
        if (!document.querySelector('style#focus-ring-styles')) {
            const style = document.createElement('style');
            style.id = 'focus-ring-styles';
            style.textContent = `
                /* Fokus-Styles nur anzeigen, wenn Tastatur zur Navigation verwendet wird */
                .keyboard-navigation *:focus {
                    outline: 2px solid var(--accent-color, #673ab7) !important;
                    outline-offset: 3px !important;
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    // DEBUG-Modus Flag
    const DEBUG_MODE = false;
    
    // Öffentliche API des Moduls
    const publicAPI = {
        // Daten- und API-Funktionen
        fetchComponentStatus,
        fetchHardwareMetrics,
        fetchBenchmarkData,
        retryFetch,
        clearCache,
        
        // Event-System
        onDataUpdate,
        offDataUpdate,
        onThemeChange,
        offThemeChange,
        onComponentChange,
        offComponentChange,
        onCategoryChange,
        offCategoryChange,
        on,
        off,
        once,
        EventTypes,
        
        // UI-Management
        setupTabNavigation,
        setupThemeToggle,
        setupFocusRing,
        
        // Theme-Management
        toggleTheme,
        
        // Modul-Verwaltung
        registerModule,
        initModules,
        
        // Kategorie- und Komponenten-Verwaltung
        setActiveCategory,
        selectComponent,
        
        // Aktuelle Zustände abfragen
        getCurrentState,
        
        // Hilfsfunktionen
        formatDuration,
        isSupportedCategory,
        
        // Fehlerbehandlung
        logError,
        handleApiError,
    };
    
    // SICHERHEIT: Debug-Funktionen nur im Debug-Modus hinzufügen
    if (DEBUG_MODE) {
        // Diese Funktionen sind im Produktionsmodus nicht verfügbar
        publicAPI.debug = {
            getCache: () => ({ ...benchmarkCache }),
            logModules: () => console.log('Registrierte Module:', [...state.registeredModules.keys()])
        };
    }
    
    return publicAPI;
})();

// Im Code-Splitting-Modus erfolgt keine automatische Initialisierung mehr
// Stattdessen wird die Initialisierung durch den ModuleLoader gesteuert

// Event-Listener für automatische Theme-Erkennung
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    // Falls sich das System-Theme ändert, Dashboard-Theme anpassen
    // aber nur, wenn der Nutzer nicht explizit ein Theme gesetzt hat
    if (VXORUtils.getCurrentState().isDarkMode !== e.matches) {
        VXORUtils.toggleTheme();
    }
});

// Modul für AMD/CommonJS/Browser-Globale exportieren
if (typeof define === 'function' && define.amd) {
    // AMD-Export
    define([], function() {
        return VXORUtils;
    });
} else if (typeof module === 'object' && module.exports) {
    // CommonJS-Export
    module.exports = VXORUtils;
} else {
    // Browser-Globale
    window.VXORUtils = VXORUtils;
}
