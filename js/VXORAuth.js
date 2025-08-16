/**
 * VXORAuth.js
 * Authentifizierungs- und Sicherheitsmodul für das VXOR Benchmark Dashboard
 * 
 * Stellt zentrale Funktionen zur Authentifizierung und Sicherung bereit:
 * - API-Token-Verwaltung
 * - CSRF-Schutz
 * - Schema-Validierung für API-Antworten
 */

const VXORAuth = (function() {
    'use strict';
    
    // Private Variablen für Auth-Token und CSRF-Token
    let apiToken = null;
    let csrfToken = null;
    
    // Flag für Debug-Modus
    const DEBUG_MODE = false;
    
    // Schemas für API-Antworten
    const apiSchemas = {
        // Benchmark-Daten-Schema
        benchmarkData: {
            required: ['results', 'timestamp'],
            validate: function(data) {
                if (!data || typeof data !== 'object') return false;
                if (!Array.isArray(data.results)) return false;
                if (typeof data.timestamp !== 'number') return false;
                return true;
            }
        },
        // Komponenten-Status-Schema
        componentStatus: {
            required: ['components', 'timestamp'],
            validate: function(data) {
                if (!data || typeof data !== 'object') return false;
                if (!Array.isArray(data.components)) return false;
                if (typeof data.timestamp !== 'number') return false;
                return true;
            }
        },
        // Hardware-Metriken-Schema
        hardwareMetrics: {
            required: ['metrics', 'timestamp'],
            validate: function(data) {
                if (!data || typeof data !== 'object') return false;
                if (!data.metrics || typeof data.metrics !== 'object') return false;
                if (typeof data.timestamp !== 'number') return false;
                return true;
            }
        }
    };
    
    /**
     * Initialisiert die Authentifizierung
     * @returns {Promise<boolean>} True bei erfolgreicher Initialisierung
     */
    async function init() {
        try {
            // Im realen System würde hier eine Authentifizierung stattfinden
            // In diesem Beispiel setzen wir ein Dummy-Token
            apiToken = 'vxor-secure-api-token-' + Date.now();
            
            // CSRF-Token generieren (normalerweise vom Server bereitgestellt)
            csrfToken = generateCSRFToken();
            
            return true;
        } catch (error) {
            console.error('Fehler bei der Authentifizierungs-Initialisierung:', error);
            return false;
        }
    }
    
    /**
     * Generiert ein CSRF-Token
     * @returns {string} CSRF-Token
     * @private
     */
    function generateCSRFToken() {
        // In einer realen Anwendung würde dieses Token vom Server kommen
        // Hier generieren wir ein zufälliges Token
        const randomBytes = new Uint8Array(16);
        window.crypto.getRandomValues(randomBytes);
        return Array.from(randomBytes)
            .map(b => b.toString(16).padStart(2, '0'))
            .join('');
    }
    
    /**
     * Gibt das aktuelle API-Token zurück
     * @returns {string|null} API-Token oder null
     */
    function getToken() {
        return apiToken;
    }
    
    /**
     * Gibt das aktuelle CSRF-Token zurück
     * @returns {string|null} CSRF-Token oder null
     */
    function getCSRFToken() {
        return csrfToken;
    }
    
    /**
     * Validiert API-Antwortdaten gegen ein Schema
     * @param {Object} data - Die zu validierenden Daten
     * @param {string} schemaType - Schema-Typ ('benchmarkData', 'componentStatus', 'hardwareMetrics')
     * @returns {boolean} True, wenn Validierung erfolgreich
     */
    function validateApiResponse(data, schemaType) {
        // Prüfen, ob das Schema existiert
        if (!apiSchemas[schemaType] || typeof apiSchemas[schemaType].validate !== 'function') {
            console.error(`Unbekanntes Schema: ${schemaType}`);
            return false;
        }
        
        // Schema-Validierung durchführen
        return apiSchemas[schemaType].validate(data);
    }
    
    /**
     * Erzeugt sicher einen DOM-Element-Inhalt (statt innerHTML)
     * @param {HTMLElement} element - DOM-Element
     * @param {string} htmlContent - Der zu setzende HTML-Inhalt
     */
    function safelySetContent(element, htmlContent) {
        if (!element) return;
        
        // Existierende Kinder entfernen
        while (element.firstChild) {
            element.removeChild(element.firstChild);
        }
        
        // Temporäres Container-Element erstellen
        const tempContainer = document.createElement('div');
        
        // Content als Text setzen (vermeidet Script-Ausführung)
        tempContainer.textContent = htmlContent;
        
        // Text-Inhalt in das Ziel-Element übernehmen
        element.appendChild(document.createTextNode(tempContainer.textContent));
    }
    
    /**
     * Validiert Daten in einem Event-Detail-Objekt
     * @param {any} detail - Das zu validierende Detail-Objekt
     * @returns {boolean} True, wenn Validierung erfolgreich
     */
    function validateEventDetail(detail) {
        // Prüfen, ob es ein Objekt ist (kein Array, keine null)
        if (typeof detail !== 'object' || Array.isArray(detail) || detail === null) {
            return false;
        }
        
        // Prüfen, ob potentiell gefährliche Eigenschaften vorhanden sind
        const detailStr = JSON.stringify(detail);
        const dangerousPatterns = [
            /<script>/i,
            /<\/script>/i,
            /<img[^>]+onerror=/i,
            /javascript:/i,
            /eval\(/i,
            /Function\(/i
        ];
        
        // Prüfen, ob eines der gefährlichen Muster enthalten ist
        return !dangerousPatterns.some(pattern => pattern.test(detailStr));
    }
    
    /**
     * Validiert URL gegen eine Whitelist
     * @param {string} url - Die zu prüfende URL
     * @returns {boolean} True, wenn die URL erlaubt ist
     */
    function isUrlWhitelisted(url) {
        const allowedDomains = [
            location.origin, // Aktuelle Domain
            'https://cdn.jsdelivr.net', // CDN für externe Bibliotheken
            'https://trusted.cdn.com' // Beispiel für vertrauenswürdigen CDN
        ];
        
        try {
            const urlObj = new URL(url, location.origin);
            return allowedDomains.some(domain => 
                urlObj.origin === domain || urlObj.href.startsWith(domain)
            );
        } catch (error) {
            console.error('Ungültige URL:', url);
            return false;
        }
    }
    
    // Debug-Hilfsfunktionen (nur im Debug-Modus verfügbar)
    let debugFunctions = null;
    if (DEBUG_MODE) {
        debugFunctions = {
            getApiSchemas: () => ({ ...apiSchemas }),
            logAuthState: () => console.log('Auth-Status:', { apiToken, csrfToken })
        };
    }
    
    // Öffentliche API
    return {
        init,
        getToken,
        getCSRFToken,
        validateApiResponse,
        safelySetContent,
        validateEventDetail,
        isUrlWhitelisted,
        debug: debugFunctions
    };
})();

// Exportieren für Modul-Systeme
if (typeof define === 'function' && define.amd) {
    // AMD-Export
    define([], function() {
        return VXORAuth;
    });
} else if (typeof module === 'object' && module.exports) {
    // CommonJS-Export
    module.exports = VXORAuth;
} else {
    // Browser-Globale (nur Debug-Zugriff entfernen)
    Object.defineProperty(window, 'VXORAuth', {
        value: VXORAuth,
        writable: false,
        configurable: false
    });
}
