/**
 * VXOR Lazy Loader
 * Optimierte Lösung für Lazy Loading von verschiedenen Asset-Typen
 * Version 1.0.0 (Phase 7.2 - Leistungsoptimierung)
 */

const VXORLazyLoader = (function() {
    'use strict';
    
    // Konfiguration
    const config = {
        imageLazyLoadThreshold: 200, // px vor dem sichtbaren Bereich laden
        componentLazyLoadThreshold: 500, // px vor dem sichtbaren Bereich laden
        debounceDelay: 150, // ms Verzögerung für Scroll-Events
    };
    
    // Speichert registrierte UI-Komponenten
    const uiComponents = new Map();
    
    // IntersectionObserver für Bild-Lazy-Loading
    let imageObserver;
    
    // IntersectionObserver für Komponenten-Lazy-Loading
    let componentObserver;
    
    // Performance-Metriken
    const metrics = {
        originalPayload: 0,
        lazyLoadedPayload: 0,
        lazyLoadedComponents: 0,
        lazyLoadedImages: 0,
        savedRequests: 0
    };
    
    /**
     * Initialisiert das Lazy Loading System
     */
    function init() {
        console.log('VXOR Lazy Loader initialisiert');
        setupImageLazyLoading();
        setupComponentLazyLoading();
        setupScrollListeners();
    }
    
    /**
     * Richtet Lazy Loading für Bilder ein
     */
    function setupImageLazyLoading() {
        // Nur einrichten, wenn IntersectionObserver unterstützt wird
        if ('IntersectionObserver' in window) {
            imageObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        const src = img.getAttribute('data-src');
                        const srcset = img.getAttribute('data-srcset');
                        
                        if (src) {
                            // Bildgröße für Metriken speichern (falls verfügbar)
                            const size = parseInt(img.getAttribute('data-size') || '0', 10);
                            metrics.lazyLoadedPayload += size;
                            metrics.lazyLoadedImages++;
                            metrics.savedRequests++;
                            
                            // Bild laden
                            img.src = src;
                            if (srcset) img.srcset = srcset;
                            img.classList.add('loaded');
                            
                            // Entfernen aus Observer, nachdem es geladen wurde
                            observer.unobserve(img);
                        }
                    }
                });
            }, {
                rootMargin: `${config.imageLazyLoadThreshold}px 0px`
            });
            
            // Alle lazy-load Bilder erfassen
            document.querySelectorAll('img[data-src]').forEach(img => {
                imageObserver.observe(img);
                // Ursprüngliche Payload für Metriken erfassen
                const size = parseInt(img.getAttribute('data-size') || '0', 10);
                metrics.originalPayload += size;
            });
        } else {
            // Fallback für Browser ohne IntersectionObserver
            loadAllImages();
        }
    }
    
    /**
     * Richtet Lazy Loading für UI-Komponenten ein
     */
    function setupComponentLazyLoading() {
        if ('IntersectionObserver' in window) {
            componentObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const element = entry.target;
                        const componentId = element.getAttribute('data-component-id');
                        
                        if (componentId && uiComponents.has(componentId)) {
                            const component = uiComponents.get(componentId);
                            if (!component.loaded) {
                                loadComponent(componentId);
                                observer.unobserve(element);
                            }
                        }
                    }
                });
            }, {
                rootMargin: `${config.componentLazyLoadThreshold}px 0px`
            });
        }
    }
    
    /**
     * Fügt Event-Listener für Scroll-Ereignisse hinzu
     */
    function setupScrollListeners() {
        // Debounce-Funktion für Scroll-Events
        let scrollTimeout;
        
        window.addEventListener('scroll', function() {
            if (scrollTimeout) clearTimeout(scrollTimeout);
            
            scrollTimeout = setTimeout(function() {
                checkVisibleComponents();
            }, config.debounceDelay);
        }, { passive: true });
        
        // Initial prüfen
        checkVisibleComponents();
    }
    
    /**
     * Registriert eine UI-Komponente für Lazy Loading
     * @param {string} componentId - Eindeutige ID der Komponente
     * @param {Function} loadCallback - Funktion, die beim Laden aufgerufen wird
     * @param {HTMLElement|string} triggerElement - Element oder CSS-Selektor, das/der das Laden auslöst
     * @param {number} estimatedSize - Geschätzte Größe der Komponente in KB
     */
    function registerComponent(componentId, loadCallback, triggerElement, estimatedSize = 0) {
        if (typeof triggerElement === 'string') {
            triggerElement = document.querySelector(triggerElement);
        }
        
        if (!triggerElement) {
            console.warn(`Trigger-Element für Komponente ${componentId} nicht gefunden`);
            return false;
        }
        
        uiComponents.set(componentId, {
            id: componentId,
            loadCallback: loadCallback,
            triggerElement: triggerElement,
            loaded: false,
            size: estimatedSize * 1024 // Konvertieren zu Bytes
        });
        
        metrics.originalPayload += estimatedSize * 1024;
        
        // Observer hinzufügen, wenn verfügbar
        if (componentObserver) {
            // data-component-id Attribut setzen für den Observer
            triggerElement.setAttribute('data-component-id', componentId);
            componentObserver.observe(triggerElement);
        }
        
        return true;
    }
    
    /**
     * Lädt eine UI-Komponente
     * @param {string} componentId - ID der zu ladenden Komponente
     */
    function loadComponent(componentId) {
        if (!uiComponents.has(componentId)) {
            console.warn(`Komponente ${componentId} nicht registriert`);
            return false;
        }
        
        const component = uiComponents.get(componentId);
        if (component.loaded) {
            return true; // Bereits geladen
        }
        
        try {
            component.loadCallback();
            component.loaded = true;
            metrics.lazyLoadedPayload += component.size;
            metrics.lazyLoadedComponents++;
            metrics.savedRequests++;
            
            console.log(`Komponente ${componentId} lazy loaded`);
            return true;
        } catch (error) {
            console.error(`Fehler beim Laden der Komponente ${componentId}:`, error);
            return false;
        }
    }
    
    /**
     * Prüft, welche Komponenten sichtbar sind und lädt sie
     */
    function checkVisibleComponents() {
        // Wenn IntersectionObserver verfügbar ist, übernimmt dieser die Arbeit
        if (componentObserver) return;
        
        // Fallback für Browser ohne IntersectionObserver
        uiComponents.forEach((component, componentId) => {
            if (!component.loaded && isElementInViewport(component.triggerElement, config.componentLazyLoadThreshold)) {
                loadComponent(componentId);
            }
        });
    }
    
    /**
     * Prüft, ob ein Element im sichtbaren Bereich ist
     * @param {HTMLElement} el - Zu prüfendes Element
     * @param {number} threshold - Schwellenwert in Pixeln
     * @returns {boolean} Ist das Element sichtbar
     */
    function isElementInViewport(el, threshold = 0) {
        const rect = el.getBoundingClientRect();
        
        return (
            rect.top <= (window.innerHeight + threshold) &&
            rect.bottom >= -threshold &&
            rect.left <= (window.innerWidth + threshold) &&
            rect.right >= -threshold
        );
    }
    
    /**
     * Wandelt alle data-src Bilder in reguläre src Bilder um
     * (Fallback für Browser ohne IntersectionObserver)
     */
    function loadAllImages() {
        document.querySelectorAll('img[data-src]').forEach(img => {
            const src = img.getAttribute('data-src');
            const srcset = img.getAttribute('data-srcset');
            
            if (src) {
                img.src = src;
                if (srcset) img.srcset = srcset;
                img.classList.add('loaded');
            }
        });
    }
    
    /**
     * Gibt eine Zusammenfassung der Lazy-Loading-Metriken zurück
     * @returns {Object} Die Lazy-Loading-Metriken
     */
    function getMetrics() {
        const savedPayload = metrics.originalPayload - metrics.lazyLoadedPayload;
        const savedPayloadPercent = metrics.originalPayload ? 
            ((savedPayload / metrics.originalPayload) * 100).toFixed(1) : 0;
        
        return {
            originalPayload: formatBytes(metrics.originalPayload),
            lazyLoadedPayload: formatBytes(metrics.lazyLoadedPayload),
            savedPayload: formatBytes(savedPayload),
            savedPayloadPercent: savedPayloadPercent + '%',
            lazyLoadedComponents: metrics.lazyLoadedComponents,
            lazyLoadedImages: metrics.lazyLoadedImages,
            savedRequests: metrics.savedRequests
        };
    }
    
    /**
     * Hilfsfunktion zum Formatieren von Bytes
     * @param {number} bytes - Anzahl der Bytes
     * @returns {string} Formatierte Größe
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
        init,
        registerComponent,
        loadComponent,
        getMetrics
    };
})();

// Automatische Initialisierung
document.addEventListener('DOMContentLoaded', VXORLazyLoader.init);
