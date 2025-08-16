/**
 * VXOR Asset Optimizer
 * Hilfsskript für Bild-Optimierung und Kompression
 * Version 1.0.0 (Phase 7.2 - Leistungsoptimierung)
 */

const VXORAssetOptimizer = (function() {
    'use strict';
    
    // Kompressionseinstellungen
    const compressionOptions = {
        // WebP-Einstellungen
        webp: {
            quality: 80,
            lossless: false
        },
        // JPEG-Einstellungen
        jpeg: {
            quality: 85,
            progressive: true
        },
        // PNG-Einstellungen
        png: {
            compressionLevel: 9,
            palette: true
        }
    };
    
    // Assets-Register
    const optimizedAssets = {
        images: {
            // Referenz-Size ist in KB
            // Der Wert 'size' ist in Bytes für genauere Berechnungen
            'quantum-bloch.jpg': { 
                original: '/assets/quantum-bloch.jpg', 
                optimized: '/assets/optimized/quantum-bloch.webp',
                size: 245 * 1024,
                width: 640,
                height: 480,
                type: 'webp',
                lazy: true
            },
            'matrix-operations.png': {
                original: '/assets/matrix-operations.png',
                optimized: '/assets/optimized/matrix-operations.webp',
                size: 187 * 1024,
                width: 800,
                height: 600,
                type: 'webp',
                lazy: true
            },
            'logo.svg': {
                original: '/assets/logo.svg',
                optimized: '/assets/optimized/logo-min.svg',
                size: 32 * 1024,
                width: 200,
                height: 60,
                type: 'svg+xml',
                lazy: false // Logo sollte sofort geladen werden
            },
            'risk-high.png': {
                original: '/assets/icons/risk-high.png',
                optimized: '/assets/optimized/risk-high.webp',
                size: 12 * 1024,
                width: 24,
                height: 24,
                type: 'webp',
                lazy: true
            },
            'risk-medium.png': {
                original: '/assets/icons/risk-medium.png',
                optimized: '/assets/optimized/risk-medium.webp',
                size: 11 * 1024,
                width: 24,
                height: 24,
                type: 'webp',
                lazy: true
            },
            'risk-low.png': {
                original: '/assets/icons/risk-low.png',
                optimized: '/assets/optimized/risk-low.webp',
                size: 10 * 1024,
                width: 24,
                height: 24,
                type: 'webp',
                lazy: true
            },
            'heatmap-bg.jpg': {
                original: '/assets/heatmap-bg.jpg',
                optimized: '/assets/optimized/heatmap-bg.webp',
                size: 320 * 1024,
                width: 1200,
                height: 800,
                type: 'webp',
                lazy: true
            }
        },
        // Zusätzliche Asset-Typen können hier hinzugefügt werden
        videos: {},
        fonts: {}
    };
    
    /**
     * Prüft, ob der Browser WebP unterstützt
     * @returns {Promise<boolean>} WebP-Unterstützung
     */
    async function checkWebPSupport() {
        return new Promise(resolve => {
            const webP = new Image();
            webP.onload = function() {
                const result = (webP.width > 0) && (webP.height > 0);
                resolve(result);
            };
            webP.onerror = function() {
                resolve(false);
            };
            webP.src = 'data:image/webp;base64,UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==';
        });
    }
    
    /**
     * Ersetzt Bilder durch optimierte Versionen mit Lazy Loading
     */
    async function optimizeImages() {
        const webPSupported = await checkWebPSupport();
        const imgElements = document.querySelectorAll('img[data-original]');
        
        imgElements.forEach(img => {
            const originalSrc = img.getAttribute('data-original');
            const assetInfo = findAssetByOriginal(originalSrc);
            
            if (assetInfo) {
                const useWebP = webPSupported && assetInfo.type === 'webp';
                const targetSrc = useWebP ? assetInfo.optimized : assetInfo.original;
                
                // Lazy loading anwenden, falls konfiguriert
                if (assetInfo.lazy) {
                    img.setAttribute('data-src', targetSrc);
                    img.setAttribute('data-size', assetInfo.size.toString());
                    img.classList.add('lazy-image');
                    
                    // Platzhalter erstellen
                    const placeholder = document.createElement('div');
                    placeholder.classList.add('lazy-placeholder');
                    placeholder.style.width = `${assetInfo.width}px`;
                    placeholder.style.height = `${assetInfo.height}px`;
                    
                    // Kleines Vorschaubild (optional)
                    if (assetInfo.thumbnail) {
                        const thumbImg = document.createElement('img');
                        thumbImg.src = assetInfo.thumbnail;
                        thumbImg.classList.add('lazy-blur');
                        placeholder.appendChild(thumbImg);
                    }
                    
                    // Platzhalter einfügen
                    img.parentNode.insertBefore(placeholder, img);
                    img.style.width = `${assetInfo.width}px`;
                    img.style.height = `${assetInfo.height}px`;
                } else {
                    // Sofortiges Laden
                    img.src = targetSrc;
                }
                
                // Größeninformationen hinzufügen
                img.setAttribute('width', assetInfo.width.toString());
                img.setAttribute('height', assetInfo.height.toString());
            }
        });
    }
    
    /**
     * Findet Asset-Informationen anhand des Original-Pfads
     * @param {string} originalPath - Pfad zum Original-Asset
     * @returns {Object|null} Asset-Informationen oder null
     */
    function findAssetByOriginal(originalPath) {
        // Relativen Pfad normalisieren
        if (originalPath.startsWith('/')) {
            originalPath = originalPath.substring(1);
        }
        
        // Durch alle Asset-Typen suchen
        for (const assetType in optimizedAssets) {
            for (const assetKey in optimizedAssets[assetType]) {
                const asset = optimizedAssets[assetType][assetKey];
                let assetPath = asset.original;
                
                // Relativen Pfad normalisieren
                if (assetPath.startsWith('/')) {
                    assetPath = assetPath.substring(1);
                }
                
                if (assetPath === originalPath) {
                    return asset;
                }
            }
        }
        
        return null;
    }
    
    /**
     * HTML-Code für ein optimiertes Bild mit Lazy Loading erstellen
     * @param {string} assetKey - Schlüssel des Assets im Register
     * @param {Object} options - Zusätzliche Optionen (alt, class, id, etc.)
     * @returns {string} HTML-Code für das Bild
     */
    function getOptimizedImageHtml(assetKey, options = {}) {
        const asset = optimizedAssets.images[assetKey];
        
        if (!asset) {
            console.warn(`Asset ${assetKey} nicht gefunden`);
            return '';
        }
        
        const alt = options.alt || '';
        const className = options.class ? ` class="${options.class}"` : '';
        const id = options.id ? ` id="${options.id}"` : '';
        const additionalAttrs = options.attrs || '';
        
        let html = '';
        
        if (asset.lazy) {
            html = `
                <img${id}${className} 
                    data-src="${asset.optimized}" 
                    data-original="${asset.original}"
                    data-size="${asset.size}"
                    alt="${alt}"
                    width="${asset.width}" 
                    height="${asset.height}"
                    ${additionalAttrs}
                />
            `;
        } else {
            html = `
                <img${id}${className} 
                    src="${asset.optimized}" 
                    data-original="${asset.original}"
                    alt="${alt}"
                    width="${asset.width}" 
                    height="${asset.height}"
                    ${additionalAttrs}
                />
            `;
        }
        
        return html.replace(/\s+/g, ' ').trim();
    }
    
    /**
     * Gibt eine Zusammenfassung der optimierten Assets
     * @returns {Object} Zusammenfassung
     */
    function getAssetSummary() {
        let totalOriginalSize = 0;
        let totalOptimizedSize = 0;
        let totalAssets = 0;
        
        // Durch alle Asset-Typen iterieren
        for (const assetType in optimizedAssets) {
            for (const assetKey in optimizedAssets[assetType]) {
                const asset = optimizedAssets[assetType][assetKey];
                totalOriginalSize += asset.size;
                
                // Geschätzte optimierte Größe (wenn nicht explizit angegeben)
                const optimizedSize = asset.optimizedSize || 
                    (asset.type === 'webp' ? asset.size * 0.6 : asset.size * 0.8);
                    
                totalOptimizedSize += optimizedSize;
                totalAssets++;
            }
        }
        
        return {
            totalAssets,
            totalOriginalSize: formatBytes(totalOriginalSize),
            totalOptimizedSize: formatBytes(totalOptimizedSize),
            savingsPercent: ((1 - (totalOptimizedSize / totalOriginalSize)) * 100).toFixed(1) + '%',
            savingsAbsolute: formatBytes(totalOriginalSize - totalOptimizedSize)
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
        optimizeImages,
        getOptimizedImageHtml,
        getAssetSummary,
        assets: optimizedAssets
    };
})();
