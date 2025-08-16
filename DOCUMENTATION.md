# VXOR Benchmark Dashboard - Dokumentation

## Übersicht

Das VXOR Benchmark Dashboard ist eine modular aufgebaute Webanwendung zur Visualisierung und Analyse von Benchmark-Daten verschiedener VXOR-Komponenten. Die Anwendung unterstützt verschiedene Benchmark-Kategorien wie Matrix-Operationen, Quantum-Berechnungen, mathematische Funktionen und in Zukunft weitere Module wie MLPerf, SWE und Security Benchmarks.

## Inhaltsverzeichnis

1. [Architektur](#architektur)
2. [Kernmodul: VXORUtils](#kernmodul-vxorutils)
3. [Benchmark-Module](#benchmark-module)
   - [VXORMatrix](#vxormatrix)
   - [VXORQuantum](#vxorquantum)
   - [VXORTMath](#vxortmath)
   - [VXORMLPerf](#vxormlperf)
   - [VXORSWEBench](#vxorswebench)
   - [VXORSecurity](#vxorsecurity)
4. [Event-System](#event-system)
5. [Theming-System](#theming-system)
6. [Barrierefreiheit](#barrierefreiheit)
7. [Responsiveness](#responsiveness)
8. [Fehlerbehandlung](#fehlerbehandlung)
9. [Erweiterungsanleitung](#erweiterungsanleitung)
10. [API-Endpunkte](#api-endpunkte)
11. [Tests und Entwicklung](#entwicklung-und-tests)

## Architektur

Das VXOR Dashboard folgt einer modularen Architektur mit einem zentralen Kernmodul und mehreren Benchmark-spezifischen Modulen:

```
VXOR Dashboard
├── Kernmodul (VXORUtils.js)
│   ├── API-Kommunikation
│   ├── Event-System
│   ├── Modul-Management
│   ├── Theming-System
│   ├── Fehlerbehandlung
│   └── Barrierefreiheit
│
├── Core-Benchmark-Module
│   ├── VXORMatrix.js (Matrix-Operationen)
│   ├── VXORQuantum.js (Quantum-Berechnungen)
│   └── VXORTMath.js (T-Mathematics)
│
├── Erweiterte Benchmark-Module
│   ├── VXORMLPerf.js (Machine Learning Performance)
│   ├── VXORSWEBench.js (Software Engineering)
│   └── VXORSecurity.js (Sicherheitsbenchmarks)
│
├── Unterstützende Komponenten
    ├── test-data.js (Mock-API für Tests)
    ├── Responsive CSS Framework
    └── Barrierefreiheits-Komponenten
│
└── HTML/CSS
    ├── benchmark_dashboard.html
    └── css/vxor-dashboard.css
```

Die Module kommunizieren über ein zentrales Event-System, das einen losen Kopplungsgrad und eine einfache Erweiterbarkeit ermöglicht.

## Kernmodul: VXORUtils

**Datei**: `js/VXORUtils.js`

Das Kernmodul stellt zentrale Dienste für alle anderen Module bereit:

### API-Kommunikation

```javascript
// Komponenten-Daten abrufen
VXORUtils.fetchComponentStatus()

// Hardware-Metriken abrufen
VXORUtils.fetchHardwareMetrics()

// Benchmark-Daten abrufen
VXORUtils.fetchBenchmarkData(componentId, category)
```

### Event-System

```javascript
// Event-Typen
VXORUtils.EventTypes.DATA_UPDATED
VXORUtils.EventTypes.THEME_CHANGED
VXORUtils.EventTypes.COMPONENT_CHANGED
VXORUtils.EventTypes.CATEGORY_CHANGED
VXORUtils.EventTypes.MODULE_REGISTERED

// Events abonnieren
VXORUtils.onDataUpdate(category, callback)
VXORUtils.onThemeChange(callback)
VXORUtils.onComponentChange(callback)
VXORUtils.onCategoryChange(callback)

// Events abmelden
VXORUtils.offDataUpdate(category, callback)
VXORUtils.offThemeChange(callback)
// ...
```

### Modul-Management

```javascript
// Modul registrieren
VXORUtils.registerModule(category, moduleObject)

// Modulstatus prüfen
VXORUtils.isModuleRegistered(category)
```

### Theming-System

```javascript
// Theme umschalten
VXORUtils.toggleTheme()

// Aktuelles Theme abfragen
VXORUtils.isDarkMode()
```

### Fehlerbehandlung

```javascript
// Fehler loggen
VXORUtils.logError(message, category, error)

// API-Fehler behandeln
VXORUtils.handleApiError(error, category, containerId)
```

### Daten-Cache

Das Kernmodul verwaltet auch einen Cache für Benchmark-Daten, um wiederholte API-Anfragen zu reduzieren.

## Benchmark-Module

### VXORMatrix

Das VXORMatrix-Modul visualisiert Benchmark-Daten für Matrix-Operationen:

- **Performance-Chart**: Balkendiagramm für verschiedene Matrix-Operationen (Multiplikation, Addition, Inversion)
- **Matrix-Größen-Tabelle**: Tabelle mit Leistungsdaten für verschiedene Matrix-Größen
- **Operation-Selector**: Auswahl verschiedener Matrix-Operationen
- **Datenabruf**: Automatischer Abruf spezifischer Matrix-Benchmark-Daten via API
- **Theme-Support**: Dynamische Anpassung an Light- und Dark-Mode
- **Barrierefreiheit**: Implementierung von ARIA-Attributen und Tastaturnavigation

```javascript
// Modul initialisieren
VXORMatrix.init()

// Matrix-Größe setzen
VXORMatrix.setMatrixSize(size)

// Matrix-Charts aktualisieren
VXORMatrix.updateMatrixCharts(data)
```

### VXORQuantum

Das VXORQuantum-Modul visualisiert Benchmark-Daten für Quantum-Berechnungen:

- **Gates-Chart**: Balkendiagramm für verschiedene Quantum-Gate-Operationen
- **Algorithmen-Chart**: Liniendiagramm für Quantum-Algorithmen
- **Bloch-Sphäre**: Interaktive 3D-Visualisierung von Quantum-Zuständen
- **Simulation-Stats**: Tabelle mit Quantum-Simulationsstatistiken
- **Dynamische UI**: Kontext-sensitive Anzeige von relevanten Informationen
- **Fehlervisualisierung**: Spezielle Visualisierung von Quantum-Fehlern

```javascript
// Modul initialisieren
VXORQuantum.init()

// Quantum-Charts aktualisieren
VXORQuantum.updateQuantumCharts(data)

// Quantum-Metriken aktualisieren
VXORQuantum.updateQuantumMetrics(data)

// Bloch-Sphären-Animation umschalten
VXORQuantum.toggleBlochAnimation()

// Qubit-Zustand setzen
VXORQuantum.setQubitState(theta, phi)
```

### VXORTMath

**Datei**: `js/VXORTMath.js`

Das VXORTMath-Modul visualisiert mathematische Algorithmen und numerische Methoden:

- **Numerische-Methoden-Chart**: Balkendiagramm für numerische Methoden-Performance
- **Optimierungsalgorithmen-Chart**: Vergleich von Konvergenzraten verschiedener Optimierungsalgorithmen
- **Algorithmus-Selektor**: Interaktive Auswahl des zu analysierenden Algorithmus
- **Performance-Metrik-Tabelle**: Detail-Ansicht der Performance-Daten

```javascript
// Modul initialisieren
VXORUtils.onDataUpdate('tmath', function(data) {
    // TMath-Daten verarbeiten
    updateTMathCharts(data);
});
```

### VXORMLPerf

Das VXORMLPerf-Modul visualisiert Machine Learning Performance-Benchmarks:

- **Inferenz-Performance-Chart**: Visualisierung der Inferenz-Geschwindigkeit verschiedener ML-Modelle
- **Training-Performance-Chart**: Vergleich der Training-Geschwindigkeit auf verschiedenen Hardware-Plattformen
- **Modelltyp-Selektor**: Interaktive Auswahl zwischen verschiedenen ML-Modellen (ResNet-50, BERT-Base, MobileNet, YOLOv5, GPT-2)
- **Metrik-Tabelle**: Detaillierte Darstellung von Batch-Größe, Durchsatz, Latenz und Präzision
- **Barrierefreiheit**: Screenreader-optimierte Selektoren und Live-Regions für Status-Updates

```javascript
// Modul initialisieren
VXORUtils.onDataUpdate('mlperf', function(data) {
    // MLPerf-Daten verarbeiten
    updateMLPerfCharts(data);
    updateMetricsTable(data);
});
```

### VXORSWEBench

Das VXORSWEBench-Modul visualisiert Software Engineering Benchmarks:

- **Code-Generierungs-Performance-Chart**: Geschwindigkeit und Qualität der Code-Generierung
- **Bugfixing-Performance-Chart**: Darstellung der Bugfixing-Effizienz
- **Task-Kategorie-Selektor**: Auswahl verschiedener Software-Engineering-Aufgaben
- **Programmiersprachen-Selektor**: Filterung nach Programmiersprachen
- **Metriktabelle**: Übersicht über Leistungsmetriken für verschiedene Aufgabenarten
- **Verbesserte Tastaturnavigation**: Optimierte Tab-Reihenfolge und Fokus-Management

```javascript
// Modul initialisieren
VXORUtils.onDataUpdate('swebench', function(data) {
    // SWEBench-Daten verarbeiten
    updateSWEBenchCharts(data);
});
```

### VXORSecurity

Das VXORSecurity-Modul visualisiert Sicherheits-Benchmarks:

- **Erkennungs-/Präventionsraten-Radar-Chart**: Radar-Chart für verschiedene Sicherheitsaspekte
- **Tool-Effizienz-Horizontal-Bar-Chart**: Horizontales Balkendiagramm für Sicherheitstools
- **Sicherheitsmatrix**: Interaktive Matrix mit farblicher Kategorisierung nach CVSS/CWE
- **Domänen-Selektor**: Auswahl verschiedener Sicherheitsdomänen (Web, Mobile, Cloud, IoT, Netzwerk)
- **Barrierefreiheit**: ARIA-Labels, Screenreader-Beschreibungen und Live-Regions für alle Matrixelemente

```javascript
// Modul initialisieren
VXORUtils.onDataUpdate('security', function(data) {
    // Security-Daten verarbeiten
    updateSecurityCharts(data);
    updateSecurityMatrix(data);
});
```
## Barrierefreiheit

Das VXOR Dashboard ist mit besonderem Fokus auf Barrierefreiheit entwickelt worden, um Benutzern mit unterschiedlichen Bedürfnissen ein optimales Nutzungserlebnis zu bieten:

### ARIA-Unterstützung

- **aria-label**: Beschreibende Labels für alle interaktiven Elemente
- **aria-live regions**: Dynamische Ankündigungen von Status- und Datenänderungen
- **aria-describedby**: Verknüpfung von Elementen mit ausführlichen Beschreibungen
- **role-Attribute**: Semantisch korrekte Rollen für alle UI-Komponenten

### Tastaturnavigation

- **Fokus-Management**: Klarer visueller Fokus-Indikator für alle interaktiven Elemente
- **Tastaturbedienung**: Alle Funktionen können ohne Maus bedient werden
- **Tab-Reihenfolge**: Logische und konsistente Navigationspfade
- **Shortcuts**: Zentrale Funktionen können über Tastenkombinationen aufgerufen werden

### Screen Reader

- **Versteckte Beschreibungen**: sr-only-Klasse für zusätzliche Kontext-Informationen
- **Live-Regions**: Dynamische Updates für bedeutende Änderungen
- **Semantisches Markup**: Strukturierte Information mit korrekter Hierarchie
- **Feedback-Mechanismen**: Bestätigung nach Benutzeraktionen

### Visuelles Design

- **Ausreichender Kontrast**: Alle Text-Elemente erfüllen WCAG AA-Standards
- **Flexibler Text**: Größe und Zeilenabstand anpassbar
- **Farbunabhängige Information**: Mehrere Hinweise neben Farbe
- **Fehlermeldungen**: Deutliche und handlungsrelevante Fehlerinformationen

## Responsiveness

Das Dashboard passt sich dynamisch unterschiedlichen Bildschirmgrößen und Geräten an:

### Responsive Layouttechniken

- **CSS Grid**: Flexible Anordnung der Dashboard-Komponenten
- **Flexbox**: Dynamische Anpassung von UI-Elementen
- **Media Queries**: Größenspezifische Layouts
- **Responsive Images**: Optimierte Grafiken für verschiedene Auflösungen

### Mobile-Optimierung

- **Touch-freundliche Bedienelemente**: Mindestgröße 44×44px für alle Interaktionselemente
- **Reduzierte Layouts**: Fokus auf wesentliche Informationen auf kleinen Bildschirmen
- **Optimierte Grafiken**: Automatische Anpassung von Charts und Diagrammen
- **Vertikale Ausrichtung**: Umstrukturierung von horizontalen zu vertikalen Layouts

## Event-System

Das VXOR Dashboard verwendet ein Event-basiertes Kommunikationssystem für die Interaktion zwischen Modulen. Zentrale Event-Typen sind:

- **DATA_UPDATED**: Wird ausgelöst, wenn neue Benchmark-Daten verfügbar sind
- **THEME_CHANGED**: Wird ausgelöst, wenn das Theme geändert wird
- **COMPONENT_CHANGED**: Wird ausgelöst, wenn eine andere Komponente ausgewählt wird
- **CATEGORY_CHANGED**: Wird ausgelöst, wenn eine andere Benchmark-Kategorie ausgewählt wird
- **MODULE_REGISTERED**: Wird ausgelöst, wenn ein neues Modul registriert wird

Beispiel für die Verwendung des Event-Systems:

```javascript
// Im Modul VXORMatrix
VXORUtils.onDataUpdate('matrix', function(data) {
    // Matrix-Daten verarbeiten
    updateMatrixCharts(data);
});

// Im Modul VXORQuantum
VXORUtils.onThemeChange(function(isDarkMode) {
    // Theme aktualisieren
    updateTheme(isDarkMode);
});
```

## Theming-System

Das Dashboard unterstützt ein dynamisches Theming-System mit Light- und Dark-Mode:

- **CSS-Variablen**: Die Styling-Informationen werden über CSS-Variablen gesteuert, die beim Theme-Wechsel aktualisiert werden
- **Theme-Toggle**: Ein Theme-Toggle-Button erlaubt das manuelle Umschalten zwischen Light- und Dark-Mode
- **System-Präferenz**: Das Dashboard respektiert die Systemeinstellung via `prefers-color-scheme`
- **Persistenz**: Die Theme-Auswahl wird im localStorage gespeichert

## Erweiterungsanleitung

### Neues Benchmark-Modul hinzufügen

1. **Modulstruktur erstellen**:

```javascript
const VXORNewModule = (function() {
    'use strict';
    
    // Private Variablen
    let moduleState = {};
    
    function init() {
        // Initialisierungscode
        
        // Event-Listener registrieren
        VXORUtils.onDataUpdate('new-category', handleData);
        VXORUtils.onThemeChange(updateTheme);
    }
    
    function handleData(data) {
        // Daten verarbeiten und UI aktualisieren
    }
    
    function updateTheme(isDarkMode) {
        // Theme aktualisieren
    }
    
    // Öffentliche API
    return {
        init,
        // Weitere öffentliche Methoden
    };
})();

// Beim Kernmodul registrieren
document.addEventListener('DOMContentLoaded', () => {
    if (typeof VXORUtils !== 'undefined') {
        VXORUtils.registerModule('new-category', VXORNewModule);
    } else {
        console.error('VXORUtils nicht gefunden!');
    }
});
```

2. **HTML-Integration**:

```html
<!-- Im <head> -->
<script src="js/VXORNewModule.js"></script>

<!-- Im <body> -->
<div class="benchmark-section" id="new-category-benchmarks">
    <h2>Neue Benchmark-Kategorie</h2>
    <!-- Chart-Container und weitere UI-Elemente -->
</div>
```

3. **CSS-Integration**:

```css
/* In vxor-dashboard.css */
#new-category-benchmarks {
    /* Spezifische Styles für die neue Kategorie */
}
```

### API erweitern

Um neue Backend-API-Endpunkte zu unterstützen:

1. Fügen Sie eine neue Fetch-Methode zu VXORUtils hinzu
2. Integrieren Sie den neuen Endpunkt in das Event-System
3. Aktualisieren Sie die dokumentierten API-Endpunkte

## API-Endpunkte

Das Dashboard kommuniziert mit folgenden Backend-API-Endpunkten:

- **`/api/component_status`**: Liefert Statusdaten zu verfügbaren VXOR-Komponenten
- **`/api/hardware_metrics`**: Liefert System-Hardwaremetriken (CPU, Speicher, Laufzeit)
- **`/api/benchmark_data`**: Liefert allgemeine Benchmark-Daten für alle Kategorien
- **`/api/benchmark/matrix`**: Liefert spezifische Matrix-Benchmark-Daten
- **`/api/benchmark/quantum`**: Liefert spezifische Quantum-Benchmark-Daten
- **`/api/benchmark/tmath`**: Liefert spezifische Mathematik-Benchmark-Daten
- **`/api/log_error`**: Endpunkt für Client-seitige Fehlerberichte

## Fehlerbehandlung

Das Dashboard implementiert eine robuste Fehlerbehandlung:

- **Client-seitiges Logging**: Fehler werden in der Konsole protokolliert und optional ans Backend gesendet
- **UI-Feedback**: Benutzerfreundliche Fehlermeldungen werden angezeigt
- **Retry-Mechanismus**: Fehlgeschlagene API-Anfragen können wiederholt werden
- **Graceful Degradation**: Bei Fehlern werden Fallback-Inhalte angezeigt
- **Timeout-Handling**: Anfragen, die zu lange dauern, werden abgebrochen

---

## Entwicklung und Tests

### Testdaten und Mock-API

Für Entwicklungs- und Testzwecke enthält das Repository eine `test-data.js`-Datei, die Mock-Daten für alle API-Endpunkte bereitstellt. Diese Datei kann in die HTML-Datei eingebunden werden, um ohne Backend-Server zu entwickeln:

```html
<script src="js/test-data.js"></script>
```

Die Mock-API enthält Testdaten für alle Benchmark-Kategorien:
- Matrix-Operationen mit verschiedenen Größen und Operationstypen
- Quantum-Berechnungen mit Gate-Operationen und Algorithmen
- T-Mathematics mit numerischen Methoden und Optimierungsalgorithmen
- MLPerf mit verschiedenen Machine-Learning-Modelltypen
- SWE-Bench mit Code-Generierungs- und Bugfixing-Metriken
- Security-Benchmarks mit einer detaillierten Sicherheitsmatrix

### Phase 7.1: Integrationstests

Das Dashboard wird in Phase 7.1 einer Reihe strenger Integrationstests unterzogen:

#### 1. Modul-Interaktionstests

Diese Tests validieren die korrekte Kommunikation zwischen den Modulen:

```javascript
// Beispiel eines Modul-Interaktionstests
function testModuleRegistration() {
    // 1. Überprüfen, ob alle Module beim Kernmodul registriert sind
    const registeredModules = VXORUtils.getRegisteredModules();
    const expectedModules = ['matrix', 'quantum', 'tmath', 'mlperf', 'swebench', 'security'];
    
    // 2. Prüfe, ob alle erwarteten Module vorhanden sind
    const allModulesRegistered = expectedModules.every(module => registeredModules.includes(module));
    
    // 3. Verifiziere korrekte Event-Kommunikation
    let dataEventReceived = false;
    VXORUtils.onDataUpdate('test', () => { dataEventReceived = true; });
    VXORUtils.notifyDataUpdate('test', {testData: true});
    
    // 4. Prüfe Ergebnisse
    console.assert(allModulesRegistered, 'Nicht alle erwarteten Module wurden registriert');
    console.assert(dataEventReceived, 'Event-Kommunikation funktioniert nicht korrekt');
}
```

#### 2. Daten-Integritätstests

Diese Tests validieren die korrekte Verarbeitung und Darstellung von Daten:

```javascript
// Beispiel eines Daten-Integritätstests
function testDataIntegrity() {
    // 1. Teste mit unvollständigen Daten
    const partialData = {results: [{metric: 'Matrix-Multiplikation', value: 10}]};
    VXORUtils.notifyDataUpdate('matrix', partialData);
    
    // 2. Teste mit fehlerhaften Daten
    const invalidData = {error: 'Ungültige Daten'};
    VXORUtils.notifyDataUpdate('quantum', invalidData);
    
    // 3. Teste mit NULL-Werten
    const nullData = {results: [{metric: 'Quantum-Gates', value: null}]};
    VXORUtils.notifyDataUpdate('quantum', nullData);
    
    // 4. Prüfe, ob Fallback-Visualisierungen angezeigt werden
    // und keine JavaScript-Fehler auftreten
}
```

#### 3. UI-Konsistenztests

Diese Tests validieren die korrekte Darstellung der Benutzeroberfläche:

```javascript
// Beispiel eines UI-Konsistenztests
function testUIConsistency() {
    // 1. Teste responsive Darstellung in verschiedenen Viewports
    function resizeWindow(width, height) {
        window.innerWidth = width;
        window.innerHeight = height;
        window.dispatchEvent(new Event('resize'));
    }
    
    // 2. Teste verschiedene Auflösungen
    [320, 768, 1024, 1440].forEach(width => {
        resizeWindow(width, 800);
        // Prüfe korrekte Darstellung der Module
    });
    
    // 3. Prüfe korrekte Tab-Navigation und Anzeige
    document.querySelectorAll('.tab-button').forEach(tab => {
        tab.click();
        // Prüfe, ob der entsprechende Inhalt angezeigt wird
    });
}
```

#### 4. Barrierefreiheitstests

Diese Tests validieren die Barrierefreiheit des Dashboards:

```javascript
// Beispiel eines Barrierefreiheitstests
function testAccessibility() {
    // 1. Teste Tastaturnavigation durch alle interaktiven Elemente
    const interactiveElements = document.querySelectorAll('button, select, a, input, [tabindex="0"]');
    interactiveElements.forEach(element => {
        element.focus();
        // Prüfe korrekten Fokus-Indikator
    });
    
    // 2. Prüfe ARIA-Attribute auf allen relevanten Elementen
    document.querySelectorAll('[role], [aria-label], [aria-live]').forEach(element => {
        // Validiere korrekte ARIA-Attribute
    });
    
    // 3. Verifiziere semantische HTML-Struktur
    const hasHeadings = document.querySelectorAll('h1, h2, h3, h4, h5, h6').length > 0;
    console.assert(hasHeadings, 'Keine Überschriften gefunden');
}
```

#### 5. Performance-Tests

Diese Tests validieren die Performance des Dashboards:

```javascript
// Beispiel eines Performance-Tests
function testPerformance() {
    // 1. Messe Ladezeit der Module
    const startTime = performance.now();
    VXORUtils.loadAllModules();
    const loadTime = performance.now() - startTime;
    console.log(`Modul-Ladezeit: ${loadTime}ms`);
    
    // 2. Messe Chart-Rendering-Zeit
    const chartRenderStart = performance.now();
    updateAllCharts();
    const chartRenderTime = performance.now() - chartRenderStart;
    console.log(`Chart-Rendering-Zeit: ${chartRenderTime}ms`);
}
```

#### 6. Edge-Case-Tests

Diese Tests validieren das Verhalten in Grenzsituationen:

```javascript
// Beispiel eines Edge-Case-Tests
function testEdgeCases() {
    // 1. Teste Verhalten bei sehr großen Datensätzen
    const largeDataset = { results: Array(1000).fill().map((_, i) => ({ 
metric: `Test ${i}`, value: i })) };
    VXORUtils.notifyDataUpdate('matrix', largeDataset);
    
    // 2. Teste parallele API-Anfragen
    Promise.all([
        VXORUtils.fetchData('matrix'),
        VXORUtils.fetchData('quantum'),
        VXORUtils.fetchData('tmath'),
        VXORUtils.fetchData('mlperf'),
        VXORUtils.fetchData('swebench'),
        VXORUtils.fetchData('security')
    ]).then(results => {
        console.log('Alle Anfragen erfolgreich');
    }).catch(error => {
        console.error('Fehler bei parallelen Anfragen', error);
    });
}
```

### Automatisierte Testläufe

Zur Ausführung aller Tests wurde ein Test-Runner implementiert, der einen umfassenden Testbericht erstellt:

```javascript
// Test-Runner ausführen
VXORTestRunner.runAllTests().then(report => {
    console.table(report.summary);
    console.log(`Gesamtergebnis: ${report.passed}/${report.total} Tests bestanden`);
});
```

Die Tests können über die Debug-Konsole im Browser ausgeführt werden, oder indem die HTML-Datei mit dem Parameter `?runTests=true` aufgerufen wird.

---

Letzte Aktualisierung: 30. April 2025
