# PRISM-Engine Integrationstestplan

## Übersicht

Dieser Testplan beschreibt die strukturierte Durchführung der Integrationstests zwischen der PRISM-Engine und den VXOR-Agenten sowie anderen Kernsystemen. Die Tests sind nach Priorität geordnet und gewährleisten, dass alle kritischen Schnittstellen korrekt funktionieren.

## Voraussetzungen

- Python 3.10+
- Pytest Framework installiert
- MLX installiert (optional, für beschleunigte Tests)
- Zugriff auf alle VXOR.AI-Module
- Umgebungsvariablen korrekt konfiguriert:
  - `MISO_ZTM_MODE`
  - `VOID_SECRET_KEY_BYTES` (falls Sicherheitstests aktiviert sind)

## Testmatrix

| ID | Testbereich | Priorität | Abhängigkeiten | Verantwortlich |
|----|-------------|-----------|----------------|----------------|
| IT-1 | PRISM ↔ T-Mathematics | Hoch | MLX, PyTorch | MATH-Team |
| IT-2 | PRISM ↔ ECHO-PRIME | Hoch | Timeline-Module | TIMELINE-Team |
| IT-3 | PRISM ↔ VX-CHRONOS | Mittel | VXOR.AI, Bridge-Module | VXOR-Team |
| IT-4 | PRISM ↔ VX-GESTALT | Mittel | VXOR.AI | VXOR-Team |
| IT-5 | End-to-End Tests | Niedrig | Alle obigen | QA-Team |
| IT-6 | Performance-Benchmarks | Niedrig | Alle obigen | PERF-Team |

## Testausführung

### 1. Umgebungsvorbereitung

```bash
# Umgebungsvariablen setzen
export MISO_ZTM_MODE=0
export MISO_ZTM_LOG_LEVEL=DEBUG

# Test-Abhängigkeiten installieren
pip install pytest pytest-cov pytest-benchmark
```

### 2. Testausführung nach Modulen

```bash
# T-Mathematics Integration (IT-1)
pytest -xvs tests/integration/test_prism_comprehensive.py::TestPrismTMathIntegration

# ECHO-PRIME Integration (IT-2)
pytest -xvs tests/integration/test_prism_comprehensive.py::TestPrismEchoPrimeIntegration

# VX-CHRONOS Integration (IT-3)
pytest -xvs tests/integration/test_prism_vxor_integration.py::TestPrismChronosIntegration

# VX-GESTALT Integration (IT-4)
pytest -xvs tests/integration/test_prism_vxor_integration.py::TestPrismGestaltIntegration

# Multi-Agent Integration
pytest -xvs tests/integration/test_prism_vxor_integration.py::TestPrismMultiVXORIntegration

# Vollständige End-to-End Tests (IT-5)
pytest -xvs tests/integration/test_prism_comprehensive.py::TestEndToEndIntegration

# Performance-Benchmarks (IT-6)
pytest -xvs tests/integration/test_prism_comprehensive.py::TestPrismPerformanceBenchmarks
```

### 3. Testberichtgenerierung

```bash
# Vollständigen Testdurchlauf mit Berichtgenerierung
pytest -xvs tests/integration/test_prism_*.py --cov=miso.simulation --cov=miso.vxor --cov-report=html
```

## Kritische Testfälle

### PRISM ↔ T-Mathematics
- **Matrix-Batch-Operationen**: Überprüfung der korrekten Tensorverarbeitung mit MLX
- **SVD-Berechnung**: Test der numerischen Stabilität und Präzision
- **Caching-Mechanismen**: Validierung der Performanceoptimierungen
- **Fallback-Mechanismen**: Überprüfung der Fehlertoleranz bei Hardware-Inkompatibilität

### PRISM ↔ ECHO-PRIME
- **Timeline-Synchronisierung**: Überprüfung des bidirektionalen Datenaustauschs
- **Wahrscheinlichkeitsberechnung**: Test der Genauigkeit bei der Wahrscheinlichkeitsmodellierung
- **Paradoxerkennung**: Validierung der korrekten Identifizierung temporaler Paradoxien
- **Zeitlinienverzweigung**: Überprüfung der korrekten Verzweigungslogik

### PRISM ↔ VX-CHRONOS
- **Bridge-Initialisierung**: Validierung der korrekten Initialisierung der ChronosEchoBridge
- **Timeline-Mapping**: Überprüfung der korrekten Zuordnung von Timeline-IDs zwischen Systemen
- **Optimierungsanwendung**: Test der Anwendung von CHRONOS-Optimierungen auf Zeitlinien
- **Ereignisverarbeitung**: Überprüfung der asynchronen Ereignisverarbeitung

### PRISM ↔ VX-GESTALT
- **Modul-Integration**: Validierung der korrekten Integration mit GestaltIntegrator
- **Feedback-Schleife**: Test der Rückkopplungsmechanismen für emergente Zustände
- **Wahrscheinlichkeitsanalyse**: Überprüfung der Wahrscheinlichkeitsberechnungen

### End-to-End Tests
- **Komplexe Reality-Forking**: Validierung des Gesamtsystems mit realistischen Szenarien
- **Vergleich von Paradoxerkennungen**: Test der Konsistenz zwischen verschiedenen Modulen

## Vorgehen bei Testfehlern

1. **Identifizierung**: Zuordnung des Fehlers zu einer spezifischen Schnittstelle oder einem Modul
2. **Isolierung**: Erstellung eines minimalen Reproduzierbarkeitstests
3. **Analyse**: Debugging der Fehlerursache (Datenfluss, Parameter, Zustandsübergang)
4. **Korrektur**: Implementierung der Fehlerbehebung
5. **Verifikation**: Wiederholung des Tests und Regression-Tests verwandter Funktionen

## Abnahmekriterien

- Alle kritischen Testfälle (IT-1, IT-2) müssen erfolgreich sein
- End-to-End Tests müssen grundlegende Funktionalität nachweisen
- Performance-Benchmarks sollten keine signifikante Verschlechterung zeigen
- Testabdeckung sollte mindestens 80% für Kernsysteme betragen

## Hinweise zur Hardware-Beschleunigung

Die Tests mit MLX-Optimierung werden automatisch übersprungen, wenn die Ausführung auf nicht-Apple-Silicon Hardware erfolgt oder MLX nicht installiert ist. In diesem Fall werden Standard-Fallback-Implementierungen genutzt.

## Temporäre Fixes für bekannte Probleme

1. **VX-CHRONOS Timeline-Synchronisierung**: Bei Timeout-Fehlern erhöhen Sie den Timeout-Wert in der ChronosEchoBridge von 5.0 auf 15.0 Sekunden.
2. **MLX-Speichernutzung**: Bei Out-of-Memory-Fehlern reduzieren Sie die Batchgröße in der PrismSimulationEngine.
