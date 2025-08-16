# vXor AGI Modul Entry-Point Konvention

## Überblick

Diese Dokumentation definiert die verbindliche Konvention für Entry-Points in allen vXor AGI-Modulen. Entry-Points sind standardisierte Funktionen, die einem externen System erlauben, ein Modul konsistent zu initialisieren, zu konfigurieren und zu aktivieren.

## Standard Entry-Points

Jedes Modul **MUSS** folgende Entry-Points implementieren:

### `init()`

```python
def init():
    """Initialisiert das Modul mit Standardwerten."""
    # Implementierung
    return instance
```

* **Zweck**: Erstellt eine Instanz des Hauptobjekts im Modul oder initialisiert das Modul.
* **Parameter**: Keine
* **Rückgabe**: Die erstellte Instanz oder `True` bei erfolgreicher Initialisierung.
* **Seiteneffekte**: Erzeugt typischerweise eine globale Instanz innerhalb des Moduls.

### `boot()`

```python
def boot():
    """Alias für init() für Kompatibilität mit Boot-Konvention."""
    return init()
```

* **Zweck**: Konsistenz mit historischen MISO-Konventionen.
* **Implementation**: Ruft einfach `init()` auf.

### Optionale Entry-Points

Die folgenden Entry-Points **SOLLTEN** implementiert werden, wenn das Modul konfigurierbar ist oder spezielle Startfunktionen hat:

### `configure(config=None)`

```python
def configure(config=None):
    """Konfiguriert das Modul mit spezifischen Einstellungen."""
    # Implementation
    return instance
```

* **Zweck**: Ermöglicht Konfiguration mit externen Parametern.
* **Parameter**: `config` - Ein Dictionary mit Konfigurationsparametern.
* **Rückgabe**: Die konfigurierte Instanz.

### `start()`

```python
def start():
    """Startet die aktive Funktionalität des Moduls."""
    # Implementation
    return instance
```

* **Zweck**: Startet Hintergrundprozesse oder Dienste.
* **Rückgabe**: Die gestartete Instanz.

### `setup()`

```python
def setup():
    """Richtet das Modul mit Standardwerten ein."""
    return init()
```

* **Zweck**: Konsistenz mit verbreiteten Python-Konventionen.
* **Implementation**: Ruft typischerweise `init()` auf.

### `activate()`

```python
def activate():
    """Aktiviert das Modul mit Standardeinstellungen."""
    # Implementation
    return instance
```

* **Zweck**: Aktiviert spezifische Funktionalität nach Initialisierung.

## Implementierungsrichtlinien

1. **Globale Instanz**: Jedes Modul sollte eine private globale Variable für die Hauptinstanz verwenden:
   ```python
   _instance = None
   
   def init():
       global _instance
       if _instance is None:
           _instance = MyClass()
       return _instance
   ```

2. **Idempotenz**: Alle Entry-Points müssen idempotent sein - wiederholte Aufrufe sollten keine unerwünschten Nebenwirkungen haben.

3. **Standardisierte Logging-Nachrichten**: Verwenden Sie konsistente Logging-Nachrichten:
   ```python
   logger.info(f"{MODULE_NAME} erfolgreich initialisiert")
   ```

4. **Fehlerbehandlung**: Fangen Sie Ausnahmen ab und geben Sie aussagekräftige Fehlermeldungen aus:
   ```python
   try:
       # Implementation
   except Exception as e:
       logger.error(f"Fehler bei Initialisierung: {e}")
       raise
   ```

5. **Parameter-Dokumentation**: Dokumentieren Sie Parameter mit korrektem Docstring-Format:
   ```python
   def configure(config=None):
       """Konfiguriert das Modul.
       
       Args:
           config (dict): Konfigurationsparameter
               - param1 (type): Beschreibung
               - param2 (type): Beschreibung
       """
   ```

## Beispiel-Implementation

```python
# Globale Instanz
_instance = None

def init():
    """Standardisierter Entry-Point: Initialisiert das Modul"""
    global _instance
    if _instance is None:
        _instance = MyClass()
        logger.info("Modul erfolgreich initialisiert")
    return _instance

def configure(config=None):
    """Standardisierter Entry-Point: Konfiguriert das Modul
    
    Args:
        config (dict): Konfigurationsparameter
            - param1 (str): Beschreibung
            - param2 (int): Beschreibung
    """
    global _instance
    if _instance is None:
        _instance = init()
    
    if config:
        # Konfiguriere die Instanz
        if 'param1' in config:
            _instance.param1 = config['param1']
    
    return _instance

def boot():
    """Alias für init() für Kompatibilität mit Boot-Konvention"""
    return init()

def setup():
    """Alias für init() für Kompatibilität mit Setup-Konvention"""
    return init()
```

## Testen von Entry-Points

Bei Modultests sollten die Entry-Points explizit mit standardisierten Tests geprüft werden:

```python
def test_module_entry_points():
    # Test init()
    instance1 = module.init()
    assert instance1 is not None
    
    # Test idempotence
    instance2 = module.init()
    assert instance1 is instance2
    
    # Test boot() alias
    instance3 = module.boot()
    assert instance1 is instance3
```

## Migration bestehender Module

Für bestehende Module:

1. Fügen Sie die standardisierten Entry-Points hinzu
2. Behalten Sie bestehende Initialisierungsmethoden bei, aber leiten Sie sie auf die neuen Entry-Points um
3. Aktualisieren Sie die Dokumentation des Moduls
4. Fügen Sie Entry-Point-Tests zur Testsuite hinzu

---

**Verantwortlich**: vXor Entwicklungsteam  
**Letzte Aktualisierung**: 2025-05-01  
**Status**: Verbindlich
