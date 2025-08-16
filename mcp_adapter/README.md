# 🔧 MISO Ultimate MCP Adapter

**Model Context Protocol (MCP) Adapter für MISO Ultimate**

Vollständige MCP-kompatible Tool-Integration mit Auto-Discovery und erweiterten Introspection-Funktionen.

## 🚀 Features

### Phase 1: Basis-Tool-Integration ✅
- **Echte API-Calls**: Keine Dummy-Daten, direkte FastAPI-Integration
- **BenchmarkMatrixTool**: Matrix-Benchmarks über `/api/benchmarks/matrix/start`
- **BenchmarkQuantumTool**: Quantum-Benchmarks über `/api/benchmarks/quantum/start`
- **BenchmarkAllTool**: Alle Benchmarks über `/api/benchmarks/all/start`

### Phase 2: Auto-Discovery & Introspection ✅
- **Automatische Tool-Erkennung**: Scannt `mcp_adapter.tools` Package
- **Tool-Metadaten-Extraktion**: Vollständige Introspection aller Tools
- **Erweiterte API-Endpunkte**: Discovery, Schema, Introspection
- **Fallback-Registrierung**: Manual registration wenn Auto-Discovery fehlschlägt

## 📡 API-Endpunkte

### Basis-Endpunkte
```bash
GET  /                    # Server-Info
GET  /health             # Health-Check mit Discovery-Stats
POST /mcp                # MCP Tool-Execution
```

### Tool-Management
```bash
GET  /tools                           # Liste aller Tools (basic)
GET  /tools/discovery                 # Vollständige Discovery-Summary
GET  /tools/{tool_name}/introspect    # Detaillierte Tool-Introspection
GET  /tools/{tool_name}/schema        # Parameter-Schema für Tool
POST /tools/rediscover               # Re-Discovery triggern
```

## 🧪 Test-Commands

### 1. Server-Status prüfen
```bash
curl http://127.0.0.1:8001/health
```

### 2. Tool-Discovery-Summary
```bash
curl http://127.0.0.1:8001/tools/discovery
```

### 3. Tool-Introspection
```bash
curl http://127.0.0.1:8001/tools/benchmark_matrix/introspect
```

### 4. Parameter-Schema abrufen
```bash
curl http://127.0.0.1:8001/tools/benchmark_matrix/schema
```

### 5. MCP-Tool ausführen
```bash
curl -X POST http://127.0.0.1:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{"tool": "benchmark_matrix", "params": {"preset": "quick"}}'
```

### 6. Re-Discovery triggern
```bash
curl -X POST http://127.0.0.1:8001/tools/rediscover
```

## 🔍 Auto-Discovery-System

### Tool-Erkennung
Das System erkennt automatisch MCP-kompatible Tools basierend auf:
- **Required Attributes**: `name`, `description`, `parameters`
- **Required Methods**: `run()`
- **Pydantic-Schema**: Parameter-Validierung
- **Async-Support**: Automatische Erkennung von async/await

### Metadaten-Extraktion
Für jedes Tool wird extrahiert:
- **Basis-Info**: Name, Beschreibung, Modul-Pfad
- **Parameter-Schema**: Vollständiges Pydantic-Schema
- **Methoden-Liste**: Alle verfügbaren Methoden
- **Versionierung**: Aus Docstring extrahiert
- **Tags**: Kategorisierung und Filterung

### Introspection-Daten
Detaillierte Runtime-Informationen:
- **Instance-Info**: Typ, ID, Memory-Usage
- **Capabilities**: API-Integration, HTTP-Client, Async-Support
- **Backend-URLs**: Verbundene APIs
- **Cleanup-Support**: Ressourcen-Management

## 🏗️ Architektur

```
mcp_adapter/
├── __init__.py           # Package-Init
├── server.py            # FastAPI MCP-Server
├── discovery.py         # Auto-Discovery-System
└── tools/
    ├── __init__.py      # Tools-Package-Init
    └── benchmarks.py    # Benchmark-Tools
```

## 🔧 Tool-Entwicklung

### MCP-Tool-Interface
```python
class MyTool:
    name = "my_tool"
    description = "Tool description"
    parameters = MyParamsModel  # Pydantic BaseModel
    
    async def run(self, params: dict) -> Dict[str, Any]:
        # Tool-Implementierung
        return {"result": "success"}
```

### Auto-Registration
Tools werden automatisch erkannt wenn sie:
1. Im `mcp_adapter.tools` Package liegen
2. Das MCP-Tool-Interface implementieren
3. Pydantic-Parameter-Schema haben
4. `run()`-Methode implementieren

## 📊 Monitoring

### Discovery-Statistiken
- **Total Discovered**: Anzahl gefundener Tools
- **Total Registered**: Anzahl registrierter Tools
- **By Module**: Verteilung nach Modulen
- **By Author**: Verteilung nach Autoren
- **By Tags**: Verteilung nach Tags

### Health-Check
```json
{
  "status": "healthy",
  "tools_registered": 3,
  "tools_discovered": 3,
  "auto_discovery_enabled": true,
  "discovery_modules": ["mcp_adapter.tools.benchmarks"]
}
```

## 🚀 Nächste Phasen

### Phase 3: Prompt-Templates per OpenAPI-Fusion
- OpenAPI-Schema-Integration
- Automatische Prompt-Generierung
- Template-basierte Tool-Calls

### Phase 4: Streaming-Bridge für WebSocket-Tools
- WebSocket-Tool-Support
- Real-time Streaming
- Live-Status-Updates

## 🔗 Integration

### Backend-Verbindung
- **FastAPI-Backend**: Port 8000
- **MCP-Server**: Port 8001
- **HTTP-Client**: httpx.AsyncClient
- **Real-time**: WebSocket-Support geplant

### Verwendung
```python
# Tool-Registry
from mcp_adapter.server import registry, discovery

# Tool-Ausführung
tool = registry.get_tool("benchmark_matrix")
result = await tool.run({"preset": "quick"})

# Introspection
introspection = discovery.get_tool_introspection("benchmark_matrix")
```
