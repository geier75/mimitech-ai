# 🔥 MISO Ultimate MCP System

[![CI Status](https://github.com/your-repo/miso-ultimate/workflows/🔥%20MISO%20Ultimate%20MCP%20CI/badge.svg)](https://github.com/your-repo/miso-ultimate/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-orange.svg)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Produktions-ready Model Context Protocol (MCP) Server für MISO Ultimate**  
> Vollständige Integration mit FastAPI Backend, Auto-Discovery, OpenAPI-Fusion und WebSocket-Streaming

## 🎯 **Features**

### ✅ **Phase 1: Basis-Tool-Integration**
- **3 MCP-Tools** mit echter FastAPI-Backend-Integration
- **Kein Dummy-Output**: Direkte API-Response-Weiterleitung
- **Pydantic-Validierung**: Vollständige Parameter-Validierung
- **Async-Support**: Echte asynchrone Tool-Ausführung

### ✅ **Phase 2: Auto-Discovery & Introspection**
- **Automatische Tool-Erkennung**: Scannt `mcp_adapter.tools` Package
- **Tool-Metadaten-Extraktion**: Vollständige Introspection aller Tools
- **Erweiterte API-Endpunkte**: Discovery, Introspection, Schema-Export
- **Fallback-Registrierung**: Funktioniert bei Auto-Discovery-Fehlern

### ✅ **Phase 3: OpenAPI-Fusion & Prompt-Templates**
- **OpenAPI-Schema-Integration**: 9 Endpunkte automatisch erkannt
- **Prompt-Template-Generierung**: 9 Templates automatisch erstellt
- **Schema-Export**: `mcp_prompt_templates.json` automatisch generiert
- **Fusion-Endpunkte**: `/openapi/fusion`, `/openapi/templates`, `/openapi/sync`

### ✅ **Phase 4: WebSocket-Streaming-Bridge**
- **WebSocket-Bridge**: Real-time Streaming-System
- **Live-Status-Updates**: Session-Management und Progress-Tracking
- **Streaming-Endpunkte**: `/streaming/status`, `/streaming/sessions`, `/mcp/streaming`
- **Real-time Integration**: Backend-WebSocket-Verbindung vorbereitet

### ✅ **Produktions-Features**
- **Smoke-Test-Suite**: Automatisierte Tests aller 9 MCP-Tools (100% bestanden)
- **Edge-Case-Handling**: Status 404/422/429/413/503 mit konsistentem JSON-Error-Schema
- **CLI-Wrapper**: Rich-formatierte Kommandozeilen-Schnittstelle für Teams
- **CI/CD-Integration**: GitHub Actions mit Security- und Performance-Tests

## 🚀 **Quick Start**

### 1. Installation
```bash
# Repository klonen
git clone https://github.com/your-repo/miso-ultimate.git
cd miso-ultimate

# Virtual Environment erstellen
python3 -m venv mcp_env
source mcp_env/bin/activate

# Dependencies installieren
pip install -r requirements.txt
pip install requests click rich
```

### 2. Services starten
```bash
# FastAPI Backend starten (Terminal 1)
python benchmark_backend_server.py

# MCP Server starten (Terminal 2)
python -m mcp_adapter.server
```

### 3. Smoke-Test ausführen
```bash
# Alle MCP-Tools testen
./run_smoke.py

# Erwartete Ausgabe:
# ✅ get_                      → 200 (0.01s)
# ✅ get_api_status            → 200 (0.01s)
# ✅ benchmark_matrix_start    → 200 (0.00s)
# ...
# 🎉 ALL TESTS PASSED!
```

## 🔧 **CLI Usage**

### Tools auflisten
```bash
python mcp-cli.py --list-tools
```

### Benchmark ausführen
```bash
python mcp-cli.py benchmark_matrix --params '{"preset":"quick"}'
python mcp-cli.py benchmark_quantum --params '{"iterations":50}'
python mcp-cli.py benchmark_all --params '{"preset":"intensive"}'
```

### System-Metriken abrufen
```bash
python mcp-cli.py system_metrics
python mcp-cli.py system_hardware
```

## 📡 **API Endpoints**

### MCP Core
- `POST /mcp` - Tool-Ausführung mit Edge-Case-Handling
- `GET /tools` - Liste aller verfügbaren Tools
- `GET /` - Server-Status und Informationen

### Discovery & Introspection
- `GET /tools/discovery` - Tool-Discovery-Summary
- `GET /tools/{tool_name}/introspect` - Tool-Introspection
- `GET /tools/{tool_name}/schema` - Tool-Parameter-Schema
- `POST /tools/rediscover` - Re-Discovery triggern

### OpenAPI Fusion
- `GET /openapi/fusion` - OpenAPI-Fusion-Summary
- `POST /openapi/sync` - OpenAPI-Schema synchronisieren
- `GET /openapi/templates` - Alle Prompt-Templates
- `GET /openapi/templates/{tool_name}` - Spezifisches Template

### WebSocket Streaming
- `GET /streaming/status` - Streaming-Bridge-Status
- `GET /streaming/sessions` - Aktive Streaming-Sessions
- `POST /streaming/cleanup/{session_id}` - Session aufräumen
- `POST /mcp/streaming` - Tool mit Streaming ausführen

## 🧪 **Testing**

### Automatisierte Tests
```bash
# Smoke-Test-Suite (alle Tools)
./run_smoke.py

# Edge-Case-Tests
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{"tool": "invalid_tool", "params": {}}' # → 404

curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{"tool": "benchmark_matrix", "params": {"matrix_sizes": []}}' # → 422
```

### Performance-Tests
```bash
# Response-Zeit testen
time curl -s http://localhost:8001/tools > /dev/null
# Erwartung: < 100ms
```

## 🔒 **Security Features**

### Rate Limiting
- **60 Requests/Minute** pro Client-IP
- **20 Concurrent Requests** global
- **HTTP 429** bei Überschreitung

### Request Validation
- **10MB Payload-Limit** für Parameter
- **Tool-Name-Validierung** (max. 100 Zeichen)
- **Parameter-Schema-Validierung** via Pydantic

### Error Handling
```json
{
  "error": "Tool not found",
  "detail": "Tool 'invalid_tool' not found. Available tools: [...]",
  "error_code": "TOOL_NOT_FOUND",
  "timestamp": "2025-07-30T15:22:40.123456"
}
```

## 📊 **Monitoring**

### Health Checks
```bash
curl http://localhost:8001/         # Server-Status
curl http://localhost:8001/tools    # Tool-Verfügbarkeit
curl http://localhost:8000/api/status # Backend-Status
```

### Metrics
- **Tool-Ausführungszeiten**: Via CLI `--verbose`
- **Error-Rates**: Via konsistente Error-Codes
- **Streaming-Sessions**: Via `/streaming/sessions`

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │───▶│   MCP Server     │───▶│ FastAPI Backend │
│                 │    │   (Port 8001)    │    │   (Port 8000)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ WebSocket Bridge │
                       │   (Port 8002)    │
                       └──────────────────┘
```

### Components
- **MCP Server**: FastAPI-basierter MCP-Server mit Tool-Registry
- **Auto-Discovery**: Automatische Tool-Erkennung und Registrierung
- **OpenAPI Fusion**: Schema-Integration und Template-Generierung
- **WebSocket Bridge**: Real-time Streaming und Session-Management
- **CLI Wrapper**: Rich-formatierte Kommandozeilen-Schnittstelle

## 🔧 **Development**

### Tool hinzufügen
1. Neue Datei in `mcp_adapter/tools/` erstellen
2. MCP-kompatible Klasse implementieren:
```python
class MyTool:
    name = "my_tool"
    description = "Tool description"
    parameters = MyParamsModel  # Pydantic model
    
    async def run(self, params: dict) -> dict:
        # Tool logic here
        return {"result": "success"}
```
3. Auto-Discovery erkennt Tool automatisch

### Testing
```bash
# Tool-Discovery testen
curl http://localhost:8001/tools/discovery

# Neues Tool testen
python mcp-cli.py my_tool --params '{}'
```

## 📈 **Performance**

### Benchmarks
- **Tool-Ausführung**: < 100ms (ohne Backend-Latenz)
- **Auto-Discovery**: < 50ms für 3 Tools
- **OpenAPI-Sync**: < 200ms für 9 Endpunkte
- **Smoke-Test-Suite**: < 1s für alle 9 Tools

### Scaling
- **Concurrent Requests**: 20 (konfigurierbar)
- **Rate Limiting**: 60/min (konfigurierbar)
- **Memory Usage**: < 100MB (ohne Backend)

## 🤝 **Contributing**

1. Fork das Repository
2. Feature-Branch erstellen: `git checkout -b feature/amazing-feature`
3. Changes committen: `git commit -m 'Add amazing feature'`
4. Branch pushen: `git push origin feature/amazing-feature`
5. Pull Request erstellen

### CI/CD
- **GitHub Actions**: Automatische Tests bei Push/PR
- **Smoke Tests**: Alle MCP-Tools werden getestet
- **Security Scan**: Secrets und Konfiguration prüfen
- **Performance Tests**: Response-Zeit-Validierung

## 📝 **License**

MIT License - siehe [LICENSE](LICENSE) für Details.

## 🆘 **Support**

- **Issues**: [GitHub Issues](https://github.com/your-repo/miso-ultimate/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/miso-ultimate/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/miso-ultimate/wiki)

---

**Made with ❤️ by the MISO Ultimate Team**
