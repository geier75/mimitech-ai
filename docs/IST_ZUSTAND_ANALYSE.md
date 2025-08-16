# MISO Ultimate – Ist‑Zustand‑Analyse

Stand: 2025‑08‑07

Diese Analyse fasst den aktuellen Systemzustand kompakt zusammen: Projektstruktur, Abhängigkeiten, Einstiegspunkte/Services, verwendete Frameworks, Benchmarks/Ergebnisse, Konfigurationen sowie nächste Schritte.

---

## 1) Projektstand in Kürze
- Modularer Aufbau mit getrennten Komponenten für Backend (FastAPI), Dashboards (Custom/HTTP + Frontend), Benchmarks, Agents und Trainingspipelines.
- Umfangreiche Benchmark‑Suite inkl. Echtzeit‑Systemmonitoring (psutil), Matrix/Tensor‑Tests, VXOR‑Latenz‑Benchmarks sowie klassische ML/LLM‑Benchmarks (z. B. GSM8K, MMLU, HumanEval usw.).
- Apple‑Silicon‑optimierter Stack: MLX, PyTorch (MPS), JAX (teilweise vorhanden), plus umfangreiche Scientific‑Python‑Bibliotheken.
- Startskripte für Backend und Dashboards, teils mit Auto‑Install von Abhängigkeiten und automatischem Öffnen des Dashboards im Browser.

---

## 2) Projektstruktur (Top‑Level – Auszug)
Wichtige Verzeichnisse und Dateien unter `/<repo-root>`:

- Core/Code
  - `miso/` (Hauptmodule und Integrationen, ~280 Einträge)
  - `vxor/` (VXOR‑Core, Benchmarks/Tests, ~2700 Einträge)
  - `engines/` (z. B. echo_prime‑Integration)
  - `math_module/`, `core/`, `utils/`, `tools/`
  - `mcp_adapter/`, `neural/`, `finance/`
- Dashboards & Frontend
  - `start_dashboard.py`, `miso_dashboard_simple.py`, `vxor_dashboard.py`
  - `dashboard-frontend/` (Node/React/Vite‑basierte Assets, inkl. `node_modules/`, README)
  - HTML: `benchmark_dashboard.html` (+ Backups)
- Benchmarks & Ergebnisse
  - `benchmarks/`, `benchmark/`, `vxor-benchmark-suite/`, `vxor_clean/benchmarks/`
  - Ergebnisse/Artefakte: `benchmark_results/`, `matrix_benchmark_results.json`, `mprime_benchmark_results.json`, `robust_performance_results.json`, `stability_test_results_*.{json,md}`, `plots/`
- Training & Modelle
  - `training/`, `trained_models/`, `training_data/`
  - `train_miso.py`, `AGI_TYPE_GENERATOR.py`
- Orchestrierung/Integrationen
  - `system_integrator.py`, `verify_miso_ultimate.py`
- Konfiguration & Doku
  - `requirements*.txt`, `requirements.lock`, `Dockerfile`
  - `docs/` (zentrale Doku, 9+ Einträge), diverse README/Plans im Root
  - `config/`, `resources/` (z. T. Konfig‑/Datenressourcen)
- Sicherheit
  - `security/` (111+ Einträge), `security_layer.py`, `encrypt_vxor.py`, `secure_vxor.sh`, `secure_*` Ordner
- Umgebungen & Logs
  - Virtuelle Umgebungen im Repo: `venv/`, `vxor_env/`, `mcp_backend_env/`, `backend_env/`, `vx_matrix_test_env/`, `venv_tests/`
  - Logs: `logs/`, `VXOR_Logs/`, diverse `*.log` (z. B. `miso_omega.log`)

Hinweis: Es liegen viele weitere thematische Ordner vor (z. B. `agi_missions/`, `agents/`, `examples/`, `optimization/`, `verify/`, `versioning/`, `whisper.cpp/`).

---

## 3) Abhängigkeiten & Umgebungen
- Python‑Requirements (Root): `requirements.txt` (allgemein), plus gezielt: `requirements_backend.txt`, `requirements_dashboard.txt`, `requirements_training.txt`, `vision_requirements.txt`; zusätzlich `requirements.lock` (gepinnt) und `requirements.in`.
- Container: `Dockerfile` vorhanden (Containerisierung möglich).
- Frontend (Dashboard): `dashboard-frontend/` mit React/Vite/ESLint/TypeScript‑Tooling (sichtbar über `node_modules/` und diverse LICENSE/README‑Dateien).
- Hauptbibliotheken (Auszug, kategorisiert):
  - Scientific/ML: numpy, scipy, pandas, matplotlib, seaborn, scikit‑learn, statsmodels
  - Deep Learning/Accelerators: torch (+ torchvision, torchaudio), MLX, coremltools, jax
  - Quantum: pennylane, qiskit
  - Web/API: fastapi, uvicorn, websockets, pydantic, CORS
  - Dashboards/Vis: streamlit, plotly, dash
  - Daten/DB/IO: sqlalchemy, alembic, h5py, pyarrow
  - Parallel/Scale: ray, dask, joblib
  - DevOps/Tests/Qualität: pytest, black, isort, mypy, pylint, wandb, mlflow
  - OS/Monitoring/Automation: psutil, selenium, pyautogui, keyboard, mouse
- Virtuelle Umgebungen werden im Repo mitgeführt (venv/ etc.). Empfehlung: Für saubere Deployments via Docker oder reproduzierbare `venv`/`conda`‑Setups außerhalb des Repos arbeiten.

Umgebungsvariablen (Auszug):
- `MISO_TEST_MODE` (vom Dashboard gelesen; steuert Testmodus)

---

## 4) Einstiegspunkte & laufende Services
- Backend/API
  - `start_backend.py`: prüft/installs Abhängigkeiten (fastapi/uvicorn/psutil/pydantic/websockets), startet den FastAPI‑Server (dev‑reload) und öffnet Dashboard‑URL.
  - `benchmark_backend_server.py`: FastAPI‑App mit CORS, Pydantic‑Modellen, WebSockets, Benchmark‑/Trainingsstatus und Systemmonitoring via psutil.
  - Standard: Uvicorn auf `http://127.0.0.1:8000` mit Swagger‑Docs unter `http://127.0.0.1:8000/docs`.
- Dashboards
  - `start_dashboard.py`: Custom HTTP‑Server, UI/HTML/CSS eingebettet, API‑Endpunkte für Trainingssteuerung/Status; Port `5151`.
  - `vxor_dashboard.py`: Ergebnis‑Dashboard (serviert JSON/PNG aus `benchmark_results/`); ebenfalls Port `5151` (konfliktfrei zu betreiben, wenn getrennt/angepasst).
  - Weitere Startskripte: `start_benchmark_dashboard.py`, `start_miso_dashboard.py`, `miso_dashboard_simple.py`.

Schnellstart (lokal):
```bash
python3 start_backend.py
# dann Browser: http://127.0.0.1:8000/docs und http://127.0.0.1:5151/benchmark_dashboard.html
```

---

## 5) APIs, Modelle & Kommunikation
- FastAPI‑basierte REST‑Endpunkte (z. B. `/`, `/api/status`) und WebSockets für Live‑Updates.
- Pydantic‑Modelle für Trainings‑/Benchmark‑Requests und Systemstatus.
- CORS‑Konfiguration: offen für alle Origins (Entwicklungsmodus).

---

## 6) Benchmarks & aktueller Status
- Benchmark‑Suiten (Auszug):
  - `comprehensive_benchmark_suite.py`, `comprehensive_matrix_benchmark.py`, `t_math_comprehensive_benchmark.py`, `t_mathematics_isolated_benchmark.py`
  - Vision‑Benchmarks: `benchmarks/vision/*`
  - VXOR‑Latenz: `benchmark/vxor_latency_benchmark.py`, `benchmark/vxor_latency_benchmark_fixed.py`
  - VXOR‑/ML‑Benchmarks: `vxor-benchmark-suite/`, `vxor_clean/benchmarks/` (GSM8K, MMLU, HumanEval, HellaSwag, ARC …)
- Ergebnisse/Artefakte (Beispiele):
  - `benchmark_results/` (JSON/PNG), `matrix_benchmark_results.json`, `mprime_benchmark_results.json`, `robust_performance_results.json`
  - Stabilität/Last: `stability_test_results_*.json|.md`, Logs `stability_tests*.log`
- Statusberichte/Dokumente:
  - `VXOR_SYSTEM_STATUS_2025_08_03_FINAL.md`, `vXor_IST_ANALYSE_FULL.md`, diverse README/Reports im Root.

Hinweis: Es bestehen sowohl „simulierte“ als auch reale Benchmarks. Der Backend‑Server integriert Echtzeit‑Systemdaten (psutil) und asynchrone Ausführung inkl. WebSocket‑Broadcasts.

---

## 7) Tests & Qualitätssicherung
- Umfangreiche Testsammlungen (`test_*.py`, `tests/`, `test_dist/`) sowie Sammelskripte:
  - `run_all_tests.py`, `run_comprehensive_tests.py` (+ part2/part3), `run_system_test.py`
- Verifizierungs-/Integrationsskripte: `verify_miso_ultimate.py`, `system_integrator.py`, `ztm_verification_test.py`

---

## 8) Sicherheit & Compliance
- Code & Dokumente unter `security/` (111+ Items), u. a. Policies, Checks, Analyse.
- Sicherheitskomponenten: `security_layer.py`, Verschlüsselung/Hardening (`encrypt_vxor.py`, `secure_vxor.sh`, `secure_*` Ordner), Logs in `secure_logs/`.

---

## 9) Wichtige Dokumente
- Zentrale READMEs: `README.md`, `README_AKTUALISIERT.md`, `README_COMPLETE_VXOR_AGI.md`, `README_EXEC_SUMMARY*.md`, `README_MCP.md`
- Projekt-/Planungsdoku: `DOCUMENTATION*.md`, `IMPLEMENTIERUNGSPLAN_*.md`, `AGI_TRAINING_MASTER_PLAN.md`, `LIVE_DOCUMENTATION_STATUS.md`, `FINAL_SYSTEM_CHECKLIST.md`
- VXOR‑Spezifisches: `VXOR_SYSTEM_OVERVIEW.md`, `VXOR_C4_Masterstruktur_NEU.md`, `VXOR_COMPLETE_BENCHMARK_REPORT.md`, Whitepaper/Model‑Card
- Compliance/Security: `SECURITY_COMPLIANCE*.md`, `VERSCHLUESSELUNG.md`

---

## 10) Empfehlungen & Nächste Schritte
1) Abhängigkeits‑Härtung
   - `requirements*.txt` konsolidieren, Versionen fixieren, optionale Extras definieren (backend/dashboard/training/vision).
   - Build‑Pfad via `Dockerfile` standardisieren; CI‑Konfiguration (optional) ergänzen.
2) Laufzeit‑/Port‑Koordination
   - Port‑Kollisionen zwischen `start_dashboard.py` und `vxor_dashboard.py` vermeiden (beide nutzen 5151). Entweder kombinieren oder Ports trennen/konfigurierbar machen.
3) Benchmark‑Statusmatrix
   - Automatisiert aus `benchmark_results/` und `*_results.json` generieren (Real vs. Simuliert, Datum, Hardware, Metriken). Optional als Seite im Dashboard ausspielen.
4) Observability
   - OpenTelemetry‑Plan umsetzen (vgl. `OTEL_MONITORING_INTEGRATION.md`), strukturierte Logs/Traces/Metrics für Benchmarks & Training.
5) Sicherheits‑Review
   - Audit der `security/`‑Artefakte und produktive Härtung (Secrets, CORS, TLS, Packaging, Supply‑Chain).
6) Repo‑Hygiene
   - Lokale `venv/`‑Ordner aus dem Repo entfernen/ignorieren, reproduzierbare Envs dokumentieren (Makefile/Script/Docs).

---

## 11) Schnellstart‑Cheatsheet
- Backend & Dashboard lokal
  - `python3 start_backend.py`
  - Browser: `http://127.0.0.1:8000/docs`, `http://127.0.0.1:5151/benchmark_dashboard.html`
- Alternative Dashboards
  - `python3 start_dashboard.py` (Port 5151)
  - `python3 vxor_dashboard.py` (Port 5151; ggf. Port anpassen)
- Benchmarks (Beispiele)
  - `python3 comprehensive_benchmark_suite.py`
  - `python3 benchmark/vxor_latency_benchmark.py`
  - `python3 t_math_comprehensive_benchmark.py`

---

## 12) Abhängigkeiten – Appendix

Quellen der Abhängigkeiten (Hauptdateien):
- `requirements.txt` (Root, allgemein)
- `requirements_backend.txt` (Backend/FastAPI)
- `requirements_dashboard.txt` (Dashboards)
- `vision_requirements.txt` (Vision‑Stack)
- `requirements_training.txt` (Training, minimal)
- Weitere bereichsspezifische Files: `vxor-benchmark-suite/requirements.txt`, `vxor/ai/VX-MATRIX/requirements.txt`, `optimization/phase5_profiling/t_mathematics/requirements.txt`, `whisper.cpp/tests/*/requirements.txt`

Exakte Inhalte (Kernauszüge)

```text
# requirements.txt (Root)
# MISO Ultimate - Anforderungen
# Optimiert für MacBook Pro M4 Max mit Apple Silicon
# Aktualisiert: 16.06.2025

# Grundlegende Anforderungen
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
psutil>=7.0.0  # Für Hardware-Erkennung und Performance-Tracking

# PyTorch mit MPS-Unterstützung
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# Apple-spezifische Optimierungen
coremltools>=6.2.0
mlx>=0.0.5  # Apple MLX Framework für Apple Silicon

# Quantum Computing
pennylane>=0.28.0
qiskit>=0.42.0

# Datenverarbeitung und -analyse
scikit-learn>=1.2.0
statsmodels>=0.14.0
opencv-python-headless>=4.7.0  # Für beschleunigte Bildverarbeitung

# Synthetische Datengenerierung
sdv>=1.0.0
faker>=18.0.0

# Deep Learning Frameworks
# TensorFlow wurde aufgrund von Kompatibilitätsproblemen mit Python 3.13 entfernt
# Stattdessen werden MLX und PyTorch als Haupt-Frameworks verwendet
jax>=0.4.10
jaxlib>=0.4.10
opencv-python-headless>=4.7.0  # Für beschleunigte Bildverarbeitung

# Web-Dashboard
streamlit>=1.22.0
plotly>=5.14.0
dash>=2.9.0

# Datenbank und Speicher
sqlalchemy>=2.0.0
alembic>=1.10.0  # Datenbank-Migrations-Tool
h5py>=3.8.0
pyarrow>=12.0.0

# Parallelisierung und Optimierung
joblib>=1.2.0
ray>=2.5.0
dask>=2023.5.0

# Entwicklungstools
pytest>=7.3.0
black>=23.3.0
isort>=5.12.0
mypy>=1.3.0
pylint>=2.17.0
pydantic>=2.0.0  # Datenvalidierung und Einstellungen

# Logging und Monitoring
wandb>=0.15.0
mlflow>=2.3.0

# Spezielle Anforderungen für MISO Ultimate
sympy>=1.12.0  # Symbolische Mathematik
networkx>=3.1.0  # Graphentheorie
gym>=0.26.0  # Reinforcement Learning
optuna>=3.2.0  # Hyperparameter-Optimierung
fastapi>=0.95.0  # API-Entwicklung

# Browser-Automatisierung und System-Integration
selenium>=4.10.0  # Browser-Automation
pyautogui>=0.9.53  # Bildschirmsteuerung
keyboard>=0.13.5  # Tastatursteuerung
mouse>=0.7.1  # Maussteuerung
```

```text
# requirements_backend.txt (Backend)
# MISO Ultimate Backend Dependencies
# ===================================

# FastAPI Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0

# Data Models & Validation
pydantic==2.5.0

# System Monitoring
psutil==5.9.6

# Scientific Computing
numpy==1.24.3
scipy==1.11.4

# Machine Learning Backends (optional)
torch>=2.0.0
# mlx>=0.0.6  # Nur für Apple Silicon

# WebSocket Support
websockets==12.0

# HTTP Client für Tests
httpx==0.25.2

# Async Support
asyncio-mqtt==0.13.0

# Logging & Utilities
python-json-logger==2.0.7
python-multipart==0.0.6

# Development Tools
pytest==7.4.3
pytest-asyncio==0.21.1
```

```text
# requirements_dashboard.txt (Dashboards)
# MISO Ultimate AGI - Dashboard-Abhängigkeiten

# Webframework
flask==2.0.1
flask-socketio==5.1.1
fastapi==0.70.0
uvicorn==0.15.0

# Datenverarbeitung
numpy==1.21.2
pandas==1.3.3
scipy==1.7.1

# Visualisierung
matplotlib==3.4.3
plotly==5.3.1
dash==2.0.0
dash-bootstrap-components==1.0.0

# ML-Frameworks
torch==1.10.0
mlx==1.0.0
scikit-learn==1.0.1

# Systemüberwachung
psutil==5.8.0

# Fortgeschrittene Visualisierung
seaborn==0.11.2
bokeh==2.4.0
altair==4.1.0

# Echtzeit-Kommunikation
websockets==10.0

# Datenspeicherung
h5py==3.5.0

# Fortgeschrittene KI-Metriken
tensorboard==2.7.0
wandb==0.12.7
```

```text
# vision_requirements.txt
# Kernpakete für die Bildverarbeitung
numpy>=1.24.0
opencv-python-headless>=4.7.0
mlx>=0.0.5
torch>=2.0.0
torchvision>=0.15.0
```

```text
# requirements_training.txt (minimal)
numpy
torch
```

Dockerfile – relevante Abschnitte:

```dockerfile
# Gepinnte Installation per Hashes
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

# Architektur-spezifische Zusätze
RUN if [ "$(uname -m)" = "aarch64" ]; then \
        pip install --no-cache-dir mlx; \
    elif [ "$(uname -m)" = "x86_64" ]; then \
        pip install --no-cache-dir torch-cuda; \
    fi
```

Bekannte Versionsdifferenzen/Konflikte (Auszug):
- FastAPI: Backend `0.104.1` vs. Dashboard `0.70.0` (veraltete Dashboard‑Pins)
- Uvicorn: `0.24.0` vs. `0.15.0`
- Websockets: `12.0` vs. `10.0`
- PyTorch: `>=2.0.0` vs. `1.10.0`
- NumPy: `1.24.x`/`>=1.24.0` vs. `1.21.2`
- SciPy: `1.11.4`/`>=1.10.0` vs. `1.7.1`
- Pandas: `>=2.0.0` vs. `1.3.3`
- scikit‑learn: `>=1.2.0` vs. `1.0.1`
- psutil: `5.9.6` vs. `5.8.0`
- h5py: `>=3.8.0` vs. `3.5.0`
- Dash/Plotly/Seaborn: Root neuere Versionen vs. Dashboard ältere Pins
- MLX: Root `>=0.0.5` vs. Dashboard `==1.0.0` (prüfen; MLX ist i. d. R. <1.0 in vielen Setups)
- Duplikat: `opencv-python-headless` ist in `requirements.txt` doppelt aufgeführt

Konsolidierungs‑Empfehlungen:
1) Versionen vereinheitlichen, Referenz auf Backend‑Pins (FastAPI ≥0.104, Uvicorn ≥0.24, Websockets ≥12, NumPy ≥1.24, Torch ≥2.0, h5py ≥3.8, scikit‑learn ≥1.2 usw.).
2) `requirements*.txt` in einen Kern (`requirements.txt`) plus Extras (`[backend]`, `[dashboard]`, `[training]`, `[vision]`) überführen oder klare Teil‑Files, die auf gemeinsame Pins referenzieren.
  3) Doppelungen/Alt‑Pins in `requirements_dashboard.txt` aktualisieren; `opencv-python-headless`‑Doppel in Root entfernen.
  4) Reproduzierbarkeit: Build über `requirements.lock` (Hash‑Pins) standardisieren; CI‑Build gegen beide Architekturen (ARM64/AMD64) testen.
  5) Docker: Arch‑spezifische Installation prüfen (z. B. `torch-cuda` Paketwahl), ggf. offizielle PyTorch‑Räder/Indices verwenden.

---

## 14) Benchmark‑Appendix – Statusmatrix & Pfade

Stand: 2025‑08‑07, 22:32

Diese Appendix fasst den Audit‑Status der Kern‑Benchmarks zusammen (Module, Configs, Datenpfade, vorhandene Resultate, Authentizität) und nennt konkrete Maßnahmen.

### Statusmatrix (Kurzüberblick)

| Suite | Module | Config | Erwartete Datenpfade (relativ zum Repo) | Daten vorhanden | Resultate vorhanden | Authentizität (Design) |
|---|---|---|---|---|---|---|
| GSM8K | `vxor_clean/benchmarks/gsm8k_benchmark.py` | `vxor_clean/config/benchmarks/gsm8k_config.json` | `data/real/gsm8k/gsm8k_problems.jsonl` (Fallback: `.json`) | Nein (nicht gefunden) | Nein | zero_simulation=true, echte Daten vorgesehen |
| SWE‑bench | `vxor_clean/benchmarks/swe_bench_benchmark.py`, `vxor_clean/benchmarks/swe_bench_enterprise.py` | `vxor_clean/config/benchmarks/swe_bench_config.json` | `data/real/swe_bench/swe-bench-lite.jsonl` (Fallback: `swe_bench.jsonl`) | Nein | Nein | zero_simulation=true, Integritätsprüfung im Loader |
| ARC | `vxor_clean/benchmarks/arc_benchmark.py` | `vxor_clean/config/benchmarks/arc_config.json` | `data/real/arc/ARC-Easy.jsonl`, `data/real/arc/ARC-Challenge.jsonl` | Nein | Nein | zero_simulation=true |
| HellaSwag | `vxor_clean/benchmarks/hellaswag_benchmark.py` | `vxor_clean/config/benchmarks/hellaswag_config.json` | `data/real/hellaswag/hellaswag_val.jsonl` | Nein | Nein | zero_simulation=true |
| MMLU | `vxor_clean/benchmarks/mmlu_benchmark.py` | `vxor_clean/config/benchmarks/mmlu_config.json` | `data/real/mmlu/<subject>.json` (57 Fächer) | Nein | Nein | zero_simulation=true |
| HumanEval | `vxor_clean/benchmarks/humaneval_benchmark.py` | `vxor_clean/config/benchmarks/humaneval_config.json` | `data/real/humaneval/HumanEval.jsonl` | Datei nicht im Repo gefunden | Ja – viele `humaneval_*.json(.gz)` unter `vxor_clean/results/` und `vxor_clean/test_results/` | zero_simulation=true; sichere Sandbox‑Ausführung |

Hinweise zur Evidenz:
- Datenverzeichnisse: `data/` ist leer; `datasets/` enthält nur Platzhalter (`_smoke/`). Keine realen Benchmark‑Dateien (`*.jsonl`, `*.json`) im Repo gefunden.
- Resultate: Unter `vxor_clean/results/` und `vxor_clean/test_results/` liegen zahlreiche HumanEval‑Ergebnisdateien (mehrere Zeitstempel). Für GSM8K, SWE‑bench, ARC, HellaSwag, MMLU wurden keine Ergebnisdateien gefunden.
- Weitere Mess‑Artefakte: `benchmark_results/` enthält interne Performance‑/Tensor‑Benchmarks (z. B. `benchmark_results_*.json`, `matmul_benchmark.png`, `svd_benchmark.png`, `t_math_benchmark_*.json`) – diese sind kein Ersatz für die oben genannten LLM‑Benchmarks.

### Geplante/empfohlene Struktur (Daten & Resultate)

Erzeuge standardisierte Verzeichnisse und lege Originaldatensätze dort ab (Lizenzbedingungen beachten):

```bash
mkdir -p data/real/{gsm8k,swe_bench,arc,hellaswag,mmlu,humaneval}
mkdir -p vxor_clean/results/{gsm8k,swe_bench,arc,hellaswag,mmlu,humaneval}
```

Optionale Verbesserung: Ein globaler `VXOR_DATA_ROOT`‑Pfad (ENV) könnte die Default‑Pfade `data/real/...` überschreiben, ohne die Configs anzupassen.

### Maßnahmen (To‑Do)

1) Datensätze beschaffen und validieren
   - GSM8K (OpenAI), SWE‑bench (Princeton NLP), ARC (AI2), HellaSwag (Rowan et al.), MMLU (Hendrycks et al.), HumanEval (OpenAI)
   - Ablage gemäß obiger Pfade; Checksummen prüfen (Configs sehen SHA256 vor)
2) Resultat‑Ablage standardisieren
   - Pro Suite Unterordner unter `vxor_clean/results/<suite>` verwenden (siehe `export_path` der jeweiligen Config)
3) Eval‑Läufe durchführen und dokumentieren
   - HumanEval ist bereits mehrfach gelaufen (Ergebnisse vorhanden). Für GSM8K, SWE‑bench, ARC, HellaSwag, MMLU Läufe mit echten Daten nachholen.
4) Doku vervollständigen
   - Dataset‑Versionen/Quellen, Prüfsummen, Befehle/Einstiegspunkte (CLI/API) je Suite ergänzen.

---

Diese Datei wird als zentraler Überblick gepflegt. Für Detail‑APIs, Benchmarkspezifika und aktuelle Resultate siehe die verlinkten Module und Ergebnisdateien.
