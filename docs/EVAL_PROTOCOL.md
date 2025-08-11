# vXor Evaluationsprotokoll

Dieser Leitfaden definiert reproduzierbare Evaluationsläufe für ARC, GLUE und IMO/Mathe.

## Umgebung
- Docker Image: `eval/Dockerfile` (Python 3.11-slim)
- Installation: `eval/requirements.txt`
- Einstieg: `Makefile` Targets `test`, `arc-eval`, `glue-eval`, `imo-eval`, `compare-external`

## Hardware & Software zu protokollieren
- CPU/GPU, RAM, OS, Kernel, Python-Version, Paketversionen
- Falls CI: Runner-Typ

## Seeds & Messung
- Seeds: standardmäßig feste Seeds im jeweiligen Runner
- Zeitmessung: `time.perf_counter_ns()`; Ausgabe ms

## Artefakte
- Ergebnisse als JSON unter `vxor/benchmarks/results/<bench>/<timestamp>.json`
- Optional CSV/MD-Aggregate (können nachgerüstet werden)

## Ausführung
```bash
make test         # Unit-Tests
make arc-eval     # ARC Smoke/Eval
make glue-eval    # GLUE Baseline/Eval
make imo-eval     # SymPy Linear Systeme
```

## Datenschutz/Compliance
- Kein Upload von proprietären Gewichten oder Schlüsseln
- Optionaler API-Vergleich (OpenAI/Google/Anthropic) nur bei gesetzten ENV-Variablen

## Validierungsstrategie
- Isolierte Container-Runs
- Rohdaten und Logs exportieren
- Ergebnisse mit Metriken (Accuracy, F1, Spearman, solved@k, Zeit) berichten
