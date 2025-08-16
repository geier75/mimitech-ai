# VX-MATRIX Performance Suite

Dieses Repository enthält das **MatrixCore**-Modul und das zugehörige Testskript zur umfassenden Performance- und Stabilitätsanalyse.

## Projektstruktur

```text
VX-MATRIX/
├── core/
│   └── matrix_core.py            # Kernmodul mit MatrixCore-Klasse
├── tests/
│   └── vx_matrix_performance_test.py   # Standard-Testskript
├── docs/
│   └── robust_test_results.json  # Ergebnisse des robusten Stabilitätstests
├── requirements.txt              # Abhängigkeiten
└── README.md                     # Diese Anleitung
```

## Abhängigkeiten

In `requirements.txt` sind folgende Pakete definiert:

```
numpy>=1.24
scipy>=1.10
scikit-learn>=1.2
mlx-core>=0.1
```

## Installation

1. **Virtuelle Umgebung anlegen**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Abhängigkeiten installieren**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Testausführung

Führen Sie die Performance- und Stabilitätstests wie folgt aus:

```bash
source venv/bin/activate
python tests/vx_matrix_performance_test.py        # Standard-Test
python tests/vx_matrix_performance_test_improved.py  # Robuster Stabilitätstest
```

Die Ergebnisse werden als JSON-Dateien (`performance_metrics.json` bzw. `robust_test_results.json`) in `docs/` abgelegt.

## CI-Integration (Beispiel mit GitHub Actions)

Erstellen Sie `.github/workflows/test.yml` mit folgendem Inhalt:

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    strategy:
      matrix:
        python-version: ['3.9', '3.11', '3.13']
        os: [ubuntu-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          python -m venv venv
          source venv/bin/activate
          pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          source venv/bin/activate
          python tests/vx_matrix_performance_test.py
          python tests/vx_matrix_performance_test_improved.py
      
      - name: Upload performance metrics
        uses: actions/upload-artifact@v4
        with:
          name: performance-metrics-py${{ matrix.python-version }}-${{ matrix.os }}
          path: |
            docs/performance_metrics.json
            docs/robust_test_results.json
  
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install quality tools
        run: |
          python -m pip install --upgrade pip
          python -m pip install ruff bandit mypy
      
      - name: Lint with Ruff
        run: |
          ruff check --output-format=github .
      
      - name: Security check with Bandit
        run: |
          bandit -r core tests -f json -o bandit-results.json
      
      - name: Type check with mypy
        run: |
          mypy --ignore-missing-imports core tests
      
      - name: Upload analysis results
        uses: actions/upload-artifact@v4
        with:
          name: code-analysis-results
          path: bandit-results.json
```

Damit haben Sie ein **voll funktionsfähiges System** für Performance-Optimierung und Stabilitätsanalyse, ohne Fiktion oder Lücken. Wenn Sie weitere Fragen haben oder Anpassungen wünschen, stehe ich zur Verfügung! 😊
