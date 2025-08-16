# vXor AGI Entry-Point-Tests: CI/CD-Integrationsanleitung

Diese Anleitung beschreibt, wie Sie die vXor Entry-Point-Konformitätstests in bestehende CI/CD-Pipelines integrieren können.

## 1. Überblick

Die Entry-Point-Tests stellen sicher, dass alle vXor-Module die standardisierten Entry-Points (init, boot, configure, setup, activate, start) implementieren und damit die Systeminitialisierung und -konfiguration konsistent bleibt.

## 2. Voraussetzungen

- Python 3.8 oder höher
- Zugriff auf das vXor-Repositorium
- CI/CD-Pipeline (GitHub Actions, Jenkins, GitLab CI, Travis CI, etc.)
- Berechtigungen zum Einrichten neuer Pipeline-Jobs

## 3. Integration in bestehende CI/CD-Pipelines

### 3.1 GitHub Actions

Fügen Sie folgende Workflow-Datei in `.github/workflows/entry_point_tests.yml` ein:

```yaml
name: vXor Entry-Point-Konformitätstests

on:
  push:
    branches: [ main, master, develop ]
  pull_request:
    branches: [ main, master, develop ]
  workflow_dispatch:
  schedule:
    - cron: '0 2 * * *'

jobs:
  entry-point-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Abhängigkeiten installieren
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    
    - name: Entry-Point-Tests ausführen
      run: |
        mkdir -p reports/badges
        python vxor_migration/ci_entry_point_test.py
      env:
        PYTHONPATH: ${{ github.workspace }}
    
    # Weitere Schritte (Berichte archivieren, Badges veröffentlichen, etc.)
```

### 3.2 Jenkins

Erstellen Sie eine neue Pipeline mit folgendem Jenkinsfile:

```groovy
pipeline {
    agent {
        docker {
            image 'python:3.10-slim'
        }
    }
    
    triggers {
        cron('0 3 * * *')
        pullRequest()
    }
    
    stages {
        stage('Setup') {
            steps {
                sh '''
                    python -m pip install --upgrade pip
                    if [ -f requirements.txt ]; then
                        pip install -r requirements.txt
                    fi
                    mkdir -p reports/badges
                '''
            }
        }
        
        stage('Entry-Point-Tests') {
            steps {
                sh '''
                    python vxor_migration/ci_entry_point_test.py
                '''
            }
        }
        
        // Weitere Stufen (Berichte veröffentlichen, etc.)
    }
}
```

### 3.3 GitLab CI

Fügen Sie folgende Konfiguration in `.gitlab-ci.yml` ein:

```yaml
entry-point-tests:
  image: python:3.10-slim
  stage: test
  script:
    - python -m pip install --upgrade pip
    - if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    - mkdir -p reports/badges
    - python vxor_migration/ci_entry_point_test.py
  artifacts:
    paths:
      - reports/
    reports:
      junit: reports/test-results.xml
  only:
    - main
    - master
    - develop
    - merge_requests
  schedule:
    - cron: "0 2 * * *"
```

### 3.4 Travis CI

Fügen Sie folgende Konfiguration in `.travis.yml` ein:

```yaml
language: python
python:
  - "3.10"

cache: pip

install:
  - pip install -r requirements.txt

script:
  - mkdir -p reports/badges
  - python vxor_migration/ci_entry_point_test.py

# Weitere Konfiguration für Deployment, Benachrichtigungen, etc.
```

### 3.5 CircleCI

Fügen Sie folgende Konfiguration in `.circleci/config.yml` ein:

```yaml
version: 2.1
jobs:
  entry-point-tests:
    docker:
      - image: cimg/python:3.10
    steps:
      - checkout
      - run:
          name: Installiere Abhängigkeiten
          command: |
            python -m pip install --upgrade pip
            if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
      - run:
          name: Führe Entry-Point-Tests aus
          command: |
            mkdir -p reports/badges
            python vxor_migration/ci_entry_point_test.py
      # Weitere Schritte (Berichte speichern, etc.)

workflows:
  version: 2
  test-workflow:
    jobs:
      - entry-point-tests
  nightly:
    triggers:
      - schedule:
          cron: "0 2 * * *"
          filters:
            branches:
              only:
                - main
                - master
    jobs:
      - entry-point-tests
```

## 4. Anpassung der Tests

### 4.1 Nur Prioritätsmodule testen

Für schnellere Tests in Pull Requests können Sie den Parameter `--priority-only` verwenden:

```bash
python vxor_migration/ci_entry_point_test.py --priority-only
```

### 4.2 Schwellenwerte anpassen

Die Standardschwellenwerte können in `ci_entry_point_test.py` angepasst werden:

```python
# Konformitätsschwelle ändern (Standardwert: 10%)
if summary.get("conforming_rate", 0) < 20:  # 20% Konformität erforderlich
    logger.warning("Konformitätsrate unter 20% - CI-Test fehlgeschlagen")
    sys.exit(1)
```

### 4.3 Silent Mode

Für eine reduzierte Ausgabe kann der Parameter `--silent` verwendet werden:

```bash
python vxor_migration/ci_entry_point_test.py --silent
```

## 5. Integration der Badges

### 5.1 Badges in README einbinden

Sie können die generierten Badges in Ihre README-Dateien einbinden:

```markdown
[![Erfolgsrate](https://your-server.com/badges/erfolgsrate.svg)](https://your-server.com/reports/entry_point_report.md)
[![Entry-Points](https://your-server.com/badges/entry-points.svg)](https://your-server.com/reports/entry_point_report.md)
[![Konformität](https://your-server.com/badges/konformitaet.svg)](https://your-server.com/reports/entry_point_report.md)
[![vXor Status](https://your-server.com/badges/vxor-status.svg)](https://your-server.com/reports/entry_point_report.md)
```

### 5.2 Badges über GitHub Pages veröffentlichen

Wenn Sie GitHub Pages verwenden, können die Badges automatisch veröffentlicht werden:

```yaml
- name: Badges als GitHub Pages veröffentlichen
  if: success() && github.event_name != 'pull_request'
  uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./reports/badges
    destination_dir: badges
    keep_files: true
```

Die Badges sind dann unter `https://[username].github.io/[repo]/badges/` verfügbar.

### 5.3 Badges in Jenkins veröffentlichen

In Jenkins können Sie die Badges mit dem HTML Publisher Plugin veröffentlichen:

```groovy
publishHTML([
    allowMissing: false,
    alwaysLinkToLastBuild: true,
    keepAll: true,
    reportDir: 'reports/badges',
    reportFiles: '*.svg',
    reportName: 'Entry-Point-Badges'
])
```

## 6. Benachrichtigungen und Integration

### 6.1 Slack-Benachrichtigungen

Integrieren Sie Slack-Benachrichtigungen für fehlgeschlagene Tests:

```yaml
# GitHub Actions
- name: Slack-Benachrichtigung
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    fields: repo,message,author,action,workflow
    text: 'Entry-Point-Tests fehlgeschlagen :x:'
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### 6.2 Pull-Request-Kommentare

Automatische Kommentare in Pull Requests:

```yaml
# GitHub Actions
- name: PR kommentieren
  if: github.event_name == 'pull_request'
  uses: actions/github-script@v5
  with:
    github-token: ${{ secrets.GITHUB_TOKEN }}
    script: |
      const fs = require('fs');
      const report = JSON.parse(fs.readFileSync('reports/ci_report.json', 'utf8'));
      const conformRate = report.summary.conforming_rate;
      
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: `## Entry-Point-Konformitätstest\n\nKonformitätsrate: ${conformRate}%`
      });
```

### 6.3 Status-Badges in Pull Requests

Verwenden Sie Status-Badges als Merge-Block:

```yaml
# GitLab CI
entry-point-status:
  stage: test
  script:
    - python ci_tools/check_entry_point_rate.py
  allow_failure: false
```

## 7. Fehlerbehebung

### 7.1 Test schlägt fehl, obwohl alle Module konform sind

**Problem**: Der Test gibt Exit-Code 1 zurück, obwohl alle Module die Entry-Point-Konvention erfüllen.

**Lösung**: Prüfen Sie die Schwellenwerte in `ci_entry_point_test.py`. Möglicherweise ist der Schwellenwert zu hoch gesetzt.

### 7.2 Probleme mit Pfaden

**Problem**: Der Test kann Module nicht finden oder importieren.

**Lösung**: Setzen Sie die PYTHONPATH-Umgebungsvariable:

```bash
PYTHONPATH=$PWD python vxor_migration/ci_entry_point_test.py
```

### 7.3 Große Repositories

**Problem**: Die Tests dauern zu lange.

**Lösung**: Verwenden Sie `--priority-only`, um nur die wichtigsten Module zu testen, und erhöhen Sie die Timeouts:

```bash
python vxor_migration/ci_entry_point_test.py --priority-only
```

## 8. Best Practices

1. **Laufzeit reduzieren**: Nur Prioritätsmodule in PR-Tests prüfen, vollständige Tests auf dem Hauptbranch ausführen.

2. **Inkrementelle Verbesserung**: Schwellenwert schrittweise erhöhen, um kontinuierliche Verbesserung zu fördern.

3. **Badges sichtbar machen**: Platzieren Sie die Status-Badges prominent in der README.

4. **Reports archivieren**: Bewahren Sie die Berichte für historische Analysen auf.

5. **Tägliche Tests**: Führen Sie täglich vollständige Tests aus, um Regressions zu erkennen.

## 9. Nächste Schritte

1. **Test-Ergebnisse verfolgen**: Implementieren Sie Dashboards zur Visualisierung der Entry-Point-Konformität über Zeit.

2. **Automatisierte PRs**: Erstellen Sie automatisierte Pull Requests zur Behebung nicht-konformer Module.

3. **Erweiterte Analysen**: Ergänzen Sie die Tests um Abdeckungs- und Komplexitätsmetriken.

---

*Dokument-Version: 1.0 - Stand: 2. Mai 2025*
