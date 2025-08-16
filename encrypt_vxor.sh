#!/bin/bash
# VXOR AI Codeverschlüsselung - Automatisiertes Shell-Skript
# Datum: 2025-04-26

# Farben für Ausgabe
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===== VXOR AI Codeverschlüsselung - Phase 2.1 =====${NC}"
echo "Aktuelle Zeit: $(date)"

# Umgebungsvariablen setzen
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONUTF8=1

# Überprüfen, ob die notwendigen Programme installiert sind
echo -e "${YELLOW}Überprüfe Abhängigkeiten...${NC}"

# PyArmor überprüfen
if ! command -v pyarmor &> /dev/null; then
    echo -e "${RED}PyArmor ist nicht installiert. Installation läuft...${NC}"
    pip install pyarmor
else
    echo -e "${GREEN}PyArmor ist installiert.${NC}"
fi

# Nuitka überprüfen
if ! python -c "import nuitka" &> /dev/null; then
    echo -e "${RED}Nuitka ist nicht installiert. Installation läuft...${NC}"
    pip install nuitka
else
    echo -e "${GREEN}Nuitka ist installiert.${NC}"
fi

# Cython überprüfen
if ! python -c "import cython" &> /dev/null; then
    echo -e "${RED}Cython ist nicht installiert. Installation läuft...${NC}"
    pip install cython
else
    echo -e "${GREEN}Cython ist installiert.${NC}"
fi

# Icon für App-Bundle vorbereiten (falls nicht vorhanden)
ICON_DIR="$(pwd)/resources"
if [ ! -d "$ICON_DIR" ]; then
    echo -e "${YELLOW}Erstelle Ressourcen-Verzeichnis...${NC}"
    mkdir -p "$ICON_DIR"
fi

# Erstelle erforderliche Verzeichnisse
mkdir -p logs
mkdir -p dist
mkdir -p backup

# Hauptausführung des Verschlüsselungsskripts
echo -e "${GREEN}Starte Verschlüsselungsprozess...${NC}"
python encrypt_vxor.py

RESULT=$?
if [ $RESULT -eq 0 ]; then
    echo -e "${GREEN}Verschlüsselung erfolgreich abgeschlossen!${NC}"
    
    # Cleanup: Zwischendateien entfernen
    echo -e "${YELLOW}Entferne temporäre Dateien...${NC}"
    find dist -name "*.py" -type f -not -name "pytransform*.py" -exec rm {} \;
    find dist -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null
    
    echo -e "${GREEN}VXOR AI Code wurde erfolgreich kompiliert und verschlüsselt.${NC}"
    echo -e "${GREEN}Die kompilierten Binaries und Bibliotheken befinden sich im 'dist'-Verzeichnis.${NC}"
else
    echo -e "${RED}Verschlüsselungsprozess fehlgeschlagen mit Exit-Code $RESULT${NC}"
    echo "Überprüfen Sie die Log-Dateien im 'logs'-Verzeichnis für Details."
fi

echo "Abschlusszeit: $(date)"
