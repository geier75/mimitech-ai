#!/bin/bash
# VXOR AI Sicherheitskompilierung - Shell-Wrapper
# Datum: 2025-04-27

# Farbkonfiguration für die Ausgabe
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ASCII-Art Banner
echo -e "${BLUE}"
echo "██╗   ██╗██╗  ██╗ ██████╗ ██████╗     ███████╗███████╗ ██████╗██╗   ██╗██████╗ ███████╗"
echo "██║   ██║╚██╗██╔╝██╔═══██╗██╔══██╗    ██╔════╝██╔════╝██╔════╝██║   ██║██╔══██╗██╔════╝"
echo "██║   ██║ ╚███╔╝ ██║   ██║██████╔╝    ███████╗█████╗  ██║     ██║   ██║██████╔╝█████╗  "
echo "╚██╗ ██╔╝ ██╔██╗ ██║   ██║██╔══██╗    ╚════██║██╔══╝  ██║     ██║   ██║██╔══██╗██╔══╝  "
echo " ╚████╔╝ ██╔╝ ██╗╚██████╔╝██║  ██║    ███████║███████╗╚██████╗╚██████╔╝██║  ██║███████╗"
echo "  ╚═══╝  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝"
echo -e "${NC}"
echo -e "${GREEN}VXOR AI Sicherheitskompilierung - Phase 2.1${NC}"
echo -e "${YELLOW}Datum: $(date)${NC}\n"

# Prüfe, ob die notwendigen Programme vorhanden sind
echo -e "${BLUE}Prüfe Abhängigkeiten...${NC}"

# Python prüfen
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 ist nicht installiert. Bitte installieren Sie Python 3.${NC}"
    exit 1
else
    python_version=$(python3 --version)
    echo -e "${GREEN}$python_version ist installiert.${NC}"
fi

# Überprüfe Nuitka und Cython
python3 -c "
try:
    import nuitka
    print('\033[0;32mNuitka ist installiert.\033[0m')
except ImportError:
    print('\033[0;31mNuitka ist nicht installiert. Wird installiert...\033[0m')
    import subprocess, sys
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'nuitka'])

try:
    import cython
    print('\033[0;32mCython ist installiert.\033[0m')
except ImportError:
    print('\033[0;31mCython ist nicht installiert. Wird installiert...\033[0m')
    import subprocess, sys
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'cython'])
"

# Umgebungsvariablen setzen
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# Sicherstellen, dass das vxor_secure.py Skript ausführbar ist
chmod +x vxor_secure.py

# Argumente parsen
ARGS=""
NO_BACKUP=false
SKIP_NUITKA=false
SKIP_CYTHON=false

for arg in "$@"; do
    case $arg in
        --no-backup)
            NO_BACKUP=true
            ARGS="$ARGS --no-backup"
            ;;
        --skip-nuitka)
            SKIP_NUITKA=true
            ARGS="$ARGS --skip-nuitka"
            ;;
        --skip-cython)
            SKIP_CYTHON=true
            ARGS="$ARGS --skip-cython"
            ;;
    esac
done

# Warnungen anzeigen
if $NO_BACKUP; then
    echo -e "${YELLOW}WARNUNG: Es wird kein Backup erstellt.${NC}"
fi

if $SKIP_NUITKA; then
    echo -e "${YELLOW}WARNUNG: Nuitka-Kompilierung wird übersprungen.${NC}"
fi

if $SKIP_CYTHON; then
    echo -e "${YELLOW}WARNUNG: Cython-Kompilierung wird übersprungen.${NC}"
fi

# Hauptskript ausführen
echo -e "\n${GREEN}Starte VXOR AI Sicherheitskompilierung...${NC}"
python3 vxor_secure.py $ARGS

# Ergebnis prüfen
RESULT=$?
if [ $RESULT -eq 0 ]; then
    echo -e "\n${GREEN}=========================================${NC}"
    echo -e "${GREEN}VXOR AI Sicherheitskompilierung erfolgreich abgeschlossen!${NC}"
    echo -e "${GREEN}Die kompilierten Dateien befinden sich im 'secure_dist'-Verzeichnis.${NC}"
    echo -e "${GREEN}Logs befinden sich im 'secure_logs'-Verzeichnis.${NC}"
    echo -e "${GREEN}=========================================${NC}"
else
    echo -e "\n${RED}=========================================${NC}"
    echo -e "${RED}VXOR AI Sicherheitskompilierung fehlgeschlagen!${NC}"
    echo -e "${RED}Bitte überprüfen Sie die Logs im 'secure_logs'-Verzeichnis.${NC}"
    echo -e "${RED}Exit-Code: $RESULT${NC}"
    echo -e "${RED}=========================================${NC}"
fi

# Fertig
echo -e "\nEndzeit: $(date)"
exit $RESULT
