#!/bin/bash
# Production Monitoring Daemon für Transfer Baseline v2.1
# Läuft alle 30 Minuten und überwacht System-Health

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/production_live_monitor.py"
LOG_FILE="$SCRIPT_DIR/monitoring_daemon.log"
PID_FILE="$SCRIPT_DIR/monitoring_daemon.pid"

# Funktion: Monitoring-Zyklus ausführen
run_monitoring_cycle() {
    echo "$(date): Starte Monitoring-Zyklus" >> "$LOG_FILE"
    
    # Führe Health-Check aus
    python3 "$PYTHON_SCRIPT" >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "$(date): Monitoring-Zyklus erfolgreich" >> "$LOG_FILE"
    else
        echo "$(date): FEHLER im Monitoring-Zyklus" >> "$LOG_FILE"
    fi
    
    echo "----------------------------------------" >> "$LOG_FILE"
}

# Funktion: Daemon starten
start_daemon() {
    if [ -f "$PID_FILE" ]; then
        echo "Monitoring-Daemon läuft bereits (PID: $(cat $PID_FILE))"
        exit 1
    fi
    
    echo "Starte Production Monitoring Daemon..."
    echo "Log-Datei: $LOG_FILE"
    echo "PID-Datei: $PID_FILE"
    
    # Daemon-Loop
    (
        echo $$ > "$PID_FILE"
        echo "$(date): Production Monitoring Daemon gestartet" >> "$LOG_FILE"
        
        while true; do
            run_monitoring_cycle
            
            # Warte 30 Minuten (1800 Sekunden)
            sleep 1800
        done
    ) &
    
    echo "Monitoring-Daemon gestartet (PID: $(cat $PID_FILE))"
}

# Funktion: Daemon stoppen
stop_daemon() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Monitoring-Daemon läuft nicht"
        exit 1
    fi
    
    PID=$(cat "$PID_FILE")
    echo "Stoppe Monitoring-Daemon (PID: $PID)..."
    
    kill "$PID"
    rm -f "$PID_FILE"
    
    echo "$(date): Production Monitoring Daemon gestoppt" >> "$LOG_FILE"
    echo "Monitoring-Daemon gestoppt"
}

# Funktion: Status prüfen
check_status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Monitoring-Daemon läuft (PID: $PID)"
            
            # Zeige letzte Log-Einträge
            echo ""
            echo "Letzte Log-Einträge:"
            tail -n 10 "$LOG_FILE"
        else
            echo "PID-Datei existiert, aber Prozess läuft nicht"
            rm -f "$PID_FILE"
        fi
    else
        echo "Monitoring-Daemon läuft nicht"
    fi
}

# Funktion: Einmaliger Test-Run
test_run() {
    echo "Führe Test-Monitoring-Zyklus aus..."
    run_monitoring_cycle
    echo "Test-Zyklus abgeschlossen. Siehe: $LOG_FILE"
}

# Hauptlogik
case "$1" in
    start)
        start_daemon
        ;;
    stop)
        stop_daemon
        ;;
    restart)
        stop_daemon
        sleep 2
        start_daemon
        ;;
    status)
        check_status
        ;;
    test)
        test_run
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|test}"
        echo ""
        echo "  start   - Startet den Monitoring-Daemon (alle 30 Min)"
        echo "  stop    - Stoppt den Monitoring-Daemon"
        echo "  restart - Startet den Daemon neu"
        echo "  status  - Zeigt Status und letzte Logs"
        echo "  test    - Führt einmaligen Test-Zyklus aus"
        exit 1
        ;;
esac
