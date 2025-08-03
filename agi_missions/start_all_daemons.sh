#!/bin/bash
# Master-Script zum Starten aller VXOR AGI-System Daemons
# Phase "Make It Work & Tighten" - Operationalisierung

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/daemon_logs"
PID_DIR="$SCRIPT_DIR/daemon_pids"

# Erstelle Verzeichnisse
mkdir -p "$LOG_DIR" "$PID_DIR"

echo "🚀 VXOR AGI-SYSTEM DAEMON STARTUP"
echo "=================================="
echo "Timestamp: $(date)"
echo "Script Dir: $SCRIPT_DIR"
echo ""

# Funktion: Daemon-Status prüfen
check_daemon_status() {
    local daemon_name=$1
    local pid_file="$PID_DIR/${daemon_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "✅ $daemon_name läuft (PID: $pid)"
            return 0
        else
            echo "❌ $daemon_name PID-Datei existiert, aber Prozess läuft nicht"
            rm -f "$pid_file"
            return 1
        fi
    else
        echo "⚪ $daemon_name läuft nicht"
        return 1
    fi
}

# Funktion: Daemon starten
start_daemon() {
    local daemon_name=$1
    local daemon_script=$2
    local daemon_args=$3
    local pid_file="$PID_DIR/${daemon_name}.pid"
    local log_file="$LOG_DIR/${daemon_name}.log"
    
    echo "🔄 Starte $daemon_name..."
    
    # Prüfe ob bereits läuft
    if check_daemon_status "$daemon_name" > /dev/null; then
        echo "⚠️ $daemon_name läuft bereits"
        return 0
    fi
    
    # Starte Daemon
    cd "$SCRIPT_DIR"
    nohup python3 "$daemon_script" $daemon_args > "$log_file" 2>&1 &
    local pid=$!
    
    # Speichere PID
    echo $pid > "$pid_file"
    
    # Kurz warten und Status prüfen
    sleep 2
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "✅ $daemon_name gestartet (PID: $pid)"
        return 0
    else
        echo "❌ $daemon_name Start fehlgeschlagen"
        rm -f "$pid_file"
        return 1
    fi
}

# Funktion: Daemon stoppen
stop_daemon() {
    local daemon_name=$1
    local pid_file="$PID_DIR/${daemon_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        echo "🛑 Stoppe $daemon_name (PID: $pid)..."
        
        kill "$pid" 2>/dev/null
        sleep 2
        
        # Force kill falls nötig
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "⚠️ Force kill $daemon_name..."
            kill -9 "$pid" 2>/dev/null
        fi
        
        rm -f "$pid_file"
        echo "✅ $daemon_name gestoppt"
    else
        echo "⚪ $daemon_name läuft nicht"
    fi
}

# Funktion: Alle Daemon-Status anzeigen
show_status() {
    echo "📊 DAEMON STATUS OVERVIEW"
    echo "========================"
    check_daemon_status "smoke_test_daemon"
    check_daemon_status "production_monitor"
    check_daemon_status "fallback_policy"
    echo ""
    
    # Zeige letzte Log-Einträge
    echo "📋 LETZTE LOG-EINTRÄGE:"
    echo "----------------------"
    
    for daemon in smoke_test_daemon production_monitor fallback_policy; do
        local log_file="$LOG_DIR/${daemon}.log"
        if [ -f "$log_file" ]; then
            echo "🔍 $daemon (letzte 3 Zeilen):"
            tail -n 3 "$log_file" | sed 's/^/  /'
            echo ""
        fi
    done
}

# Funktion: Alle Daemons starten
start_all() {
    echo "🚀 Starte alle VXOR AGI-System Daemons..."
    echo ""
    
    # 1. Smoke-Test-Daemon (15 Min Intervall)
    start_daemon "smoke_test_daemon" "smoke_test_daemon.py" "daemon 15"
    
    # 2. Production Live Monitor (30 Min Intervall)  
    start_daemon "production_monitor" "production_live_monitor.py" ""
    
    # 3. Fallback Policy Monitor (kontinuierlich)
    start_daemon "fallback_policy" "vx_control_fallback_policy.py" ""
    
    echo ""
    echo "🎯 DAEMON STARTUP ABGESCHLOSSEN"
    echo "==============================="
    show_status
}

# Funktion: Alle Daemons stoppen
stop_all() {
    echo "🛑 Stoppe alle VXOR AGI-System Daemons..."
    echo ""
    
    stop_daemon "smoke_test_daemon"
    stop_daemon "production_monitor" 
    stop_daemon "fallback_policy"
    
    echo ""
    echo "🏁 ALLE DAEMONS GESTOPPT"
}

# Funktion: Daemons neustarten
restart_all() {
    echo "🔄 Starte alle Daemons neu..."
    stop_all
    sleep 3
    start_all
}

# Funktion: Health-Check aller Daemons
health_check() {
    echo "🏥 DAEMON HEALTH CHECK"
    echo "====================="
    
    local all_healthy=true
    
    # Prüfe jeden Daemon
    for daemon in smoke_test_daemon production_monitor fallback_policy; do
        local pid_file="$PID_DIR/${daemon}.pid"
        local log_file="$LOG_DIR/${daemon}.log"
        
        echo "🔍 Prüfe $daemon..."
        
        # PID-Check
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "  ✅ Prozess läuft (PID: $pid)"
            else
                echo "  ❌ Prozess läuft nicht (PID-Datei vorhanden)"
                all_healthy=false
            fi
        else
            echo "  ❌ Keine PID-Datei gefunden"
            all_healthy=false
        fi
        
        # Log-Check (letzte 5 Minuten)
        if [ -f "$log_file" ]; then
            local recent_logs=$(find "$log_file" -mmin -5 2>/dev/null)
            if [ -n "$recent_logs" ]; then
                echo "  ✅ Aktuelle Logs vorhanden"
            else
                echo "  ⚠️ Keine aktuellen Logs (>5 Min alt)"
            fi
        else
            echo "  ❌ Keine Log-Datei gefunden"
            all_healthy=false
        fi
        
        echo ""
    done
    
    if [ "$all_healthy" = true ]; then
        echo "🎉 ALLE DAEMONS GESUND"
        return 0
    else
        echo "⚠️ EINIGE DAEMONS HABEN PROBLEME"
        return 1
    fi
}

# Funktion: Test-Run aller Systeme
test_run() {
    echo "🧪 SYSTEM TEST-RUN"
    echo "=================="
    
    # Smoke-Test
    echo "🔍 Führe Smoke-Test aus..."
    python3 "$SCRIPT_DIR/smoke_test_daemon.py" > "$LOG_DIR/test_smoke.log" 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Smoke-Test erfolgreich"
    else
        echo "❌ Smoke-Test fehlgeschlagen"
    fi
    
    # Canary-Test
    echo "🎯 Führe Canary-Test aus..."
    python3 "$SCRIPT_DIR/automated_deploy_canary.py" > "$LOG_DIR/test_canary.log" 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Canary-Test erfolgreich"
    else
        echo "❌ Canary-Test fehlgeschlagen"
    fi
    
    echo ""
    echo "🏁 TEST-RUN ABGESCHLOSSEN"
}

# Hauptlogik
case "$1" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        restart_all
        ;;
    status)
        show_status
        ;;
    health)
        health_check
        ;;
    test)
        test_run
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|health|test}"
        echo ""
        echo "VXOR AGI-System Daemon Manager"
        echo "=============================="
        echo ""
        echo "Commands:"
        echo "  start   - Startet alle Daemons"
        echo "  stop    - Stoppt alle Daemons"
        echo "  restart - Startet alle Daemons neu"
        echo "  status  - Zeigt Status aller Daemons"
        echo "  health  - Führt Health-Check durch"
        echo "  test    - Führt System-Test aus"
        echo ""
        echo "Daemons:"
        echo "  - smoke_test_daemon (15 Min Intervall)"
        echo "  - production_monitor (30 Min Intervall)"
        echo "  - fallback_policy (kontinuierlich)"
        echo ""
        exit 1
        ;;
esac

exit 0
