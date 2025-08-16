#!/bin/bash
# ZTM Service Script
# Description: Start/Stop/Restart the Zero-Trust Monitoring system

# Configuration
SERVICE_NAME="ztm-service"
PYTHON_PATH="/usr/bin/python3"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ztm_launcher.py"
LOG_DIR="/var/log/ztm"
PID_FILE="/var/run/${SERVICE_NAME}.pid"
CONFIG_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/config" && pwd)/ztm_config.yaml"

# Environment variables
export PYTHONPATH="${PYTHONPATH}:$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

get_pid() {
    cat "${PID_FILE}" 2>/dev/null
}

is_running() {
    local pid
    pid=$(get_pid)
    if [ -z "${pid}" ]; then
        return 1
    fi
    ps -p "${pid}" > /dev/null 2>&1
}

start() {
    if is_running; then
        echo "${SERVICE_NAME} is already running (PID: $(get_pid))"
        return 0
    fi
    
    echo "Starting ${SERVICE_NAME}..."
    
    # Create log directory if it doesn't exist
    mkdir -p "${LOG_DIR}"
    chmod 755 "${LOG_DIR}"
    
    # Start the service
    nohup ${PYTHON_PATH} "${SCRIPT_PATH}" --config "${CONFIG_FILE}" >> "${LOG_DIR}/${SERVICE_NAME}.log" 2>&1 & 
    echo $! > "${PID_FILE}"
    
    # Check if started successfully
    sleep 2
    if is_running; then
        echo "${SERVICE_NAME} started successfully (PID: $(get_pid))"
        return 0
    else
        echo "Failed to start ${SERVICE_NAME}"
        return 1
    fi
}

stop() {
    if ! is_running; then
        echo "${SERVICE_NAME} is not running"
        return 0
    fi
    
    echo "Stopping ${SERVICE_NAME}..."
    
    local pid
    pid=$(get_pid)
    
    # Send SIGTERM
    kill -15 "${pid}" >/dev/null 2>&1
    
    # Wait for process to exit
    local timeout=10
    while is_running; do
        if [ "${timeout}" -le 0 ]; then
            # Force kill if not responding
            kill -9 "${pid}" >/dev/null 2>&1
            break
        fi
        sleep 1
        timeout=$((timeout - 1))
    done
    
    rm -f "${PID_FILE}"
    echo "${SERVICE_NAME} stopped"
}

status() {
    if is_running; then
        echo "${SERVICE_NAME} is running (PID: $(get_pid))"
        return 0
    else
        echo "${SERVICE_NAME} is not running"
        return 1
    fi
}

restart() {
    stop
    start
}

reload() {
    if ! is_running; then
        echo "${SERVICE_NAME} is not running"
        return 1
    fi
    
    echo "Reloading ${SERVICE_NAME} configuration..."
    kill -HUP "$(get_pid)" >/dev/null 2>&1
    echo "Configuration reloaded"
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    reload)
        reload
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|reload}"
        exit 1
        ;;
esac

exit 0
