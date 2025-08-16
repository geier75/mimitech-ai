#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO / VXOR Debug Prompt: MLX JIT nur aktivieren, wenn VX-MATRIX final ist
"""

import os
import sys
import logging
import importlib

# Konfiguration des Loggers
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MISO.debug.mlx_jit_control")

# Füge das Root-Verzeichnis zum Pythonpfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# MLX importieren
try:
    import mlx
    import mlx.core as mx
    HAS_MLX = True
    logger.info("MLX erfolgreich importiert")
except ImportError:
    HAS_MLX = False
    logger.error("MLX konnte nicht importiert werden. Bitte installieren Sie MLX.")
    sys.exit(1)

# Versuche, die Benchmark-Module aus der aktuellen Projektstruktur zu importieren
try:
    # Importiere den Quick Benchmark direkt
    from benchmarks.vision.quick_benchmark import run_quick_benchmark
    logger.info("Quick Benchmark erfolgreich importiert")
except ImportError as e:
    logger.error(f"Benchmark-Import-Fehler: {e}")
    logger.error("Stelle sicher, dass das Benchmark-Modul korrekt installiert ist.")
    sys.exit(1)

def check_vx_matrix_status():
    """
    Prüft den Implementierungsstatus von VX-MATRIX
    """
    # Da wir keine direkte vx_matrix.status haben, simulieren wir die Statusprüfung
    # durch Überprüfung, ob der Ordner vx_matrix_test_env existiert
    vx_matrix_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vx_matrix_test_env')
    if os.path.exists(vx_matrix_path):
        # Hier könnten Sie eine detailliertere Prüfung implementieren
        # Für jetzt nehmen wir an, dass es nicht final ist
        return {"VX-MATRIX": "IN_PROGRESS"}
    else:
        return {"VX-MATRIX": "NOT_FOUND"}

def activate_mlx_jit_if_ready():
    """
    Aktiviert MLX JIT Compilation nur wenn VX-MATRIX vollständig implementiert ist.
    Führt dann einen schnellen Benchmark durch, um die Leistung zu testen.
    """
    try:
        status = check_vx_matrix_status()
        
        logger.info(f"VX-MATRIX Status: {status}")

        if status.get("VX-MATRIX") == "FINAL":
            logger.info("✅ VX-MATRIX vollständig implementiert – MLX JIT wird aktiviert.")
            os.environ["MLX_DISABLE_JIT"] = "0"
            # MLX JIT aktivieren
            try:
                # In der neueren Version könnte es anders sein
                mlx.set_default_device(mx.gpu)
                logger.info("MLX GPU-Modus aktiviert")
            except AttributeError:
                logger.warning("MLX GPU-Konfiguration nicht verfügbar in dieser Version")
        else:
            logger.warning("⚠️ VX-MATRIX nicht final – MLX JIT bleibt deaktiviert.")
            os.environ["MLX_DISABLE_JIT"] = "1"
            # MLX JIT deaktivieren
            try:
                # Einige Versionen von MLX haben möglicherweise andere API
                mlx.set_default_device(mx.cpu)
                logger.info("MLX CPU-Modus aktiviert (JIT deaktiviert)")
            except AttributeError:
                logger.warning("MLX CPU-Konfiguration nicht verfügbar in dieser Version")

        logger.info("🔁 Starte Benchmark mit aktuellem MLX-Status...")
        # Rufe den quick_benchmark mit angepassten Parametern auf
        run_quick_benchmark()
        
    except Exception as e:
        logger.error(f"Fehler bei der Ausführung: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # Ausführung starten
    activate_mlx_jit_if_ready()
