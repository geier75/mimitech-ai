#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO_Ultimate 15.32.28 - OpenTelemetry Tracing Demo
Demonstriert die Integration von MISO mit OpenTelemetry für verteiltes Tracing.
"""

import os
import time
import random
import logging
import sys
from datetime import datetime

# OpenTelemetry Imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO-OTEL-Demo")

# Prüfe, ob OTEL_EXPORTER_OTLP_ENDPOINT gesetzt ist
if "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ:
    logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT nicht gesetzt. Verwende Standard-Endpunkt.")
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"

endpoint = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
logger.info(f"Verwende OTLP-Endpunkt: {endpoint}")


def setup_tracer():
    """Initialisiert den OpenTelemetry Tracer."""
    # Resource für die MISO-Anwendung definieren
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: "MISO_Ultimate",
        ResourceAttributes.SERVICE_VERSION: "15.32.28-rc1",
        "environment": "demo",
        "deployment.id": "rc1-otel-trace-demo"
    })
    
    # Tracer Provider mit Resource initialisieren
    provider = TracerProvider(resource=resource)
    
    # OTLP-Exporter und Span-Prozessor konfigurieren
    otlp_exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
    span_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(span_processor)
    
    # Globalen Tracer Provider setzen
    trace.set_tracer_provider(provider)
    
    return trace.get_tracer("miso.demo")


def simulate_miso_modules(tracer):
    """Simuliert die Verarbeitung zwischen verschiedenen MISO-Modulen."""
    with tracer.start_as_current_span("miso.main_process") as main_span:
        main_span.set_attribute("start_time", datetime.now().isoformat())
        main_span.set_attribute("user", "demo_user")
        
        logger.info("MISO Hauptprozess gestartet")
        
        # Simuliere T-Math-Modul
        with tracer.start_as_current_span("miso.t_math_module") as tmath_span:
            tmath_span.set_attribute("module", "T-MATH")
            logger.info("T-Math-Modul berechnet Tensoren...")
            time.sleep(0.5)  # Simulierte Berechnung
            
            # Tensor-Berechnung tracen
            with tracer.start_as_current_span("t_math.tensor_calculation") as calc_span:
                calc_span.set_attribute("operation", "matrix_multiply")
                calc_span.set_attribute("tensor_shape", "[128, 512, 512]")
                logger.info("Berechne Matrix-Multiplikation...")
                time.sleep(0.8)
                
                # Simuliere einen gelegentlichen Fehler
                if random.random() < 0.2:
                    calc_span.set_status(trace.Status(trace.StatusCode.ERROR))
                    calc_span.record_exception(ValueError("Tensor dimensions mismatch"))
                    logger.error("Fehler bei der Tensor-Berechnung!")
        
        # Simuliere PRISM-Modul (Inference)
        with tracer.start_as_current_span("miso.prism_module") as prism_span:
            prism_span.set_attribute("module", "PRISM")
            logger.info("PRISM-Modul führt Inferenz durch...")
            
            # Modell laden
            with tracer.start_as_current_span("prism.load_model"):
                logger.info("Lade KI-Modell...")
                time.sleep(0.3)
            
            # Inferenz durchführen
            with tracer.start_as_current_span("prism.inference") as infer_span:
                infer_span.set_attribute("model", "vxor_transformer_v3")
                infer_span.set_attribute("batch_size", 16)
                logger.info("Führe Inferenz mit VXOR Transformer V3 durch...")
                time.sleep(1.2)
        
        # Simuliere ECHO-Modul (Ausgabe)
        with tracer.start_as_current_span("miso.echo_module") as echo_span:
            echo_span.set_attribute("module", "ECHO")
            logger.info("ECHO-Modul bereitet Ausgabe vor...")
            
            # Daten formatieren
            with tracer.start_as_current_span("echo.format_output"):
                logger.info("Formatiere Ausgabedaten...")
                time.sleep(0.4)
            
            # Visualisierung erzeugen
            with tracer.start_as_current_span("echo.generate_visualization"):
                logger.info("Erzeuge Visualisierung...")
                time.sleep(0.7)
        
        # Simuliere ZTM-Validator
        with tracer.start_as_current_span("miso.security.ztm_validator") as ztm_span:
            ztm_span.set_attribute("module", "ZTM-Validator")
            ztm_span.set_attribute("security_level", "ULTRA")
            logger.info("ZTM-Validator führt Sicherheitsüberprüfung durch...")
            time.sleep(0.5)
        
        main_span.set_attribute("end_time", datetime.now().isoformat())
        logger.info("MISO Hauptprozess abgeschlossen")


def main():
    """Hauptfunktion für die OpenTelemetry-Demo."""
    # Tracer initialisieren
    tracer = setup_tracer()
    
    try:
        logger.info("=== MISO OpenTelemetry Tracing Demo gestartet ===")
        logger.info(f"Verwende OTLP-Endpunkt: {endpoint}")
        
        # Simulierte MISO-Module ausführen
        simulate_miso_modules(tracer)
        
        logger.info("Demo abgeschlossen. Traces wurden an den Collector gesendet.")
        logger.info("Für weitere Informationen prüfen Sie Ihre Beobachtbarkeitsplattform.")
        
        # Kurze Pause, um sicherzustellen, dass alle Spans übertragen werden
        time.sleep(5)
        
        return 0
    except Exception as e:
        logger.error(f"Fehler in der Demo: {str(e)}")
        return 1
    finally:
        # Shutdown ist wichtig, um sicherzustellen, dass alle Spans exportiert werden
        trace.get_tracer_provider().shutdown()


if __name__ == "__main__":
    sys.exit(main())
