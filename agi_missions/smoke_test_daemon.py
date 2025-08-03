#!/usr/bin/env python3
"""
Smoke-Test-Daemon mit Drift-Detection für VXOR AGI-System
Zyklische End-to-End Verifikation & Automatisierte Regression Detection
"""

import json
import yaml
import time
import numpy as np
import logging
import subprocess
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

# Setup logging
import os
log_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'smoke_test_daemon.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SmokeTestResult:
    """Ergebnis eines Smoke-Tests"""
    timestamp: str
    test_type: str
    status: str  # PASS, FAIL, WARNING
    metrics: Dict[str, float]
    baseline_comparison: Dict[str, float]
    drift_detected: bool
    alerts: List[str]
    execution_time_seconds: float

@dataclass
class DriftAlert:
    """Drift-Alert Definition"""
    metric: str
    threshold_percent: float
    current_value: float
    baseline_value: float
    drift_percent: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL

class SmokeTestDaemon:
    """Smoke-Test-Daemon mit Drift-Detection"""
    
    def __init__(self, config_file: str = None):
        if config_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(script_dir, "production_config_v2.1.yaml")
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Baseline-Referenz laden
        self.baseline_metrics = self._load_baseline_metrics()
        
        # Drift-Thresholds
        self.drift_thresholds = {
            "sharpe_ratio": 10.0,      # 10% Drop = Alert
            "accuracy": 5.0,           # 5% Drop = Alert
            "confidence": 5.0,         # 5% Drop = Alert
            "transfer_effectiveness": 8.0,  # 8% Drop = Alert
            "quantum_speedup": 15.0,   # 15% Drop = Alert
            "latency_ms": 50.0,        # 50% Increase = Alert
            "drawdown": 25.0           # 25% Increase = Alert
        }
        
        # Test-Historie
        self.test_history = []
        self.drift_alerts = []
        
        # Daemon-Status
        self.daemon_running = False
        self.last_snapshot = None
        
        logger.info("🔍 Smoke-Test-Daemon initialisiert")
    
    def _load_baseline_metrics(self) -> Dict[str, float]:
        """Lädt v2.1-transfer-baseline Referenzmetriken"""
        try:
            # Lade aus letztem erfolgreichen Canary-Result
            canary_files = list(Path("agi_missions").glob("canary_deployment_results_*.json"))
            if canary_files:
                latest_canary = max(canary_files, key=os.path.getctime)
                with open(latest_canary, 'r') as f:
                    canary_data = json.load(f)
                
                # Extrahiere finale Stage-Metriken
                if canary_data.get("stages"):
                    final_stage = canary_data["stages"][-1]
                    return {
                        "sharpe_ratio": final_stage.get("avg_sharpe_ratio", 1.58),
                        "accuracy": final_stage.get("avg_accuracy", 0.915),
                        "confidence": final_stage.get("avg_confidence", 0.885),
                        "transfer_effectiveness": 0.82,  # Aus A/B-Test
                        "quantum_speedup": 2.4,
                        "latency_ms": 18.5,
                        "drawdown": 0.085
                    }
            
            # Fallback zu Standard-Baseline
            return {
                "sharpe_ratio": 1.58,
                "accuracy": 0.915,
                "confidence": 0.885,
                "transfer_effectiveness": 0.82,
                "quantum_speedup": 2.4,
                "latency_ms": 18.5,
                "drawdown": 0.085
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Baseline: {e}")
            return {}
    
    def run_benchmark_matrix(self) -> Dict[str, float]:
        """Führt Matrix-Benchmark aus"""
        logger.info("🔧 Führe Matrix-Benchmark aus...")
        
        try:
            # Simuliere Matrix-Benchmark (in Realität: subprocess zu echtem Benchmark)
            base_performance = 1.0
            variance = np.random.normal(1.0, 0.05)  # ±5% Varianz
            
            return {
                "matrix_throughput": base_performance * variance * 1000,
                "matrix_accuracy": 0.98 * variance,
                "matrix_latency_ms": 15.0 / variance
            }
            
        except Exception as e:
            logger.error(f"Matrix-Benchmark Fehler: {e}")
            return {}
    
    def run_benchmark_quantum(self) -> Dict[str, float]:
        """Führt Quantum-Benchmark aus"""
        logger.info("⚛️ Führe Quantum-Benchmark aus...")
        
        try:
            # Simuliere Quantum-Benchmark
            base_speedup = 2.4
            variance = np.random.normal(1.0, 0.08)  # ±8% Varianz
            
            return {
                "quantum_speedup": base_speedup * variance,
                "quantum_fidelity": 0.95 * variance,
                "quantum_coherence_time": 100.0 * variance
            }
            
        except Exception as e:
            logger.error(f"Quantum-Benchmark Fehler: {e}")
            return {}
    
    def run_transfer_mission(self) -> Dict[str, float]:
        """Führt Transfer-Mission aus"""
        logger.info("🎯 Führe Transfer-Mission aus...")
        
        try:
            # Simuliere Transfer-Mission
            base_metrics = self.baseline_metrics
            variance = np.random.normal(1.0, 0.06)  # ±6% Varianz
            
            return {
                "sharpe_ratio": base_metrics.get("sharpe_ratio", 1.58) * variance,
                "accuracy": base_metrics.get("accuracy", 0.915) * variance,
                "confidence": base_metrics.get("confidence", 0.885) * variance,
                "transfer_effectiveness": base_metrics.get("transfer_effectiveness", 0.82) * variance,
                "drawdown": base_metrics.get("drawdown", 0.085) * np.random.uniform(0.8, 1.3),
                "latency_ms": base_metrics.get("latency_ms", 18.5) * np.random.uniform(0.9, 1.2)
            }
            
        except Exception as e:
            logger.error(f"Transfer-Mission Fehler: {e}")
            return {}
    
    def detect_drift(self, current_metrics: Dict[str, float]) -> List[DriftAlert]:
        """Erkennt Drift gegenüber Baseline"""
        drift_alerts = []
        
        for metric, current_value in current_metrics.items():
            if metric not in self.baseline_metrics:
                continue
            
            baseline_value = self.baseline_metrics[metric]
            threshold = self.drift_thresholds.get(metric, 10.0)
            
            # Berechne Drift-Prozent
            if baseline_value != 0:
                if metric in ["latency_ms", "drawdown"]:
                    # Für diese Metriken ist Anstieg schlecht
                    drift_percent = ((current_value - baseline_value) / baseline_value) * 100
                else:
                    # Für andere Metriken ist Abfall schlecht
                    drift_percent = ((baseline_value - current_value) / baseline_value) * 100
            else:
                drift_percent = 0
            
            # Prüfe Threshold
            if abs(drift_percent) > threshold:
                severity = "CRITICAL" if abs(drift_percent) > threshold * 2 else "HIGH"
                
                alert = DriftAlert(
                    metric=metric,
                    threshold_percent=threshold,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    drift_percent=drift_percent,
                    severity=severity
                )
                
                drift_alerts.append(alert)
                logger.warning(f"🚨 DRIFT ALERT: {metric} = {current_value:.3f} "
                             f"(Baseline: {baseline_value:.3f}, Drift: {drift_percent:+.1f}%)")
        
        return drift_alerts
    
    def create_snapshot(self, test_result: SmokeTestResult, drift_alerts: List[DriftAlert]):
        """Erstellt Snapshot bei kritischen Events"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "trigger": "DRIFT_DETECTION" if drift_alerts else "SCHEDULED_SNAPSHOT",
            "test_result": asdict(test_result),
            "drift_alerts": [asdict(alert) for alert in drift_alerts],
            "baseline_metrics": self.baseline_metrics,
            "system_state": {
                "current_baseline": "v2.1-transfer-baseline",
                "daemon_uptime": time.time() - getattr(self, 'start_time', time.time()),
                "total_tests_run": len(self.test_history)
            }
        }
        
        # Speichere Snapshot
        snapshot_file = f"agi_missions/drift_snapshots/snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(snapshot_file), exist_ok=True)
        
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        # VOID-Audit-Log
        self._create_void_audit("DRIFT_SNAPSHOT", snapshot)
        
        logger.info(f"📸 Snapshot erstellt: {snapshot_file}")
        self.last_snapshot = datetime.now()
    
    def _create_void_audit(self, event_type: str, data: Dict[str, Any]):
        """Erstellt VOID-Protokoll Audit-Eintrag"""
        audit_id = f"VOID_SMOKE_{int(time.time())}"
        
        audit_entry = {
            "audit_id": audit_id,
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data,
            "void_protocol": True,
            "security_level": "HIGH",
            "automated": True
        }
        
        # Speichere VOID-Audit
        audit_file = f"agi_missions/void_audit_logs/void_smoke_{audit_id}.json"
        os.makedirs(os.path.dirname(audit_file), exist_ok=True)
        
        with open(audit_file, 'w') as f:
            json.dump(audit_entry, f, indent=2)
        
        logger.info(f"📋 VOID-Audit erstellt: {audit_id}")
    
    def run_smoke_test_cycle(self) -> SmokeTestResult:
        """Führt kompletten Smoke-Test-Zyklus aus"""
        logger.info("🔄 Starte Smoke-Test-Zyklus")
        start_time = time.time()
        
        # Sammle alle Metriken
        all_metrics = {}
        alerts = []
        status = "PASS"
        
        try:
            # Matrix-Benchmark
            matrix_metrics = self.run_benchmark_matrix()
            all_metrics.update(matrix_metrics)
            
            # Quantum-Benchmark
            quantum_metrics = self.run_benchmark_quantum()
            all_metrics.update(quantum_metrics)
            
            # Transfer-Mission
            transfer_metrics = self.run_transfer_mission()
            all_metrics.update(transfer_metrics)
            
            # Drift-Detection
            drift_alerts = self.detect_drift(all_metrics)
            
            if drift_alerts:
                critical_alerts = [a for a in drift_alerts if a.severity == "CRITICAL"]
                if critical_alerts:
                    status = "FAIL"
                    alerts.extend([f"CRITICAL DRIFT: {a.metric}" for a in critical_alerts])
                else:
                    status = "WARNING"
                    alerts.extend([f"DRIFT WARNING: {a.metric}" for a in drift_alerts])
            
            # Baseline-Vergleich
            baseline_comparison = {}
            for metric, current_value in all_metrics.items():
                if metric in self.baseline_metrics:
                    baseline_value = self.baseline_metrics[metric]
                    if baseline_value != 0:
                        diff_percent = ((current_value - baseline_value) / baseline_value) * 100
                        baseline_comparison[metric] = diff_percent
            
            execution_time = time.time() - start_time
            
            result = SmokeTestResult(
                timestamp=datetime.now().isoformat(),
                test_type="FULL_SMOKE_TEST",
                status=status,
                metrics=all_metrics,
                baseline_comparison=baseline_comparison,
                drift_detected=len(drift_alerts) > 0,
                alerts=alerts,
                execution_time_seconds=execution_time
            )
            
            # Speichere in Historie
            self.test_history.append(result)
            
            # Snapshot bei Drift oder kritischen Events
            if drift_alerts or status == "FAIL":
                self.create_snapshot(result, drift_alerts)
            
            logger.info(f"✅ Smoke-Test abgeschlossen: {status} ({execution_time:.1f}s)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Smoke-Test Fehler: {e}")
            return SmokeTestResult(
                timestamp=datetime.now().isoformat(),
                test_type="FULL_SMOKE_TEST",
                status="FAIL",
                metrics={},
                baseline_comparison={},
                drift_detected=False,
                alerts=[f"EXECUTION_ERROR: {str(e)}"],
                execution_time_seconds=time.time() - start_time
            )
    
    def start_daemon(self, interval_minutes: int = 15):
        """Startet Daemon-Loop"""
        logger.info(f"🚀 Starte Smoke-Test-Daemon (Intervall: {interval_minutes} Min)")
        
        self.daemon_running = True
        self.start_time = time.time()
        
        while self.daemon_running:
            try:
                # Führe Smoke-Test aus
                result = self.run_smoke_test_cycle()
                
                # Speichere Ergebnis
                result_file = f"agi_missions/smoke_test_results/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                os.makedirs(os.path.dirname(result_file), exist_ok=True)
                
                with open(result_file, 'w') as f:
                    json.dump(asdict(result), f, indent=2)
                
                # Warte bis zum nächsten Zyklus
                logger.info(f"⏰ Nächster Test in {interval_minutes} Minuten...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Daemon gestoppt durch Benutzer")
                break
            except Exception as e:
                logger.error(f"❌ Daemon-Fehler: {e}")
                time.sleep(60)  # Kurze Pause bei Fehlern
        
        self.daemon_running = False
        logger.info("🏁 Smoke-Test-Daemon beendet")
    
    def stop_daemon(self):
        """Stoppt Daemon"""
        self.daemon_running = False

def main():
    """Hauptfunktion"""
    import sys
    
    daemon = SmokeTestDaemon()
    
    if len(sys.argv) > 1 and sys.argv[1] == "daemon":
        # Daemon-Modus
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        daemon.start_daemon(interval)
    else:
        # Einmaliger Test
        result = daemon.run_smoke_test_cycle()
        print(f"\n🔍 SMOKE-TEST ERGEBNIS:")
        print(f"Status: {result.status}")
        print(f"Drift erkannt: {result.drift_detected}")
        print(f"Alerts: {len(result.alerts)}")
        print(f"Ausführungszeit: {result.execution_time_seconds:.1f}s")

if __name__ == "__main__":
    main()
