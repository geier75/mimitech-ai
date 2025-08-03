#!/usr/bin/env python3
"""
Continuous AGI Training Daemon
Automatisches, kontinuierliches Training der VXOR AGI-Fähigkeiten
"""

import json
import time
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from agi_training_mission import AGITrainingMission

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agi_missions/daemon_logs/continuous_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousAGITrainingDaemon:
    """Daemon für kontinuierliches AGI-Training"""
    
    def __init__(self, config_file: str = "agi_missions/continuous_training_config.json"):
        self.config = self._load_config(config_file)
        self.trainer = AGITrainingMission()
        self.is_running = False
        self.training_thread = None
        self.last_training_time = None
        self.training_schedule = self._setup_training_schedule()
        
        logger.info("🤖 Continuous AGI Training Daemon initialisiert")
        logger.info(f"📅 Training-Schedule: {len(self.training_schedule)} geplante Sessions")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Lädt Daemon-Konfiguration"""
        default_config = {
            "training_intervals": {
                "comprehensive": {"hours": 24, "enabled": True},
                "meta_learning": {"hours": 8, "enabled": True},
                "reasoning": {"hours": 12, "enabled": True},
                "creative": {"hours": 16, "enabled": True},
                "quantum": {"hours": 6, "enabled": True},
                "transfer": {"hours": 18, "enabled": True}
            },
            "performance_thresholds": {
                "min_improvement": 0.05,
                "max_training_duration": 3600,  # 1 Stunde
                "cooldown_period": 1800  # 30 Minuten
            },
            "adaptive_training": {
                "enabled": True,
                "performance_based_scheduling": True,
                "difficulty_adjustment": True
            },
            "monitoring": {
                "log_level": "INFO",
                "save_detailed_logs": True,
                "alert_on_failures": True
            }
        }
        
        try:
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                # Merge mit default config
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
            else:
                config = default_config
                # Speichere default config
                Path(config_file).parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                logger.info(f"📄 Default-Konfiguration erstellt: {config_file}")
            
            return config
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfiguration: {e}")
            return default_config
    
    def _setup_training_schedule(self) -> Dict[str, Any]:
        """Richtet Training-Schedule ein"""
        schedule_info = {}

        for training_type, config in self.config["training_intervals"].items():
            if config["enabled"]:
                hours = config["hours"]

                schedule_info[training_type] = {
                    "interval_hours": hours,
                    "next_run": datetime.now() + timedelta(hours=hours),
                    "enabled": True,
                    "last_run": None
                }

                logger.info(f"📅 {training_type} Training geplant alle {hours} Stunden")

        return schedule_info
    
    def _run_scheduled_training(self, training_type: str):
        """Führt geplantes Training aus"""
        if not self.is_running:
            return
        
        logger.info(f"🚀 Starte geplantes {training_type} Training")
        
        try:
            # Prüfe Performance-Schwellenwerte
            current_capabilities = self.trainer.get_training_status()
            
            if self._should_run_training(training_type, current_capabilities):
                # Führe Training aus
                session = self.trainer.start_training_session(training_type)
                
                start_time = time.time()
                
                # Simuliere Training-Ausführung
                phases = ["preparation", "focused_training", "validation", "integration"]
                phase_results = []
                
                for phase in phases:
                    if not self.is_running:  # Check if daemon was stopped
                        logger.info("🛑 Training unterbrochen - Daemon gestoppt")
                        return
                    
                    result = self.trainer.execute_training_phase(session, phase)
                    phase_results.append(result)
                    
                    # Kurze Pause zwischen Phasen
                    time.sleep(2)
                
                duration = time.time() - start_time
                
                # Logge Ergebnisse
                logger.info(f"✅ {training_type} Training abgeschlossen in {duration:.1f}s")
                
                # Update letzten Training-Zeitpunkt
                self.last_training_time = datetime.now()
                
                # Speichere Training-Statistiken
                self._save_training_stats(training_type, session, phase_results, duration)
                
            else:
                logger.info(f"⏭️ {training_type} Training übersprungen (Schwellenwerte nicht erreicht)")
                
        except Exception as e:
            logger.error(f"❌ Fehler beim {training_type} Training: {e}")
    
    def _should_run_training(self, training_type: str, current_status: Dict[str, Any]) -> bool:
        """Prüft ob Training ausgeführt werden soll"""
        # Prüfe Cooldown-Period
        if self.last_training_time:
            cooldown = self.config["performance_thresholds"]["cooldown_period"]
            time_since_last = (datetime.now() - self.last_training_time).total_seconds()
            if time_since_last < cooldown:
                return False
        
        # Adaptive Training Logic
        if self.config["adaptive_training"]["enabled"]:
            # Prüfe Performance-basierte Planung
            if self.config["adaptive_training"]["performance_based_scheduling"]:
                avg_capability = sum(current_status["current_capabilities"].values()) / len(current_status["current_capabilities"])
                
                # Wenn Performance hoch ist, reduziere Training-Frequenz
                if avg_capability > 0.95:
                    return False
                
                # Wenn Performance niedrig ist, erhöhe Training-Frequenz
                if avg_capability < 0.80:
                    return True
        
        return True
    
    def _save_training_stats(self, training_type: str, session: Any, results: List[Dict], duration: float):
        """Speichert Training-Statistiken"""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "training_type": training_type,
            "session_id": session.session_id,
            "duration_seconds": duration,
            "phases_completed": len(results),
            "success": True,
            "performance_improvements": {},
            "daemon_info": {
                "continuous_training": True,
                "scheduled_execution": True
            }
        }
        
        # Sammle Performance-Verbesserungen
        for result in results:
            if "performance_metrics" in result:
                for module, metrics in result["performance_metrics"].items():
                    stats["performance_improvements"][module] = metrics.get("improvement", 0.0)
        
        # Speichere Statistiken
        stats_file = f"agi_missions/training_results/continuous_stats_{int(time.time())}.json"
        Path(stats_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"📊 Training-Statistiken gespeichert: {stats_file}")
    
    def start_daemon(self):
        """Startet den Continuous Training Daemon"""
        if self.is_running:
            logger.warning("⚠️ Daemon läuft bereits")
            return
        
        self.is_running = True
        
        # Starte Schedule-Thread
        self.training_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.training_thread.start()
        
        logger.info("🚀 Continuous AGI Training Daemon gestartet")
        logger.info("📅 Schedule aktiv - Training läuft automatisch")
        
        # Zeige nächste geplante Trainings
        self._show_next_trainings()
    
    def _run_scheduler(self):
        """Führt den Schedule-Loop aus"""
        while self.is_running:
            try:
                current_time = datetime.now()

                # Prüfe alle geplanten Trainings
                for training_type, info in self.training_schedule.items():
                    if info["enabled"] and current_time >= info["next_run"]:
                        # Führe Training aus
                        self._run_scheduled_training(training_type)

                        # Update nächsten Lauf
                        info["next_run"] = current_time + timedelta(hours=info["interval_hours"])
                        info["last_run"] = current_time

                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"❌ Scheduler-Fehler: {e}")
                time.sleep(60)
    
    def stop_daemon(self):
        """Stoppt den Daemon"""
        if not self.is_running:
            logger.warning("⚠️ Daemon läuft nicht")
            return
        
        logger.info("🛑 Stoppe Continuous AGI Training Daemon...")
        self.is_running = False
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
        
        logger.info("✅ Daemon gestoppt")
    
    def _show_next_trainings(self):
        """Zeigt nächste geplante Trainings"""
        logger.info("📅 NÄCHSTE GEPLANTE TRAININGS:")
        
        for training_type, info in self.training_schedule.items():
            if info["enabled"]:
                next_run = info["next_run"]
                time_until = next_run - datetime.now()
                hours_until = time_until.total_seconds() / 3600
                
                logger.info(f"  🎯 {training_type}: in {hours_until:.1f} Stunden ({next_run.strftime('%H:%M')})")
    
    def get_daemon_status(self) -> Dict[str, Any]:
        """Gibt Daemon-Status zurück"""
        return {
            "is_running": self.is_running,
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "training_schedule": {
                training_type: {
                    "enabled": info["enabled"],
                    "interval_hours": info["interval_hours"],
                    "next_run": info["next_run"].isoformat()
                }
                for training_type, info in self.training_schedule.items()
            },
            "current_capabilities": self.trainer.get_training_status()["current_capabilities"]
        }
    
    def run_immediate_training(self, training_type: str = "comprehensive"):
        """Führt sofortiges Training aus"""
        logger.info(f"🚀 Starte sofortiges {training_type} Training")
        self._run_scheduled_training(training_type)

def main():
    """Hauptfunktion für Daemon"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous AGI Training Daemon")
    parser.add_argument("--action", choices=["start", "stop", "status", "train"], 
                       default="start", help="Daemon-Aktion")
    parser.add_argument("--training-type", default="comprehensive",
                       help="Training-Typ für sofortiges Training")
    parser.add_argument("--config", default="agi_missions/continuous_training_config.json",
                       help="Konfigurationsdatei")
    
    args = parser.parse_args()
    
    daemon = ContinuousAGITrainingDaemon(args.config)
    
    if args.action == "start":
        print("🤖 CONTINUOUS AGI TRAINING DAEMON")
        print("=" * 50)
        
        daemon.start_daemon()
        
        try:
            # Halte Daemon am Leben
            while daemon.is_running:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n🛑 Daemon wird gestoppt...")
            daemon.stop_daemon()
    
    elif args.action == "stop":
        daemon.stop_daemon()
    
    elif args.action == "status":
        status = daemon.get_daemon_status()
        print("\n📊 DAEMON STATUS:")
        print(f"Running: {status['is_running']}")
        print(f"Last Training: {status['last_training_time']}")
        print("\n📅 Training Schedule:")
        for training_type, info in status['training_schedule'].items():
            print(f"  {training_type}: {info['interval_hours']}h intervals, next: {info['next_run']}")
    
    elif args.action == "train":
        daemon.run_immediate_training(args.training_type)

if __name__ == "__main__":
    main()
