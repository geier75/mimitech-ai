#!/usr/bin/env python3
"""
AGI Training Dashboard
Übersicht über alle Training-Aktivitäten und Fortschritte
"""

import json
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from agi_training_mission import AGITrainingMission

class AGITrainingDashboard:
    """Dashboard für AGI-Training Übersicht"""
    
    def __init__(self):
        self.trainer = AGITrainingMission()
        self.training_results_dir = Path("agi_missions/training_results")
        self.training_results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Erstellt Training-Zusammenfassung"""
        # Lade alle Training-Ergebnisse
        training_files = list(self.training_results_dir.glob("*.json"))
        
        summary = {
            "total_sessions": len(training_files),
            "training_types": {},
            "performance_trends": {},
            "recent_activities": [],
            "system_status": self.trainer.get_training_status()
        }
        
        # Analysiere Training-Dateien
        for file_path in training_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extrahiere Training-Typ
                if "training_type" in data:
                    training_type = data["training_type"]
                    if training_type not in summary["training_types"]:
                        summary["training_types"][training_type] = {
                            "count": 0,
                            "total_improvements": 0,
                            "avg_improvement": 0
                        }
                    
                    summary["training_types"][training_type]["count"] += 1
                    
                    # Sammle Performance-Verbesserungen
                    if "performance_improvements" in data:
                        improvements = list(data["performance_improvements"].values())
                        if improvements:
                            avg_improvement = sum(improvements) / len(improvements)
                            summary["training_types"][training_type]["total_improvements"] += avg_improvement
                
                # Sammle kürzliche Aktivitäten
                if "start_time" in data:
                    summary["recent_activities"].append({
                        "timestamp": data["start_time"],
                        "type": data.get("training_type", "unknown"),
                        "success_rate": data.get("success_rate", 0),
                        "new_capabilities": len(data.get("new_capabilities", []))
                    })
                    
            except Exception as e:
                print(f"Fehler beim Lesen von {file_path}: {e}")
        
        # Berechne Durchschnitte
        for training_type, stats in summary["training_types"].items():
            if stats["count"] > 0:
                stats["avg_improvement"] = stats["total_improvements"] / stats["count"]
        
        # Sortiere kürzliche Aktivitäten
        summary["recent_activities"].sort(key=lambda x: x["timestamp"], reverse=True)
        summary["recent_activities"] = summary["recent_activities"][:10]  # Nur die letzten 10
        
        return summary
    
    def display_dashboard(self):
        """Zeigt das Training-Dashboard an"""
        summary = self.get_training_summary()
        
        print("\n🧠 VXOR AGI TRAINING DASHBOARD")
        print("=" * 60)
        print(f"📊 Letzte Aktualisierung: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # System Status
        status = summary["system_status"]
        print("🎯 SYSTEM STATUS:")
        print(f"  📈 Abgeschlossene Sessions: {status['training_sessions_completed']}")
        print(f"  ⭐ Durchschnittliche Success Rate: {status['avg_success_rate']:.1%}")
        print(f"  🆕 Neue Fähigkeiten entwickelt: {status['total_new_capabilities']}")
        print()
        
        # Aktuelle Fähigkeiten
        print("🔍 AKTUELLE FÄHIGKEITEN:")
        capabilities = status["current_capabilities"]
        for capability, score in sorted(capabilities.items(), key=lambda x: x[1], reverse=True):
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {capability:20} [{bar}] {score:.1%}")
        print()
        
        # Training-Typen Statistiken
        if summary["training_types"]:
            print("📊 TRAINING-TYPEN STATISTIKEN:")
            for training_type, stats in summary["training_types"].items():
                print(f"  🎯 {training_type:15} | Sessions: {stats['count']:2} | Ø Verbesserung: {stats['avg_improvement']:.1%}")
            print()
        
        # Kürzliche Aktivitäten
        if summary["recent_activities"]:
            print("📅 KÜRZLICHE TRAINING-AKTIVITÄTEN:")
            for activity in summary["recent_activities"][:5]:
                timestamp = datetime.fromisoformat(activity["timestamp"].replace('Z', '+00:00'))
                time_ago = self._time_ago(timestamp)
                print(f"  🕐 {time_ago:15} | {activity['type']:12} | Success: {activity['success_rate']:.1%} | +{activity['new_capabilities']} Fähigkeiten")
            print()
        
        # Empfehlungen
        self._show_recommendations(summary)
    
    def _time_ago(self, timestamp: datetime) -> str:
        """Berechnet Zeit seit Timestamp"""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=None)
        
        now = datetime.now()
        if timestamp.tzinfo:
            now = now.replace(tzinfo=timestamp.tzinfo)
        
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours}h ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes}m ago"
        else:
            return "just now"
    
    def _show_recommendations(self, summary: Dict[str, Any]):
        """Zeigt Training-Empfehlungen"""
        print("💡 TRAINING-EMPFEHLUNGEN:")
        
        capabilities = summary["system_status"]["current_capabilities"]
        
        # Finde schwächste Fähigkeiten
        weak_capabilities = [(name, score) for name, score in capabilities.items() if score < 0.80]
        weak_capabilities.sort(key=lambda x: x[1])
        
        if weak_capabilities:
            print("  🎯 Fokus auf schwache Bereiche:")
            for name, score in weak_capabilities[:3]:
                print(f"    - {name}: {score:.1%} (Verbesserung empfohlen)")
        
        # Training-Frequenz Empfehlungen
        training_counts = summary["training_types"]
        if training_counts:
            least_trained = min(training_counts.items(), key=lambda x: x[1]["count"])
            print(f"  🔄 Mehr {least_trained[0]} Training empfohlen (nur {least_trained[1]['count']} Sessions)")
        
        # Allgemeine Empfehlungen
        avg_capability = sum(capabilities.values()) / len(capabilities)
        if avg_capability > 0.90:
            print("  🚀 System-Performance exzellent - Fokus auf spezialisierte Trainings")
        elif avg_capability > 0.80:
            print("  📈 Gute Performance - Kontinuierliches Training empfohlen")
        else:
            print("  ⚠️ Performance unter Ziel - Intensives Training erforderlich")
        
        print()
    
    def export_report(self, filename: str = None) -> str:
        """Exportiert Training-Report"""
        if filename is None:
            filename = f"agi_missions/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = self.get_training_summary()
        
        # Erweitere Report um zusätzliche Metriken
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "recommendations": self._generate_recommendations(summary),
            "performance_analysis": self._analyze_performance_trends(summary)
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Training-Report exportiert: {filename}")
        return filename
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generiert strukturierte Empfehlungen"""
        recommendations = []
        
        capabilities = summary["system_status"]["current_capabilities"]
        
        # Schwache Bereiche identifizieren
        for name, score in capabilities.items():
            if score < 0.75:
                recommendations.append({
                    "type": "capability_improvement",
                    "priority": "HIGH",
                    "capability": name,
                    "current_score": score,
                    "target_score": 0.85,
                    "suggested_training": "focused_training",
                    "estimated_sessions": 3
                })
            elif score < 0.85:
                recommendations.append({
                    "type": "capability_improvement",
                    "priority": "MEDIUM",
                    "capability": name,
                    "current_score": score,
                    "target_score": 0.90,
                    "suggested_training": "regular_training",
                    "estimated_sessions": 2
                })
        
        return recommendations
    
    def _analyze_performance_trends(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analysiert Performance-Trends"""
        return {
            "total_training_sessions": summary["total_sessions"],
            "training_diversity": len(summary["training_types"]),
            "avg_system_performance": sum(summary["system_status"]["current_capabilities"].values()) / len(summary["system_status"]["current_capabilities"]),
            "improvement_trend": "positive" if summary["total_sessions"] > 0 else "neutral",
            "most_effective_training": max(summary["training_types"].items(), key=lambda x: x[1]["avg_improvement"])[0] if summary["training_types"] else None
        }
    
    def run_live_dashboard(self, refresh_interval: int = 30):
        """Führt Live-Dashboard aus"""
        print("🔴 LIVE AGI TRAINING DASHBOARD")
        print("Drücken Sie Ctrl+C zum Beenden")
        print()
        
        try:
            while True:
                # Clear screen (works on most terminals)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                self.display_dashboard()
                
                print(f"🔄 Nächste Aktualisierung in {refresh_interval} Sekunden...")
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n👋 Live-Dashboard beendet")

def main():
    """Hauptfunktion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AGI Training Dashboard")
    parser.add_argument("--live", action="store_true", help="Live-Dashboard starten")
    parser.add_argument("--export", action="store_true", help="Report exportieren")
    parser.add_argument("--refresh", type=int, default=30, help="Refresh-Intervall für Live-Dashboard")
    
    args = parser.parse_args()
    
    dashboard = AGITrainingDashboard()
    
    if args.live:
        dashboard.run_live_dashboard(args.refresh)
    elif args.export:
        dashboard.export_report()
    else:
        dashboard.display_dashboard()

if __name__ == "__main__":
    main()
