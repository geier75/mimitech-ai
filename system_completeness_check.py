#!/usr/bin/env python3
"""
VXOR AGI-System Vollständigkeitsprüfung
Überprüft alle kritischen Komponenten auf der externen Festplatte
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemCompletenessChecker:
    """Prüft Vollständigkeit des VXOR AGI-Systems"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.missing_components = []
        self.found_components = []
        self.critical_issues = []
        
    def check_critical_directories(self) -> Dict[str, bool]:
        """Prüft kritische Verzeichnisse"""
        logger.info("🔍 Prüfe kritische Verzeichnisse...")
        
        critical_dirs = {
            "agi_missions": "AGI Mission Framework",
            "miso": "MISO Core System", 
            "vxor": "VXOR Kernmodule",
            "agents": "Multi-Agent System",
            "security": "Sicherheitsframework",
            "engines": "Computation Engines",
            "tests": "Test Framework",
            "benchmark_results": "Performance Validierung",
            "vxor_c4_export_full": "Systemdokumentation",
            "backup": "System Backups"
        }
        
        results = {}
        for dir_name, description in critical_dirs.items():
            dir_path = self.base_path / dir_name
            exists = dir_path.exists() and dir_path.is_dir()
            results[dir_name] = exists
            
            if exists:
                self.found_components.append(f"✅ {description}: {dir_name}")
                logger.info(f"✅ {description}: {dir_name}")
            else:
                self.missing_components.append(f"❌ {description}: {dir_name}")
                logger.error(f"❌ FEHLT: {description}: {dir_name}")
        
        return results
    
    def check_agi_missions(self) -> Dict[str, Any]:
        """Prüft AGI Missions Komponenten"""
        logger.info("🎯 Prüfe AGI Missions...")
        
        agi_path = self.base_path / "agi_missions"
        if not agi_path.exists():
            self.critical_issues.append("❌ KRITISCH: agi_missions Verzeichnis fehlt")
            return {"status": "MISSING", "components": []}
        
        required_files = {
            "agi_mission_executor.py": "Mission Executor",
            "transfer_mission_executor.py": "Transfer Learning Engine",
            "compare_transfer_baselines.py": "A/B-Test Framework",
            "production_config_v2.1.yaml": "Produktionskonfiguration",
            "deployment_decision_report.md": "Deployment Report",
            "production_live_monitor.py": "Live Monitoring",
            "vx_control_fallback_policy.py": "Fallback Policy",
            "canary_deployment_executor.py": "Canary Deployment",
            "ab_test_report.yaml": "A/B-Test Ergebnisse",
            "new_baseline.json": "Neue Baseline Daten",
            "old_baseline.json": "Alte Baseline Daten"
        }
        
        found_files = {}
        for filename, description in required_files.items():
            file_path = agi_path / filename
            exists = file_path.exists()
            found_files[filename] = exists
            
            if exists:
                self.found_components.append(f"✅ {description}: {filename}")
            else:
                self.missing_components.append(f"❌ {description}: {filename}")
        
        # Prüfe Canary Results
        canary_results = list(agi_path.glob("canary_deployment_results_*.json"))
        if canary_results:
            self.found_components.append(f"✅ Canary Results: {len(canary_results)} Dateien")
        else:
            self.missing_components.append("❌ Canary Deployment Results fehlen")
        
        return {
            "status": "OK" if all(found_files.values()) else "INCOMPLETE",
            "components": found_files,
            "canary_results": len(canary_results)
        }
    
    def check_core_systems(self) -> Dict[str, Any]:
        """Prüft Kernsysteme"""
        logger.info("🔧 Prüfe Kernsysteme...")
        
        core_systems = {
            "miso": {
                "vxor/vx_psi.py": "VX-PSI (Self-Awareness)",
                "vxor/vx_memex.py": "VX-MEMEX (Memory)",
                "vxor/vx_quantum.py": "VX-QUANTUM (Quantum Computing)",
                "simulation/prism_engine.py": "PRISM (Simulation)",
                "logic/qlogik_engine.py": "Q-LOGIK (Logic Engine)",
                "math/t_mathematics/": "T-MATHEMATICS (Math Engine)"
            },
            "vxor": {
                "core/vx_core.py": "VXOR Core",
                "math/t_mathematics.py": "T-Math Integration",
                "vision/vx_vision_core.py": "VX-VISION",
                "security/core.py": "Security Core",
                "agents/vx_memex.py": "Agent Framework"
            }
        }
        
        results = {}
        for system_name, components in core_systems.items():
            system_path = self.base_path / system_name
            system_results = {}
            
            for component_path, description in components.items():
                full_path = system_path / component_path
                exists = full_path.exists()
                system_results[component_path] = exists
                
                if exists:
                    self.found_components.append(f"✅ {description}")
                else:
                    self.missing_components.append(f"❌ {description}: {component_path}")
            
            results[system_name] = system_results
        
        return results
    
    def check_git_status(self) -> Dict[str, Any]:
        """Prüft Git Status und Tags"""
        logger.info("📋 Prüfe Git Status...")
        
        git_path = self.base_path / ".git"
        if not git_path.exists():
            self.critical_issues.append("❌ KRITISCH: Git Repository nicht initialisiert")
            return {"status": "NO_GIT"}
        
        # Prüfe Tags
        try:
            import subprocess
            result = subprocess.run(
                ["git", "tag", "--list"], 
                cwd=self.base_path, 
                capture_output=True, 
                text=True
            )
            
            tags = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            required_tags = ["v2.0-transfer-baseline", "v2.1-transfer-baseline"]
            found_tags = [tag for tag in required_tags if tag in tags]
            missing_tags = [tag for tag in required_tags if tag not in tags]
            
            if found_tags:
                self.found_components.append(f"✅ Git Tags: {', '.join(found_tags)}")
            
            if missing_tags:
                self.missing_components.append(f"❌ Fehlende Tags: {', '.join(missing_tags)}")
            
            return {
                "status": "OK",
                "tags": tags,
                "required_tags_found": len(found_tags),
                "total_tags": len(tags)
            }
            
        except Exception as e:
            self.critical_issues.append(f"❌ Git Status Fehler: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    def check_production_readiness(self) -> Dict[str, Any]:
        """Prüft Produktionsbereitschaft"""
        logger.info("🚀 Prüfe Produktionsbereitschaft...")
        
        production_indicators = {
            "agi_missions/production_config_v2.1.yaml": "Produktionskonfiguration",
            "agi_missions/deployment_decision_report.md": "Deployment Report",
            "agi_missions/ab_test_report.yaml": "A/B-Test Validierung",
            "agi_missions/production_live_monitor.py": "Live Monitoring",
            "agi_missions/vx_control_fallback_policy.py": "Fallback Policy"
        }
        
        readiness_score = 0
        total_indicators = len(production_indicators)
        
        for file_path, description in production_indicators.items():
            full_path = self.base_path / file_path
            if full_path.exists():
                readiness_score += 1
                self.found_components.append(f"✅ {description}")
            else:
                self.missing_components.append(f"❌ {description}: {file_path}")
        
        # Prüfe Canary Results
        canary_path = self.base_path / "agi_missions"
        canary_files = list(canary_path.glob("canary_deployment_results_*.json")) if canary_path.exists() else []
        
        if canary_files:
            readiness_score += 1
            self.found_components.append(f"✅ Canary Validierung: {len(canary_files)} Results")
        else:
            self.missing_components.append("❌ Canary Deployment Results fehlen")
            total_indicators += 1
        
        readiness_percentage = (readiness_score / total_indicators) * 100
        
        return {
            "readiness_score": readiness_score,
            "total_indicators": total_indicators,
            "readiness_percentage": readiness_percentage,
            "status": "READY" if readiness_percentage >= 90 else "NOT_READY"
        }
    
    def generate_completeness_report(self) -> Dict[str, Any]:
        """Generiert vollständigen Bericht"""
        logger.info("📊 Generiere Vollständigkeitsbericht...")
        
        # Führe alle Prüfungen durch
        directories = self.check_critical_directories()
        agi_missions = self.check_agi_missions()
        core_systems = self.check_core_systems()
        git_status = self.check_git_status()
        production_readiness = self.check_production_readiness()
        
        # Berechne Gesamtscore
        total_components = len(self.found_components) + len(self.missing_components)
        found_percentage = (len(self.found_components) / total_components * 100) if total_components > 0 else 0
        
        report = {
            "timestamp": "2025-08-03T05:00:00Z",
            "system_status": "COMPLETE" if found_percentage >= 95 else "INCOMPLETE",
            "completeness_percentage": found_percentage,
            "summary": {
                "found_components": len(self.found_components),
                "missing_components": len(self.missing_components),
                "critical_issues": len(self.critical_issues)
            },
            "checks": {
                "directories": directories,
                "agi_missions": agi_missions,
                "core_systems": core_systems,
                "git_status": git_status,
                "production_readiness": production_readiness
            },
            "found_components": self.found_components,
            "missing_components": self.missing_components,
            "critical_issues": self.critical_issues
        }
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """Druckt Zusammenfassung"""
        print("\n" + "="*80)
        print("🔍 VXOR AGI-SYSTEM VOLLSTÄNDIGKEITSPRÜFUNG")
        print("="*80)
        
        status_emoji = "✅" if report["system_status"] == "COMPLETE" else "⚠️"
        print(f"{status_emoji} SYSTEM STATUS: {report['system_status']}")
        print(f"📊 VOLLSTÄNDIGKEIT: {report['completeness_percentage']:.1f}%")
        print(f"✅ GEFUNDEN: {report['summary']['found_components']} Komponenten")
        print(f"❌ FEHLT: {report['summary']['missing_components']} Komponenten")
        print(f"🚨 KRITISCH: {report['summary']['critical_issues']} Issues")
        
        # Produktionsbereitschaft
        prod_status = report["checks"]["production_readiness"]
        prod_emoji = "🚀" if prod_status["status"] == "READY" else "⚠️"
        print(f"{prod_emoji} PRODUKTIONSBEREITSCHAFT: {prod_status['readiness_percentage']:.1f}%")
        
        # Git Status
        git_status = report["checks"]["git_status"]
        if git_status["status"] == "OK":
            print(f"📋 GIT TAGS: {git_status['total_tags']} Tags gefunden")
        
        print("="*80)
        
        if report["critical_issues"]:
            print("\n🚨 KRITISCHE ISSUES:")
            for issue in report["critical_issues"]:
                print(f"  {issue}")
        
        if report["missing_components"]:
            print(f"\n❌ FEHLENDE KOMPONENTEN ({len(report['missing_components'])}):")
            for component in report["missing_components"][:10]:  # Zeige nur erste 10
                print(f"  {component}")
            if len(report["missing_components"]) > 10:
                print(f"  ... und {len(report['missing_components']) - 10} weitere")

def main():
    """Hauptfunktion"""
    checker = SystemCompletenessChecker()
    report = checker.generate_completeness_report()
    
    # Speichere Report
    with open("system_completeness_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Drucke Zusammenfassung
    checker.print_summary(report)
    
    return report

if __name__ == "__main__":
    main()
