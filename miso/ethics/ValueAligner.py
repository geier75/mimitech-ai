#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - Value Aligner

Dieses Modul implementiert ein System zur Ausrichtung von Entscheidungen an definierten
Werten und Prioritäten. Es kann Konflikte zwischen verschiedenen Werten erkennen und
Kompromissvorschläge basierend auf einer Wertehierarchie generieren.

Autor: MISO ULTIMATE AGI Team
Datum: 26.04.2025
"""

import os
import json
import logging
import datetime
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set

# Konfiguration für Logging
log_dir = Path("/Volumes/My Book/MISO_Ultimate 15.32.28/logs/ethics")
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("miso.ethics.value_aligner")
logger.setLevel(logging.INFO)

# Datei-Handler für strukturierte Logs
file_handler = logging.FileHandler(log_dir / f"value_alignment_log_{datetime.datetime.now().strftime('%Y%m%d')}.json")
logger.addHandler(file_handler)

# Konsolen-Handler für Entwicklungszwecke
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


class ValueAligner:
    """
    System zur Ausrichtung von Entscheidungen an definierte Werte und Prioritäten.
    
    Diese Klasse bietet Methoden zur Anpassung von Entscheidungen an eine Hierarchie
    ethischer Werte, zur Erkennung von Wertkonflikten und zur Generierung von 
    Kompromissvorschlägen in konfliktreichen Situationen.
    """
    
    def __init__(self, values_hierarchy_path: Optional[str] = None):
        """
        Initialisiert den ValueAligner.
        
        Args:
            values_hierarchy_path: Optional, Pfad zur JSON-Datei mit Wertehierarchie.
                                  Wenn nicht angegeben, wird eine Standardhierarchie verwendet.
        """
        # Lade Wertehierarchie
        self.values_hierarchy = self._load_values_hierarchy(values_hierarchy_path)
        
        # Erstelle eine sortierte Liste der Werte nach Priorität
        self.prioritized_values = sorted(
            self.values_hierarchy["values"],
            key=lambda value: value.get("priority", 0),
            reverse=True
        )
        
        # Statistiken und Metriken
        self.alignment_stats = {
            "aligned_decisions": 0,
            "detected_conflicts": 0,
            "generated_compromises": 0,
            "alignment_changes": {value["id"]: 0 for value in self.values_hierarchy["values"]}
        }
        
        logger.info(f"ValueAligner initialisiert mit {len(self.values_hierarchy['values'])} Werten")
    
    def _load_values_hierarchy(self, values_hierarchy_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """
        Lädt die Wertehierarchie aus einer JSON-Datei.
        
        Args:
            values_hierarchy_path: Pfad zur JSON-Datei mit Wertehierarchie.
            
        Returns:
            Dictionary mit Wertehierarchie.
        """
        # Standardhierarchie definieren
        default_hierarchy = {
            "version": "1.0",
            "description": "MISO Ultimate AGI Wertehierarchie",
            "values": [
                {
                    "id": "human_wellbeing",
                    "name": "Menschliches Wohlbefinden",
                    "description": "Schutz und Förderung des physischen und psychischen Wohlbefindens von Menschen.",
                    "priority": 100
                },
                {
                    "id": "truth_accuracy",
                    "name": "Wahrheit und Genauigkeit",
                    "description": "Verpflichtung zur Wahrheit, Genauigkeit und Vermeidung von Falschinformationen.",
                    "priority": 95
                },
                {
                    "id": "autonomy",
                    "name": "Autonomie und Selbstbestimmung",
                    "description": "Respekt vor der Autonomie und Selbstbestimmung von Individuen.",
                    "priority": 90
                },
                {
                    "id": "fairness",
                    "name": "Fairness und Gerechtigkeit",
                    "description": "Faire und gerechte Behandlung aller Menschen ohne Diskriminierung.",
                    "priority": 85
                },
                {
                    "id": "privacy",
                    "name": "Privatsphäre und Datenschutz",
                    "description": "Schutz persönlicher Informationen und Respekt vor der Privatsphäre.",
                    "priority": 80
                },
                {
                    "id": "transparency",
                    "name": "Transparenz",
                    "description": "Offenheit und Nachvollziehbarkeit in Entscheidungsprozessen und Informationsbereitstellung.",
                    "priority": 75
                },
                {
                    "id": "beneficence",
                    "name": "Wohltätigkeit",
                    "description": "Aktives Handeln zum Wohle anderer und zur Verbesserung der Welt.",
                    "priority": 70
                }
            ],
            "conflict_resolution": {
                "default_strategy": "priority_based",
                "tie_breaker": "context_dependent",
                "documentation_required": True
            }
        }
        
        # Standardpfad, falls nicht angegeben
        if values_hierarchy_path is None:
            values_hierarchy_path = Path("/Volumes/My Book/MISO_Ultimate 15.32.28/miso/ethics/values_hierarchy.json")
        
        # Versuche, benutzerdefinierte Wertehierarchie zu laden
        if os.path.exists(values_hierarchy_path):
            try:
                with open(values_hierarchy_path, 'r', encoding='utf-8') as f:
                    custom_hierarchy = json.load(f)
                    return custom_hierarchy
            except Exception as e:
                logger.error(f"Fehler beim Laden der Wertehierarchie: {e}")
                logger.info("Verwende Standardhierarchie stattdessen")
        else:
            # Wenn die Datei nicht existiert, erstelle sie mit der Standardhierarchie
            try:
                values_dir = os.path.dirname(values_hierarchy_path)
                os.makedirs(values_dir, exist_ok=True)
                
                with open(values_hierarchy_path, 'w', encoding='utf-8') as f:
                    json.dump(default_hierarchy, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Standardhierarchie wurde in {values_hierarchy_path} gespeichert")
            except Exception as e:
                logger.error(f"Fehler beim Speichern der Standardhierarchie: {e}")
        
        return default_hierarchy
    
    def align_decision_with_values(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Justiert eine Entscheidung, um sie mit den definierten Werten in Einklang zu bringen.
        
        Args:
            decision_context: Dictionary mit Informationen über den Entscheidungskontext.
                            Muss mindestens 'decision', 'alternatives' und 'context' enthalten.
            
        Returns:
            Ein Dictionary mit der ausgerichteten Entscheidung und Begründung.
        """
        start_time = datetime.datetime.now()
        self.alignment_stats["aligned_decisions"] += 1
        
        # Validiere Eingabe
        if not isinstance(decision_context, dict) or 'decision' not in decision_context:
            raise ValueError("Der Entscheidungskontext muss ein Dictionary mit mindestens 'decision' sein")
        
        # Extrahiere relevante Informationen
        original_decision = decision_context["decision"]
        alternatives = decision_context.get("alternatives", [])
        context = decision_context.get("context", {})
        
        # Erstelle Basis-Bericht
        alignment = {
            "alignment_id": str(uuid.uuid4()),
            "timestamp": start_time.isoformat(),
            "original_decision": original_decision,
            "aligned_decision": original_decision,  # Standard: keine Änderung
            "was_modified": False,
            "value_analysis": [],
            "conflicts": [],
            "rationale": "",
            "metadata": {
                "context_type": context.get("type", "general"),
                "alternatives_count": len(alternatives)
            }
        }
        
        # Analysiere die Entscheidung auf Wertekonformität
        value_conflicts = self._detect_value_conflicts(original_decision, context)
        
        # Wenn Konflikte erkannt wurden, passe die Entscheidung an
        if value_conflicts:
            alignment["conflicts"] = value_conflicts
            self.alignment_stats["detected_conflicts"] += 1
            
            # Generiere einen Kompromiss, wenn Konflikte vorhanden sind
            compromise = self._generate_best_compromise(
                original_decision, alternatives, value_conflicts, context)
            
            if compromise["decision"] != original_decision:
                alignment["aligned_decision"] = compromise["decision"]
                alignment["was_modified"] = True
                alignment["rationale"] = compromise["rationale"]
                
                # Aktualisiere Statistiken
                self.alignment_stats["generated_compromises"] += 1
                for value_id in compromise["affected_values"]:
                    if value_id in self.alignment_stats["alignment_changes"]:
                        self.alignment_stats["alignment_changes"][value_id] += 1
        
        # Führe Werteanalyse für die finale Entscheidung durch
        alignment["value_analysis"] = self._analyze_decision_values(
            alignment["aligned_decision"], context)
        
        # Berechne Gesamtdauer
        end_time = datetime.datetime.now()
        alignment["alignment_duration_ms"] = (end_time - start_time).total_seconds() * 1000
        
        # Speichere Bericht als strukturiertes JSON-Log
        self._log_alignment(alignment)
        
        return alignment
    
    def _detect_value_conflicts(self, decision: Any, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Erkennt Konflikte zwischen verschiedenen Werten in einer Entscheidung.
        
        Args:
            decision: Die zu analysierende Entscheidung.
            context: Zusätzliche Kontextinformationen.
            
        Returns:
            Eine Liste von erkannten Wertkonflikten.
        """
        conflicts = []
        
        # In einer vollständigen Implementierung würde hier eine detaillierte
        # Analyse der Entscheidung auf Konflikte mit verschiedenen Werten erfolgen
        
        # Simuliere Konfliktanalyse für Demonstrationszwecke
        import random
        
        # Einfache Simulation von Konflikterkennung
        # In einer echten Implementierung würde hier eine komplexe Analyse stattfinden
        
        # Wähle zufällig 0-2 Wertekonflikte aus
        num_conflicts = min(2, random.randint(0, 2))
        
        if num_conflicts > 0:
            # Wähle zufällig Werte aus, die in Konflikt stehen könnten
            available_values = self.values_hierarchy["values"].copy()
            random.shuffle(available_values)
            
            for i in range(num_conflicts):
                if i >= len(available_values) - 1:
                    break
                    
                value1 = available_values[i]
                value2 = available_values[i + 1]
                
                # Erstelle einen simulierten Konflikt
                conflicts.append({
                    "conflict_id": str(uuid.uuid4()),
                    "values": [
                        {"id": value1["id"], "name": value1["name"], "priority": value1["priority"]},
                        {"id": value2["id"], "name": value2["name"], "priority": value2["priority"]}
                    ],
                    "description": f"Konflikt zwischen {value1['name']} und {value2['name']} in der Entscheidung",
                    "severity": random.choice(["low", "medium", "high"]),
                    "context_factors": [f"Kontextfaktor {i+1}"]
                })
        
        return conflicts
    
    def _generate_best_compromise(self, 
                                original_decision: Any, 
                                alternatives: List[Any],
                                conflicts: List[Dict[str, Any]],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generiert einen optimalen Kompromiss bei Wertkonflikten.
        
        Args:
            original_decision: Die ursprüngliche Entscheidung.
            alternatives: Alternative Entscheidungsmöglichkeiten.
            conflicts: Liste der erkannten Wertkonflikte.
            context: Zusätzliche Kontextinformationen.
            
        Returns:
            Ein Dictionary mit dem besten Kompromiss und Begründung.
        """
        # In einer vollständigen Implementierung würde hier ein komplexer
        # Algorithmus zur Findung des optimalen Kompromisses implementiert sein
        
        # Einfache Simulation für Demonstrationszwecke
        import random
        
        # Standard: keine Änderung notwendig
        compromise = {
            "decision": original_decision,
            "rationale": "Die ursprüngliche Entscheidung entspricht allen relevanten Werten.",
            "affected_values": []
        }
        
        # Wenn es Konflikte gibt, simuliere einen Kompromiss
        if conflicts:
            # Sammle alle betroffenen Werte
            affected_value_ids = set()
            for conflict in conflicts:
                for value in conflict["values"]:
                    affected_value_ids.add(value["id"])
            
            compromise["affected_values"] = list(affected_value_ids)
            
            # Entscheide, ob eine Änderung notwendig ist (70% Wahrscheinlichkeit)
            if random.random() < 0.7 and alternatives:
                # Wähle eine zufällige Alternative
                new_decision = random.choice(alternatives)
                
                # Erstelle eine Begründung basierend auf dem höchstprioren betroffenen Wert
                highest_priority_value = None
                for value in self.prioritized_values:
                    if value["id"] in affected_value_ids:
                        highest_priority_value = value
                        break
                
                if highest_priority_value:
                    compromise["decision"] = new_decision
                    compromise["rationale"] = (
                        f"Die Entscheidung wurde angepasst, um dem Wert '{highest_priority_value['name']}' "
                        f"besser zu entsprechen. Die Konflikte mit anderen Werten wurden abgewogen und "
                        f"ein optimaler Kompromiss wurde gefunden."
                    )
                    
                    # Füge detailliertere Begründung hinzu, wenn schwerwiegende Konflikte vorhanden sind
                    severe_conflicts = [c for c in conflicts if c.get("severity") == "high"]
                    if severe_conflicts:
                        compromise["rationale"] += (
                            f" Besonders berücksichtigt wurden die schwerwiegenden Konflikte im Bezug auf "
                            f"{', '.join([c['values'][0]['name'] for c in severe_conflicts])}."
                        )
        
        return compromise
    
    def _analyze_decision_values(self, decision: Any, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analysiert eine Entscheidung im Hinblick auf verschiedene Werte.
        
        Args:
            decision: Die zu analysierende Entscheidung.
            context: Zusätzliche Kontextinformationen.
            
        Returns:
            Eine Liste von Werteanalysen für die Entscheidung.
        """
        # In einer vollständigen Implementierung würde hier eine detaillierte
        # Analyse der Entscheidung im Hinblick auf jeden Wert durchgeführt
        
        # Simuliere Werteanalyse für Demonstrationszwecke
        import random
        
        value_analysis = []
        
        # Analysiere die Entscheidung für jeden Wert
        for value in self.prioritized_values:
            # Simuliere einen Konformitätsscore (0-100%)
            conformity_score = random.uniform(60, 100)
            value_impact = "neutral"
            
            if conformity_score >= 80:
                value_impact = "positive"
            elif conformity_score <= 70:
                value_impact = "negative"
            
            value_analysis.append({
                "value_id": value["id"],
                "value_name": value["name"],
                "conformity_score": round(conformity_score, 2),
                "impact": value_impact,
                "details": f"Analyse für Wert '{value['name']}'"
            })
        
        return value_analysis
    
    def _log_alignment(self, alignment: Dict[str, Any]) -> None:
        """
        Speichert eine Werteanpassung als strukturiertes JSON-Log.
        
        Args:
            alignment: Die zu speichernde Werteanpassung.
        """
        try:
            log_entry = json.dumps(alignment)
            logger.info(log_entry)
            
            # Speichere vollständige Anpassung als separate JSON-Datei für signifikante Fälle
            if alignment["was_modified"] or alignment["conflicts"]:
                align_dir = log_dir / "value_alignments"
                os.makedirs(align_dir, exist_ok=True)
                
                align_file = align_dir / f"value_alignment_{alignment['alignment_id']}.json"
                with open(align_file, 'w', encoding='utf-8') as f:
                    json.dump(alignment, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Werteanpassung: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über die Werteanpassungsaktivitäten zurück.
        
        Returns:
            Ein Dictionary mit Statistiken.
        """
        return self.alignment_stats


# Einfache Testfunktion
def test_value_aligner():
    """Einfacher Test für den ValueAligner."""
    aligner = ValueAligner()
    
    # Teste einen einfachen Entscheidungskontext ohne Konflikte
    simple_context = {
        "decision": "Bereitstellen von neutralen, faktischen Informationen zu einem historischen Ereignis",
        "alternatives": [
            "Informationen mit interpretativem Kontext bereitstellen",
            "Nur Rohdaten ohne Kontext bereitstellen"
        ],
        "context": {
            "type": "educational",
            "sensitivity": "low",
            "audience": "general"
        }
    }
    
    # Teste einen komplexeren Entscheidungskontext mit potenziellen Konflikten
    complex_context = {
        "decision": "Bereitstellen von sensiblen Informationen auf Anfrage ohne Zugriffsbeschränkungen",
        "alternatives": [
            "Informationen mit Zugangsbeschränkungen bereitstellen",
            "Nur allgemeine, nicht-sensible Informationen bereitstellen",
            "Anfrage für sensible Informationen ablehnen"
        ],
        "context": {
            "type": "information_request",
            "sensitivity": "high",
            "audience": "specific",
            "data_category": "privacy_relevant"
        }
    }
    
    # Führe Werteanpassungen durch
    print("Teste einfachen Entscheidungskontext...")
    simple_result = aligner.align_decision_with_values(simple_context)
    print(f"Ursprüngliche Entscheidung: {simple_result['original_decision']}")
    print(f"Angepasste Entscheidung: {simple_result['aligned_decision']}")
    print(f"Wurde modifiziert: {simple_result['was_modified']}")
    
    print("\nTeste komplexen Entscheidungskontext...")
    complex_result = aligner.align_decision_with_values(complex_context)
    print(f"Ursprüngliche Entscheidung: {complex_result['original_decision']}")
    print(f"Angepasste Entscheidung: {complex_result['aligned_decision']}")
    print(f"Wurde modifiziert: {complex_result['was_modified']}")
    
    # Zeige Begründung, wenn die Entscheidung angepasst wurde
    if complex_result['was_modified']:
        print(f"\nBegründung für Anpassung: {complex_result['rationale']}")
        
        # Zeige Konflikte an
        if complex_result['conflicts']:
            print("\nErkannte Konflikte:")
            for i, conflict in enumerate(complex_result['conflicts'], 1):
                print(f"{i}. Konflikt zwischen {conflict['values'][0]['name']} und {conflict['values'][1]['name']}")
    
    # Zeige Statistiken
    print("\nStatistiken:")
    stats = aligner.get_statistics()
    print(f"Angepasste Entscheidungen: {stats['aligned_decisions']}")
    print(f"Erkannte Konflikte: {stats['detected_conflicts']}")
    print(f"Generierte Kompromisse: {stats['generated_compromises']}")
    
    return simple_result, complex_result


if __name__ == "__main__":
    test_value_aligner()
