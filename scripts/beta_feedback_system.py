#!/usr/bin/env python3
"""
VXOR B2C Alpha - A/B Testing & Feedback System
Mini-Beta Framework für 10-20 Nutzer mit offline-fähigem Feedback
"""

import os
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

class BetaFeedbackSystem:
    def __init__(self, bundle_dir: str):
        self.bundle_dir = Path(bundle_dir)
        self.feedback_dir = self.bundle_dir / "feedback"
        self.feedback_dir.mkdir(exist_ok=True)
        
        # A/B Test Konfiguration
        self.ab_tests = {
            'response_style': {
                'A': 'detailed',  # Ausführlichere Antworten
                'B': 'concise'    # Knappere Antworten
            },
            'explanation_depth': {
                'A': 'step_by_step',  # Schritt-für-Schritt Erklärungen
                'B': 'summary_only'   # Nur Zusammenfassung
            }
        }
        
        # Nutzer zu Test-Gruppe zuordnen
        self.load_user_assignments()
        
    def load_user_assignments(self):
        """Lade oder erstelle Nutzer-zu-Gruppe Zuordnungen"""
        assignments_file = self.feedback_dir / "user_assignments.json"
        
        if assignments_file.exists():
            with open(assignments_file, 'r') as f:
                self.user_assignments = json.load(f)
        else:
            # Neue Zuordnungen erstellen
            self.user_assignments = {}
            
            # API-Keys aus config lesen
            api_keys_file = self.bundle_dir / "config" / "api_keys.txt"
            if api_keys_file.exists():
                with open(api_keys_file, 'r') as f:
                    lines = f.readlines()
                
                users = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(':')
                        if len(parts) >= 2:
                            users.append(parts[1])  # user_id
                
                # 50/50 Split für A/B Tests
                for i, user_id in enumerate(users):
                    group = 'A' if i % 2 == 0 else 'B'
                    self.user_assignments[user_id] = {
                        'response_style': group,
                        'explanation_depth': 'A' if (i // 2) % 2 == 0 else 'B'
                    }
            
            # Speichern
            with open(assignments_file, 'w') as f:
                json.dump(self.user_assignments, f, indent=2)
    
    def get_user_variant(self, user_id: str, test_name: str) -> str:
        """Hole Test-Variante für Nutzer"""
        return self.user_assignments.get(user_id, {}).get(test_name, 'A')
    
    def log_interaction(self, user_id: str, query: str, response: str, 
                       metadata: Dict[str, Any] = None):
        """Logge Nutzer-Interaktion für Analyse"""
        if metadata is None:
            metadata = {}
            
        interaction = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'query_hash': hashlib.md5(query.encode()).hexdigest()[:8],  # Keine PII
            'response_length': len(response),
            'variants': {
                test: self.get_user_variant(user_id, test) 
                for test in self.ab_tests.keys()
            },
            'metadata': metadata
        }
        
        # Tägliche Log-Datei
        log_date = datetime.now().strftime('%Y%m%d')
        log_file = self.feedback_dir / f"interactions_{log_date}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(interaction) + '\n')
    
    def create_feedback_form(self, user_id: str, interaction_id: str) -> str:
        """Erstelle Feedback-Formular für Nutzer"""
        form_id = str(uuid.uuid4())[:8]
        
        form_data = {
            'form_id': form_id,
            'user_id': user_id,
            'interaction_id': interaction_id,
            'created': datetime.now().isoformat(),
            'questions': [
                {
                    'id': 'quality',
                    'type': 'rating',
                    'question': 'Wie hilfreich war die Antwort?',
                    'scale': '1-5'
                },
                {
                    'id': 'accuracy',
                    'type': 'rating', 
                    'question': 'Wie genau war die Antwort?',
                    'scale': '1-5'
                },
                {
                    'id': 'style',
                    'type': 'choice',
                    'question': 'Welchen Antwort-Stil bevorzugen Sie?',
                    'options': ['Ausführlich', 'Knapp', 'Keine Präferenz']
                },
                {
                    'id': 'improvement',
                    'type': 'text',
                    'question': 'Was könnte verbessert werden? (optional)'
                }
            ]
        }
        
        form_file = self.feedback_dir / f"form_{form_id}.json"
        with open(form_file, 'w') as f:
            json.dump(form_data, f, indent=2)
            
        return form_id
    
    def submit_feedback(self, form_id: str, responses: Dict[str, Any]):
        """Verarbeite eingereichte Feedback-Daten"""
        form_file = self.feedback_dir / f"form_{form_id}.json"
        
        if not form_file.exists():
            return False
            
        with open(form_file, 'r') as f:
            form_data = json.load(f)
        
        # Feedback hinzufügen
        form_data['responses'] = responses
        form_data['submitted'] = datetime.now().isoformat()
        
        # Als abgeschlossen markieren
        completed_file = self.feedback_dir / f"completed_{form_id}.json"
        with open(completed_file, 'w') as f:
            json.dump(form_data, f, indent=2)
            
        # Original-Form löschen
        form_file.unlink()
        
        return True
    
    def generate_24h_retrospective(self) -> Dict[str, Any]:
        """Generiere 24h Retrospektive für Alpha-Testing"""
        end_time = datetime.now()
        start_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Log-Dateien sammeln
        interactions = []
        feedback = []
        
        for log_file in self.feedback_dir.glob("interactions_*.jsonl"):
            with open(log_file, 'r') as f:
                for line in f:
                    interaction = json.loads(line.strip())
                    interaction_time = datetime.fromisoformat(interaction['timestamp'])
                    if start_time <= interaction_time <= end_time:
                        interactions.append(interaction)
        
        for feedback_file in self.feedback_dir.glob("completed_*.json"):
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
                submitted_time = datetime.fromisoformat(feedback_data['submitted'])
                if start_time <= submitted_time <= end_time:
                    feedback.append(feedback_data)
        
        # Analyse
        total_interactions = len(interactions)
        unique_users = len(set(i['user_id'] for i in interactions))
        
        # A/B Test Ergebnisse
        variant_performance = {}
        for test_name in self.ab_tests.keys():
            variants = {}
            for interaction in interactions:
                variant = interaction['variants'].get(test_name, 'A')
                if variant not in variants:
                    variants[variant] = {'count': 0, 'avg_response_length': 0}
                variants[variant]['count'] += 1
                variants[variant]['avg_response_length'] += interaction['response_length']
            
            # Durchschnitte berechnen
            for variant in variants:
                if variants[variant]['count'] > 0:
                    variants[variant]['avg_response_length'] /= variants[variant]['count']
            
            variant_performance[test_name] = variants
        
        # Feedback-Analyse
        if feedback:
            avg_quality = sum(f['responses'].get('quality', 3) for f in feedback) / len(feedback)
            avg_accuracy = sum(f['responses'].get('accuracy', 3) for f in feedback) / len(feedback)
            
            # Top-3 Verbesserungsvorschläge
            improvements = [f['responses'].get('improvement', '') for f in feedback 
                          if f['responses'].get('improvement', '').strip()]
        else:
            avg_quality = 0
            avg_accuracy = 0
            improvements = []
        
        retrospective = {
            'period': f"{start_time.isoformat()} - {end_time.isoformat()}",
            'summary': {
                'total_interactions': total_interactions,
                'unique_users': unique_users,
                'feedback_submissions': len(feedback)
            },
            'ab_test_results': variant_performance,
            'quality_metrics': {
                'avg_quality_rating': avg_quality,
                'avg_accuracy_rating': avg_accuracy,
                'feedback_response_rate': len(feedback) / max(1, total_interactions)
            },
            'top_improvements': improvements[:3],
            'recommended_actions': []
        }
        
        # Empfehlungen basierend auf Daten
        if avg_quality < 3.5:
            retrospective['recommended_actions'].append("Qualität verbessern: Types 04/08/42 nachschärfen")
        
        if unique_users < 5:
            retrospective['recommended_actions'].append("Mehr Alpha-Tester einladen")
            
        if len(feedback) < total_interactions * 0.1:
            retrospective['recommended_actions'].append("Feedback-Prozess vereinfachen")
        
        # Speichern
        retro_file = self.feedback_dir / f"retrospective_{datetime.now().strftime('%Y%m%d')}.json"
        with open(retro_file, 'w') as f:
            json.dump(retrospective, f, indent=2)
            
        return retrospective

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='VXOR Beta Feedback System')
    parser.add_argument('--bundle-dir', required=True, help='B2C Bundle directory')
    parser.add_argument('--action', required=True, 
                       choices=['setup', 'retrospective', 'status'])
    
    args = parser.parse_args()
    
    feedback_system = BetaFeedbackSystem(args.bundle_dir)
    
    if args.action == 'setup':
        print("✅ Beta Feedback System eingerichtet")
        print(f"   Nutzer-Gruppen: {len(feedback_system.user_assignments)}")
        
    elif args.action == 'retrospective':
        retro = feedback_system.generate_24h_retrospective()
        print("📊 24h Retrospektive erstellt:")
        print(f"   Interaktionen: {retro['summary']['total_interactions']}")
        print(f"   Unique Users: {retro['summary']['unique_users']}")
        print(f"   Feedback-Rate: {retro['quality_metrics']['feedback_response_rate']:.1%}")
        
    elif args.action == 'status':
        print("📈 Beta-Testing Status:")
        print(f"   Zugewiesene Nutzer: {len(feedback_system.user_assignments)}")
        
        # Zähle Interaktionen heute
        today = datetime.now().strftime('%Y%m%d')
        interaction_file = feedback_system.feedback_dir / f"interactions_{today}.jsonl"
        if interaction_file.exists():
            with open(interaction_file, 'r') as f:
                interactions_today = len(f.readlines())
        else:
            interactions_today = 0
            
        print(f"   Interaktionen heute: {interactions_today}")

if __name__ == "__main__":
    main()
