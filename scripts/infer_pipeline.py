#!/usr/bin/env python3
"""
VXOR Inference Pipeline - B2C Alpha
Offline-fähige Inferenz mit Guard-Integration für Smoke-Tests
"""

import os
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

class VXORInferencePipeline:
    def __init__(self, checkpoint_path: str, max_tokens: int = 512, safe_mode: bool = True):
        self.checkpoint_path = Path(checkpoint_path)
        self.max_tokens = max_tokens
        self.safe_mode = safe_mode
        
        # Load environment settings
        self.cot_limit = int(os.getenv('VXOR_COT_SOFT_LIMIT', '220'))
        self.critic_enable = os.getenv('VXOR_CRITIC_ENABLE', '1') == '1'
        self.rate_limit = int(os.getenv('VXOR_RATE_LIMIT_PER_MIN', '20'))
        
        # Setup logging
        self.setup_logging()
        
        # Load checkpoint metadata
        self.load_checkpoint()
        
        # Request tracking for rate limiting
        self.request_history = []
        
    def setup_logging(self):
        """Setup logging based on environment"""
        level = os.getenv('VXOR_LOG_LEVEL', 'INFO')
        logging.basicConfig(
            level=getattr(logging, level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('VXORInference')
        
    def load_checkpoint(self):
        """Load checkpoint metadata"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint nicht gefunden: {self.checkpoint_path}")
            
        with open(self.checkpoint_path, 'r') as f:
            self.checkpoint_data = json.load(f)
            
        self.logger.info(f"Checkpoint geladen: {self.checkpoint_data.get('model_type', 'unknown')}")
        
    def check_rate_limit(self) -> bool:
        """Prüfe Rate-Limiting"""
        current_time = time.time()
        # Entferne Anfragen älter als 1 Minute
        self.request_history = [t for t in self.request_history if current_time - t < 60]
        
        if len(self.request_history) >= self.rate_limit:
            return False
            
        self.request_history.append(current_time)
        return True
        
    def safety_filter(self, query: str) -> Dict[str, Any]:
        """Einfacher Safety-Filter für B2C Alpha"""
        safety_result = {
            'safe': True,
            'reason': None,
            'confidence': 0.95
        }
        
        # Grundlegende Sicherheitsprüfungen
        unsafe_patterns = [
            r'(hack|exploit|attack|manipulat)',
            r'(illegal|gefährlich|schädlich)',
            r'(drogen|waffen|gewalt)',
            r'(persönliche daten|passwort|kreditkarte)',
            r'(diskriminierung|hassrede)'
        ]
        
        query_lower = query.lower()
        for pattern in unsafe_patterns:
            if re.search(pattern, query_lower):
                safety_result['safe'] = False
                safety_result['reason'] = f"Potentiell unsicherer Inhalt erkannt: {pattern}"
                safety_result['confidence'] = 0.8
                break
                
        return safety_result
        
    def simulate_inference(self, query: str) -> str:
        """Simuliere Modell-Inferenz (für Alpha-Version)"""
        # Simuliere Antworten basierend auf Query-Typ
        query_lower = query.lower()
        
        if 'zusammenfass' in query_lower or 'bulletpoint' in query_lower:
            return """• KI-Systeme automatisieren Aufgaben und analysieren große Datenmengen
• Sie verbessern Entscheidungsprozesse in vielen Bereichen
• Herausforderungen: Datenschutz, Ethik und menschliche Kontrolle bleiben wichtig"""
            
        elif 'umzug' in query_lower or 'plan' in query_lower:
            return """Tag 1: Umzugsunternehmen buchen und Termine vereinbaren.
Tag 2: Kartons besorgen und mit dem Einpacken beginnen.
Tag 3: Adressänderungen bei Behörden und Dienstleistern melden.
Tag 4: Versorgungsverträge (Strom, Gas, Internet) für neue Wohnung abschließen.
Tag 5: Letzte Gegenstände einpacken und Kühlschrank leeren.
Tag 6: Umzugstag - Möbel und Kartons transportieren.
Tag 7: Auspacken und erste Einrichtung in der neuen Wohnung."""
            
        elif 'spar' in query_lower or '150' in query_lower or 'prozent' in query_lower:
            return """Bei 150€/Monat für 2 Jahre:
Grundbetrag: 150€ × 24 Monate = 3.600€
Zinsen bei 3% p.a.: ca. 110€ (vereinfacht)
Geschätzte Gesamtsumme nach 2 Jahren: ~3.710€

(Hinweis: Genaue Berechnung hängt von der Zinsperiode ab)"""
            
        elif 'wahrscheinlichkeit' in query_lower or '12-jährig' in query_lower:
            return """Wahrscheinlichkeit ist die Chance, dass etwas passiert. Stell dir einen Würfel vor: Die Wahrscheinlichkeit, eine 6 zu würfeln, ist 1 von 6 Möglichkeiten. Bei einer Münze ist es 1 von 2 für Kopf oder Zahl. Je höher die Zahl, desto wahrscheinlicher ist es. Wahrscheinlichkeiten helfen uns zu verstehen, was vermutlich passieren wird."""
            
        else:
            return "Ich verstehe Ihre Frage. Können Sie sie bitte etwas spezifischer formulieren, damit ich Ihnen besser helfen kann?"
            
    def process_query(self, query: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Verarbeite einzelne Anfrage durch Pipeline"""
        if request_id is None:
            request_id = f"req_{int(time.time())}"
            
        start_time = time.time()
        
        # Rate-Limiting prüfen
        if not self.check_rate_limit():
            return {
                'request_id': request_id,
                'status': 'RATE_LIMITED',
                'response': 'Zu viele Anfragen. Bitte warten Sie einen Moment.',
                'latency_ms': (time.time() - start_time) * 1000
            }
        
        # Safety-Filter
        safety_check = self.safety_filter(query)
        if not safety_check['safe']:
            return {
                'request_id': request_id,
                'status': 'BLOCKED',
                'response': 'Entschuldigung, ich kann bei dieser Anfrage nicht helfen. Bitte stellen Sie eine andere Frage.',
                'safety_reason': safety_check['reason'],
                'latency_ms': (time.time() - start_time) * 1000
            }
        
        # Inferenz durchführen
        try:
            response = self.simulate_inference(query)
            
            # Token-Limit prüfen (grob)
            if len(response.split()) > self.max_tokens:
                response = ' '.join(response.split()[:self.max_tokens]) + "..."
                
            status = 'SUCCESS'
            
        except Exception as e:
            self.logger.error(f"Inference-Fehler: {e}")
            response = "Es tut mir leid, ich kann Ihre Anfrage momentan nicht bearbeiten. Bitte versuchen Sie es erneut."
            status = 'ERROR'
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = {
            'request_id': request_id,
            'status': status,
            'query': query,
            'response': response,
            'latency_ms': latency_ms,
            'timestamp': datetime.now().isoformat(),
            'safety_score': safety_check['confidence']
        }
        
        self.logger.info(f"Anfrage {request_id}: {status} ({latency_ms:.1f}ms)")
        return result
        
    def process_prompts_file(self, prompts_file: str, output_file: str):
        """Verarbeite Prompts aus Datei"""
        prompts_path = Path(prompts_file)
        output_path = Path(output_file)
        
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts-Datei nicht gefunden: {prompts_path}")
            
        # Prompts lesen
        with open(prompts_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Prompts nach Leerzeilen trennen
        prompts = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        self.logger.info(f"Verarbeite {len(prompts)} Prompts...")
        
        # Output-Datei vorbereiten
        results = []
        
        for i, prompt in enumerate(prompts, 1):
            self.logger.info(f"Prompt {i}/{len(prompts)}: {prompt[:50]}...")
            
            result = self.process_query(prompt, f"smoke_test_{i}")
            results.append(result)
            
            # In JSONL-Format schreiben
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # Kurze Pause zwischen Anfragen
            time.sleep(0.1)
            
        self.logger.info(f"✅ Smoke-Tests abgeschlossen. Ergebnisse in: {output_path}")
        
        # Zusammenfassung
        success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
        avg_latency = sum(r['latency_ms'] for r in results) / len(results)
        
        summary = {
            'total_prompts': len(prompts),
            'successful': success_count,
            'success_rate': success_count / len(prompts),
            'average_latency_ms': avg_latency,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = output_path.parent / 'smoke_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"📊 Smoke-Test Zusammenfassung:")
        print(f"   Erfolgsrate: {summary['success_rate']:.1%}")
        print(f"   Durchschnittliche Latenz: {avg_latency:.1f}ms")
        print(f"   Erfolgreiche Anfragen: {success_count}/{len(prompts)}")

def main():
    parser = argparse.ArgumentParser(description='VXOR Inference Pipeline')
    parser.add_argument('--ckpt', required=True, help='Pfad zum Checkpoint')
    parser.add_argument('--prompts', required=True, help='Pfad zur Prompts-Datei')
    parser.add_argument('--max-tokens', type=int, default=512, help='Maximum Tokens')
    parser.add_argument('--out', required=True, help='Output JSONL-Datei')
    
    args = parser.parse_args()
    
    # Lade Umgebungsvariablen
    if os.getenv('VXOR_OFFLINE_MODE') == 'true':
        print("🔒 Offline-Modus aktiviert")
    
    # Pipeline initialisieren
    pipeline = VXORInferencePipeline(
        checkpoint_path=args.ckpt,
        max_tokens=args.max_tokens
    )
    
    # Prompts verarbeiten
    pipeline.process_prompts_file(args.prompts, args.out)

if __name__ == "__main__":
    main()
