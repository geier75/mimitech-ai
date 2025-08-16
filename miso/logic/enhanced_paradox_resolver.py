#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enhanced Paradox Resolver für MISO Ultimate

Erweiterte Paradoxauflösung mit automatischer Strategieauswahl
für komplexe, mehrstufige Paradoxien.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional
import numpy as np

# MISO-Module
from miso.math.t_mathematics.engine import TMathEngine
from miso.logic.qlogik_integration import QLOGIKIntegrationManager
from miso.logic.enhanced_paradox_detector import (
    EnhancedParadoxDetector, ResolutionStrategy, ParadoxType
)

# Logger konfigurieren
logger = logging.getLogger("MISO.Logic.EnhancedParadoxResolver")

class EnhancedParadoxResolver:
    """Erweiterte Paradoxauflösung mit automatischer Strategieauswahl"""
    
    def __init__(self, tmath_engine: Optional[TMathEngine] = None, qlogik_manager: Optional[QLOGIKIntegrationManager] = None):
        self.tmath_engine = tmath_engine or TMathEngine()
        self.qlogik_manager = qlogik_manager
        self.resolution_success_rate = {
            ResolutionStrategy.TIMELINE_BRANCHING: 0.85,
            ResolutionStrategy.CAUSAL_LOOP_BREAKING: 0.75,
            ResolutionStrategy.INFORMATION_ISOLATION: 0.70,
            ResolutionStrategy.QUANTUM_DECOHERENCE: 0.90,
            ResolutionStrategy.LOGICAL_REFRAMING: 0.60,
            ResolutionStrategy.TEMPORAL_QUARANTINE: 0.95
        }
        
        logger.info("Enhanced Paradox Resolver initialisiert")
    
    def resolve_paradox(self, timeline_data: Dict[str, Any], paradox_signature: Optional[Dict] = None) -> Dict[str, Any]:
        """Hauptmethode für Paradoxauflösung"""
        try:
            if not paradox_signature:
                # Erkenne Paradox zuerst
                detector = EnhancedParadoxDetector(self.tmath_engine)
                detection_result = detector.detect_paradox(timeline_data)
                
                if not detection_result.get('paradox_detected', False):
                    return {'resolved': False, 'reason': 'no_paradox_detected'}
                
                paradox_signature = detection_result.get('primary_signature', {})
            
            # Wähle optimale Auflösungsstrategie
            strategy = self._select_optimal_strategy(paradox_signature)
            
            # Führe Auflösung durch
            resolution_result = self._execute_resolution_strategy(
                timeline_data, paradox_signature, strategy
            )
            
            # Validiere Auflösung
            validation_result = self._validate_resolution(
                timeline_data, resolution_result
            )
            
            return {
                'resolved': validation_result['valid'],
                'strategy': strategy.value,
                'resolution_data': resolution_result,
                'validation': validation_result,
                'confidence': resolution_result.get('confidence', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Paradoxauflösung: {e}")
            return {'resolved': False, 'error': str(e)}
    
    def _select_optimal_strategy(self, paradox_signature: Dict) -> ResolutionStrategy:
        """Wähle optimale Auflösungsstrategie basierend auf Erfolgsrate und Komplexität"""
        candidates = paradox_signature.get('resolution_candidates', [])
        
        if not candidates:
            return ResolutionStrategy.TIMELINE_BRANCHING  # Fallback
        
        # Wähle Strategie basierend auf Erfolgsrate und Komplexität
        best_strategy = None
        best_score = 0.0
        
        complexity = paradox_signature.get('complexity', 1)
        
        for candidate in candidates:
            try:
                strategy = ResolutionStrategy(candidate)
                success_rate = self.resolution_success_rate.get(strategy, 0.5)
                complexity_penalty = complexity * 0.05  # Reduziere Score bei höherer Komplexität
                score = success_rate - complexity_penalty
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
            except ValueError:
                continue
        
        return best_strategy or ResolutionStrategy.TIMELINE_BRANCHING
    
    def _execute_resolution_strategy(self, timeline_data: Dict, paradox_signature: Dict, strategy: ResolutionStrategy) -> Dict[str, Any]:
        """Führe Auflösungsstrategie aus"""
        strategy_methods = {
            ResolutionStrategy.TIMELINE_BRANCHING: self._resolve_via_timeline_branching,
            ResolutionStrategy.CAUSAL_LOOP_BREAKING: self._resolve_via_causal_loop_breaking,
            ResolutionStrategy.INFORMATION_ISOLATION: self._resolve_via_information_isolation,
            ResolutionStrategy.QUANTUM_DECOHERENCE: self._resolve_via_quantum_decoherence,
            ResolutionStrategy.TEMPORAL_QUARANTINE: self._resolve_via_temporal_quarantine,
            ResolutionStrategy.LOGICAL_REFRAMING: self._resolve_via_logical_reframing
        }
        
        method = strategy_methods.get(strategy, self._resolve_via_logical_reframing)
        return method(timeline_data, paradox_signature)
    
    def _resolve_via_timeline_branching(self, timeline_data: Dict, paradox_signature: Dict) -> Dict[str, Any]:
        """Auflösung durch Timeline-Verzweigung"""
        events = timeline_data.get('events', [])
        
        # Identifiziere problematische Events
        paradox_events = self._identify_paradox_events(events, paradox_signature)
        
        # Erstelle alternative Timeline-Zweige
        branches = []
        for i, paradox_event in enumerate(paradox_events):
            branch = {
                'branch_id': f'branch_{i}',
                'events': [e for e in events if e != paradox_event],
                'modified_event': self._create_alternative_event(paradox_event),
                'probability': 1.0 / max(len(paradox_events), 1)
            }
            branches.append(branch)
        
        # Falls keine spezifischen Paradox-Events gefunden, erstelle Standard-Branches
        if not branches:
            branches = [
                {
                    'branch_id': 'branch_0',
                    'events': events,
                    'modification': 'timeline_split_at_paradox_point',
                    'probability': 1.0
                }
            ]
        
        return {
            'method': 'timeline_branching',
            'branches': branches,
            'confidence': 0.8,
            'paradox_resolved': True
        }
    
    def _resolve_via_causal_loop_breaking(self, timeline_data: Dict, paradox_signature: Dict) -> Dict[str, Any]:
        """Auflösung durch Durchbrechen kausaler Schleifen"""
        events = timeline_data.get('events', [])
        
        # Identifiziere und breche kausale Verbindungen
        modified_events = []
        broken_connections = []
        
        for event in events:
            modified_event = event.copy()
            
            # Entferne problematische kausale Abhängigkeiten
            if event.get('based_on'):
                broken_connections.append({
                    'event': event.get('action', 'unknown'),
                    'removed_dependency': event['based_on']
                })
                del modified_event['based_on']
            
            # Markiere Event als kausal isoliert
            modified_event['causal_isolation'] = True
            modified_events.append(modified_event)
        
        return {
            'method': 'causal_loop_breaking',
            'modified_events': modified_events,
            'broken_connections': broken_connections,
            'confidence': 0.75,
            'paradox_resolved': len(broken_connections) > 0
        }
    
    def _resolve_via_information_isolation(self, timeline_data: Dict, paradox_signature: Dict) -> Dict[str, Any]:
        """Auflösung durch Informationsisolation"""
        events = timeline_data.get('events', [])
        
        isolated_events = []
        quarantined_info = []
        
        for event in events:
            # Identifiziere informationsbasierte Events
            if any(keyword in str(event.get('action', '')).lower() 
                   for keyword in ['formula', 'information', 'knowledge', 'data']):
                
                quarantined_info.append({
                    'original_event': event,
                    'isolation_method': 'temporal_compartment',
                    'access_restrictions': ['read_only', 'temporal_locked']
                })
                
                # Erstelle isolierte Version
                isolated_event = event.copy()
                isolated_event['isolated'] = True
                isolated_event['access_restricted'] = True
                isolated_event['information_quarantined'] = True
                isolated_events.append(isolated_event)
            else:
                isolated_events.append(event)
        
        return {
            'method': 'information_isolation',
            'isolated_events': isolated_events,
            'quarantined_information': quarantined_info,
            'confidence': 0.7,
            'paradox_resolved': len(quarantined_info) > 0
        }
    
    def _resolve_via_quantum_decoherence(self, timeline_data: Dict, paradox_signature: Dict) -> Dict[str, Any]:
        """Auflösung durch Quantendekohärenz"""
        events = timeline_data.get('events', [])
        
        collapsed_events = []
        decoherence_operations = []
        
        for event in events:
            if event.get('state') == 'superposition':
                # Kollabiere Superposition zu definitivem Zustand
                collapsed_state = 'collapsed_state_a' if np.random.random() > 0.5 else 'collapsed_state_b'
                
                collapsed_event = event.copy()
                collapsed_event['state'] = collapsed_state
                collapsed_event['decoherence_applied'] = True
                collapsed_event['original_superposition'] = event.get('state')
                collapsed_events.append(collapsed_event)
                
                decoherence_operations.append({
                    'original_state': 'superposition',
                    'collapsed_to': collapsed_state,
                    'probability': 0.5,
                    'decoherence_time': event.get('time', 0)
                })
            else:
                collapsed_events.append(event)
        
        return {
            'method': 'quantum_decoherence',
            'collapsed_events': collapsed_events,
            'decoherence_operations': decoherence_operations,
            'confidence': 0.85,
            'paradox_resolved': len(decoherence_operations) > 0
        }
    
    def _resolve_via_temporal_quarantine(self, timeline_data: Dict, paradox_signature: Dict) -> Dict[str, Any]:
        """Auflösung durch temporale Quarantäne"""
        events = timeline_data.get('events', [])
        
        quarantine_zones = []
        safe_events = []
        
        # Identifiziere problematische Zeitbereiche
        paradox_events = self._identify_paradox_events(events, paradox_signature)
        
        for event in events:
            if event in paradox_events or event.get('paradox', False):
                # Erstelle Quarantäne-Zone um problematisches Event
                quarantine_zones.append({
                    'event': event,
                    'quarantine_start': event.get('time', 0) - 10,
                    'quarantine_end': event.get('time', 0) + 10,
                    'access_level': 'restricted',
                    'containment_protocol': 'temporal_isolation',
                    'monitoring_required': True
                })
            else:
                safe_events.append(event)
        
        return {
            'method': 'temporal_quarantine',
            'safe_events': safe_events,
            'quarantine_zones': quarantine_zones,
            'confidence': 0.9,
            'paradox_resolved': len(quarantine_zones) > 0
        }
    
    def _resolve_via_logical_reframing(self, timeline_data: Dict, paradox_signature: Dict) -> Dict[str, Any]:
        """Auflösung durch logische Neuformulierung"""
        events = timeline_data.get('events', [])
        
        reframed_events = []
        reframing_operations = []
        
        for event in events:
            reframed_event = event.copy()
            
            # Wende logische Frameworks an
            if event.get('paradox', False) or self._is_problematic_event(event):
                reframed_event['logical_framework'] = 'many_worlds_interpretation'
                reframed_event['consistency_check'] = True
                reframed_event['logical_constraints'] = ['non_contradiction', 'temporal_consistency']
                
                reframing_operations.append({
                    'event': event.get('action', 'unknown'),
                    'framework_applied': 'many_worlds_interpretation',
                    'constraints_added': ['non_contradiction', 'temporal_consistency']
                })
            
            reframed_events.append(reframed_event)
        
        return {
            'method': 'logical_reframing',
            'reframed_events': reframed_events,
            'reframing_operations': reframing_operations,
            'confidence': 0.6,
            'paradox_resolved': len(reframing_operations) > 0
        }
    
    def _validate_resolution(self, original_data: Dict, resolution_result: Dict) -> Dict[str, Any]:
        """Validiere Paradoxauflösung"""
        try:
            method = resolution_result.get('method', 'unknown')
            confidence = resolution_result.get('confidence', 0.0)
            resolved = resolution_result.get('paradox_resolved', False)
            
            # Berechne Validierungs-Score basierend auf verschiedenen Faktoren
            validation_factors = {
                'method_reliability': self.resolution_success_rate.get(
                    ResolutionStrategy(method) if method in [s.value for s in ResolutionStrategy] else None, 0.5
                ),
                'resolution_confidence': confidence,
                'paradox_resolved_flag': 1.0 if resolved else 0.0
            }
            
            validation_score = (
                validation_factors['method_reliability'] * 0.4 +
                validation_factors['resolution_confidence'] * 0.4 +
                validation_factors['paradox_resolved_flag'] * 0.2
            )
            
            # Zusätzliche Validierungen basierend auf Methode
            method_specific_validation = self._validate_method_specific(method, resolution_result)
            
            final_score = validation_score * method_specific_validation['multiplier']
            
            return {
                'valid': final_score > 0.6,
                'validation_score': final_score,
                'method_used': method,
                'resolution_confidence': confidence,
                'validation_factors': validation_factors,
                'method_specific_validation': method_specific_validation
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Validierung: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _validate_method_specific(self, method: str, resolution_result: Dict) -> Dict[str, Any]:
        """Methodenspezifische Validierung"""
        if method == 'timeline_branching':
            branches = resolution_result.get('branches', [])
            return {
                'multiplier': 1.0 if len(branches) > 0 else 0.5,
                'details': f'{len(branches)} branches created'
            }
        elif method == 'causal_loop_breaking':
            broken_connections = resolution_result.get('broken_connections', [])
            return {
                'multiplier': 1.0 if len(broken_connections) > 0 else 0.3,
                'details': f'{len(broken_connections)} connections broken'
            }
        elif method == 'information_isolation':
            quarantined = resolution_result.get('quarantined_information', [])
            return {
                'multiplier': 1.0 if len(quarantined) > 0 else 0.4,
                'details': f'{len(quarantined)} information units isolated'
            }
        elif method == 'quantum_decoherence':
            decoherence_ops = resolution_result.get('decoherence_operations', [])
            return {
                'multiplier': 1.0 if len(decoherence_ops) > 0 else 0.2,
                'details': f'{len(decoherence_ops)} quantum states collapsed'
            }
        elif method == 'temporal_quarantine':
            quarantine_zones = resolution_result.get('quarantine_zones', [])
            return {
                'multiplier': 1.0 if len(quarantine_zones) > 0 else 0.1,
                'details': f'{len(quarantine_zones)} temporal zones quarantined'
            }
        else:  # logical_reframing
            reframing_ops = resolution_result.get('reframing_operations', [])
            return {
                'multiplier': 0.8 if len(reframing_ops) > 0 else 0.3,
                'details': f'{len(reframing_ops)} logical reframings applied'
            }
    
    def _identify_paradox_events(self, events: List[Dict], paradox_signature: Dict) -> List[Dict]:
        """Identifiziere problematische Events basierend auf Paradox-Signatur"""
        paradox_events = []
        
        # Basiere Identifikation auf Paradox-Typ
        paradox_type = paradox_signature.get('type', '')
        
        for event in events:
            if self._is_event_problematic_for_type(event, paradox_type):
                paradox_events.append(event)
        
        return paradox_events
    
    def _is_event_problematic_for_type(self, event: Dict, paradox_type: str) -> bool:
        """Prüfe ob Event für spezifischen Paradox-Typ problematisch ist"""
        action = str(event.get('action', '')).lower()
        
        if paradox_type == 'temporal':
            return any(keyword in action for keyword in ['time_travel', 'send_message', 'receive_message'])
        elif paradox_type == 'causal':
            return any(keyword in action for keyword in ['prevent', 'cause', 'based_on'])
        elif paradox_type == 'information':
            return any(keyword in action for keyword in ['formula', 'give', 'receive', 'discover'])
        elif paradox_type == 'quantum':
            return event.get('state') == 'superposition' or 'quantum' in action
        else:
            return event.get('paradox', False)
    
    def _create_alternative_event(self, original_event: Dict) -> Dict[str, Any]:
        """Erstelle alternatives Event für Timeline-Branching"""
        alternative = original_event.copy()
        alternative['alternative'] = True
        alternative['original_action'] = original_event.get('action', 'unknown')
        
        # Modifiziere Action basierend auf Typ
        original_action = original_event.get('action', '')
        if 'send_message' in original_action:
            alternative['action'] = 'message_not_sent'
        elif 'time_travel' in original_action:
            alternative['action'] = 'time_travel_prevented'
        elif 'prevent_meeting' in original_action:
            alternative['action'] = 'meeting_allowed'
        else:
            alternative['action'] = f'alternative_{original_action}'
        
        return alternative
    
    def _is_problematic_event(self, event: Dict) -> bool:
        """Prüfe ob Event generell problematisch ist"""
        return (event.get('paradox', False) or 
                'time_travel' in str(event.get('action', '')).lower() or
                event.get('state') == 'superposition' or
                event.get('based_on') is not None)
