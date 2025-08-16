#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Deep-State-Netzwerkanalyse

Dieses Modul implementiert die Netzwerkanalyse für das Deep-State-Modul.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import logging
import datetime
import hashlib
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.analysis.deep_state_network")

@dataclass
class NetworkNode:
    """Netzwerkknoten für die Deep-State-Analyse"""
    node_id: str
    node_type: str
    node_name: str
    node_description: str
    trust_score: float
    first_seen: datetime.datetime
    last_seen: datetime.datetime
    context_clusters: Set[str] = field(default_factory=set)
    connections: Dict[str, float] = field(default_factory=dict)
    
    def add_connection(self, target_node_id: str, connection_strength: float) -> None:
        """
        Fügt eine Verbindung zu einem anderen Knoten hinzu
        
        Args:
            target_node_id: ID des Zielknotens
            connection_strength: Stärke der Verbindung (0-1)
        """
        self.connections[target_node_id] = connection_strength
    
    def update_last_seen(self) -> None:
        """Aktualisiert den Zeitpunkt des letzten Sehens"""
        self.last_seen = datetime.datetime.now()
    
    def add_context_cluster(self, context_cluster: str) -> None:
        """
        Fügt ein Kontext-Cluster hinzu
        
        Args:
            context_cluster: Hinzuzufügendes Kontext-Cluster
        """
        self.context_clusters.add(context_cluster)
    
    def get_connection_strength(self, target_node_id: str) -> float:
        """
        Gibt die Stärke der Verbindung zu einem anderen Knoten zurück
        
        Args:
            target_node_id: ID des Zielknotens
            
        Returns:
            Stärke der Verbindung (0-1)
        """
        return self.connections.get(target_node_id, 0.0)
    
    def get_total_connection_strength(self) -> float:
        """
        Gibt die Gesamtstärke aller Verbindungen zurück
        
        Returns:
            Gesamtstärke aller Verbindungen
        """
        if not self.connections:
            return 0.0
        
        return sum(self.connections.values()) / len(self.connections)
    
    def is_in_context(self, context_cluster: str) -> bool:
        """
        Überprüft, ob der Knoten in einem bestimmten Kontext-Cluster ist
        
        Args:
            context_cluster: Zu überprüfendes Kontext-Cluster
            
        Returns:
            True, wenn der Knoten im Kontext-Cluster ist, sonst False
        """
        if not context_cluster:
            return True
        
        # Extrahiere Tags aus dem Kontext-Cluster
        context_tags = set(tag.strip().lower() for tag in context_cluster.split(","))
        
        # Extrahiere Tags aus den Kontext-Clustern des Knotens
        node_tags = set()
        for cluster in self.context_clusters:
            node_tags.update(tag.strip().lower() for tag in cluster.split(","))
        
        # Überprüfe, ob mindestens ein Tag übereinstimmt
        return bool(context_tags.intersection(node_tags))

class NetworkAnalyzer:
    """
    Netzwerkanalyse für das Deep-State-Modul
    
    Diese Klasse implementiert die Netzwerkanalyse für das Deep-State-Modul.
    Sie verfolgt Verbindungen zwischen Quellen und berechnet einen Score basierend
    auf den gefundenen Verbindungen.
    """
    
    def __init__(self):
        """Initialisiert den NetworkAnalyzer"""
        self.nodes = {}
        self._load_default_nodes()
        logger.info(f"NetworkAnalyzer mit {len(self.nodes)} Knoten initialisiert")
    
    def _load_default_nodes(self) -> None:
        """Lädt die Standardknoten"""
        # In einer realen Implementierung würden diese Knoten aus einer Datenbank geladen
        # Hier fügen wir einige Beispielknoten hinzu
        now = datetime.datetime.now()
        
        # Beispielknoten für Medien
        self.add_node(
            node_id="MEDIA_001",
            node_type="MEDIA",
            node_name="Globale Nachrichten AG",
            node_description="Großes Medienunternehmen mit globaler Reichweite",
            trust_score=0.7,
            first_seen=now,
            last_seen=now,
            context_clusters={"medien,nachrichten,global", "politik,wirtschaft"}
        )
        
        self.add_node(
            node_id="MEDIA_002",
            node_type="MEDIA",
            node_name="Freie Presse GmbH",
            node_description="Unabhängiges Medienunternehmen",
            trust_score=0.8,
            first_seen=now,
            last_seen=now,
            context_clusters={"medien,nachrichten,unabhängig", "politik,gesellschaft"}
        )
        
        # Beispielknoten für Organisationen
        self.add_node(
            node_id="ORG_001",
            node_type="ORGANIZATION",
            node_name="Weltforum für Wirtschaft",
            node_description="Internationale Organisation für wirtschaftliche Zusammenarbeit",
            trust_score=0.6,
            first_seen=now,
            last_seen=now,
            context_clusters={"wirtschaft,global,forum", "politik,international"}
        )
        
        self.add_node(
            node_id="ORG_002",
            node_type="ORGANIZATION",
            node_name="Stiftung für globale Zukunft",
            node_description="Stiftung zur Förderung globaler Initiativen",
            trust_score=0.5,
            first_seen=now,
            last_seen=now,
            context_clusters={"stiftung,global,zukunft", "politik,gesellschaft,wirtschaft"}
        )
        
        # Beispielknoten für Personen
        self.add_node(
            node_id="PERSON_001",
            node_type="PERSON",
            node_name="Dr. Max Mustermann",
            node_description="Bekannter Experte für internationale Beziehungen",
            trust_score=0.7,
            first_seen=now,
            last_seen=now,
            context_clusters={"person,experte,international", "politik,beziehungen"}
        )
        
        # Beispielverbindungen
        self.add_connection("MEDIA_001", "ORG_001", 0.8)
        self.add_connection("MEDIA_001", "PERSON_001", 0.6)
        self.add_connection("ORG_001", "ORG_002", 0.7)
        self.add_connection("PERSON_001", "ORG_002", 0.5)
    
    def add_node(self, 
                node_id: str, 
                node_type: str, 
                node_name: str, 
                node_description: str, 
                trust_score: float,
                first_seen: datetime.datetime,
                last_seen: datetime.datetime,
                context_clusters: Optional[Set[str]] = None) -> None:
        """
        Fügt einen neuen Knoten hinzu
        
        Args:
            node_id: ID des Knotens
            node_type: Typ des Knotens
            node_name: Name des Knotens
            node_description: Beschreibung des Knotens
            trust_score: Vertrauensbewertung des Knotens
            first_seen: Zeitpunkt des ersten Sehens
            last_seen: Zeitpunkt des letzten Sehens
            context_clusters: Kontext-Cluster des Knotens
        """
        if node_id in self.nodes:
            logger.warning(f"Knoten mit ID {node_id} existiert bereits und wird überschrieben")
        
        self.nodes[node_id] = NetworkNode(
            node_id=node_id,
            node_type=node_type,
            node_name=node_name,
            node_description=node_description,
            trust_score=trust_score,
            first_seen=first_seen,
            last_seen=last_seen,
            context_clusters=context_clusters or set()
        )
        
        logger.info(f"Knoten {node_id} hinzugefügt")
    
    def remove_node(self, node_id: str) -> bool:
        """
        Entfernt einen Knoten
        
        Args:
            node_id: ID des zu entfernenden Knotens
            
        Returns:
            True, wenn der Knoten entfernt wurde, sonst False
        """
        if node_id not in self.nodes:
            logger.warning(f"Knoten mit ID {node_id} nicht gefunden")
            return False
        
        # Entferne Verbindungen zu diesem Knoten
        for other_node_id, node in self.nodes.items():
            if node_id in node.connections:
                del node.connections[node_id]
        
        # Entferne den Knoten
        del self.nodes[node_id]
        logger.info(f"Knoten {node_id} entfernt")
        
        return True
    
    def add_connection(self, source_node_id: str, target_node_id: str, connection_strength: float) -> bool:
        """
        Fügt eine Verbindung zwischen zwei Knoten hinzu
        
        Args:
            source_node_id: ID des Quellknotens
            target_node_id: ID des Zielknotens
            connection_strength: Stärke der Verbindung (0-1)
            
        Returns:
            True, wenn die Verbindung hinzugefügt wurde, sonst False
        """
        if source_node_id not in self.nodes:
            logger.warning(f"Quellknoten mit ID {source_node_id} nicht gefunden")
            return False
        
        if target_node_id not in self.nodes:
            logger.warning(f"Zielknoten mit ID {target_node_id} nicht gefunden")
            return False
        
        # Füge Verbindung in beide Richtungen hinzu
        self.nodes[source_node_id].add_connection(target_node_id, connection_strength)
        self.nodes[target_node_id].add_connection(source_node_id, connection_strength)
        
        logger.info(f"Verbindung zwischen {source_node_id} und {target_node_id} mit Stärke {connection_strength:.2f} hinzugefügt")
        
        return True
    
    def get_node(self, node_id: str) -> Optional[NetworkNode]:
        """
        Gibt einen Knoten zurück
        
        Args:
            node_id: ID des Knotens
            
        Returns:
            Knoten oder None, wenn der Knoten nicht gefunden wurde
        """
        return self.nodes.get(node_id)
    
    def get_node_count(self) -> int:
        """
        Gibt die Anzahl der Knoten zurück
        
        Returns:
            Anzahl der Knoten
        """
        return len(self.nodes)
    
    def analyze_network(self, source_id: str, context_cluster: str = "") -> float:
        """
        Analysiert das Netzwerk für eine bestimmte Quelle
        
        Args:
            source_id: ID der Quelle
            context_cluster: Kontext-Cluster für die Filterung der Knoten
            
        Returns:
            Score zwischen 0 und 1 basierend auf den gefundenen Verbindungen
        """
        # Überprüfe, ob die Quelle als Knoten existiert
        if source_id not in self.nodes:
            logger.info(f"Quelle mit ID {source_id} nicht als Knoten gefunden, erstelle temporären Knoten")
            return 0.0
        
        # Aktualisiere den Zeitpunkt des letzten Sehens
        self.nodes[source_id].update_last_seen()
        
        # Filtere Knoten basierend auf dem Kontext-Cluster
        relevant_nodes = self._filter_nodes_by_context(context_cluster)
        
        if not relevant_nodes:
            logger.warning(f"Keine relevanten Knoten für Kontext-Cluster {context_cluster} gefunden")
            return 0.0
        
        # Berechne den Score basierend auf den Verbindungen
        source_node = self.nodes[source_id]
        
        # Direkte Verbindungen
        direct_connections = [
            (target_id, strength)
            for target_id, strength in source_node.connections.items()
            if target_id in relevant_nodes
        ]
        
        # Indirekte Verbindungen (2. Ordnung)
        indirect_connections = []
        for direct_target_id, direct_strength in direct_connections:
            direct_target = self.nodes[direct_target_id]
            for indirect_target_id, indirect_strength in direct_target.connections.items():
                if indirect_target_id != source_id and indirect_target_id in relevant_nodes:
                    # Berechne die Stärke der indirekten Verbindung
                    combined_strength = direct_strength * indirect_strength * 0.5
                    indirect_connections.append((indirect_target_id, combined_strength))
        
        # Kombiniere direkte und indirekte Verbindungen
        all_connections = direct_connections + indirect_connections
        
        if not all_connections:
            return 0.0
        
        # Berechne den Score
        connection_score = sum(strength for _, strength in all_connections) / len(all_connections)
        
        # Gewichte den Score basierend auf der Anzahl der Verbindungen
        connection_count_factor = min(1.0, len(all_connections) / 10)
        
        # Berechne den finalen Score
        final_score = connection_score * connection_count_factor
        
        logger.info(f"Netzwerk-Score für Quelle {source_id}: {final_score:.2f}")
        
        return final_score
    
    def _filter_nodes_by_context(self, context_cluster: str) -> Set[str]:
        """
        Filtert Knoten basierend auf dem Kontext-Cluster
        
        Args:
            context_cluster: Kontext-Cluster für die Filterung der Knoten
            
        Returns:
            Set von Knoten-IDs, die dem Kontext-Cluster entsprechen
        """
        if not context_cluster:
            return set(self.nodes.keys())
        
        # Filtere Knoten basierend auf dem Kontext-Cluster
        return {
            node_id
            for node_id, node in self.nodes.items()
            if node.is_in_context(context_cluster)
        }
    
    def get_potential_connections(self, source_id: str, context_cluster: str = "") -> List[str]:
        """
        Gibt eine Liste potenzieller Verbindungen zurück
        
        Args:
            source_id: ID der Quelle
            context_cluster: Kontext-Cluster für die Filterung der Knoten
            
        Returns:
            Liste potenzieller Verbindungen als Strings
        """
        # Überprüfe, ob die Quelle als Knoten existiert
        if source_id not in self.nodes:
            return []
        
        # Filtere Knoten basierend auf dem Kontext-Cluster
        relevant_nodes = self._filter_nodes_by_context(context_cluster)
        
        if not relevant_nodes:
            return []
        
        # Berechne potenzielle Verbindungen
        source_node = self.nodes[source_id]
        
        # Direkte Verbindungen
        direct_connections = [
            (target_id, strength)
            for target_id, strength in source_node.connections.items()
            if target_id in relevant_nodes
        ]
        
        # Sortiere nach Stärke
        direct_connections.sort(key=lambda x: x[1], reverse=True)
        
        # Erstelle Beschreibungen
        connection_descriptions = []
        
        for target_id, strength in direct_connections[:5]:  # Begrenze auf die Top 5
            target_node = self.nodes[target_id]
            connection_descriptions.append(
                f"{target_node.node_name} ({target_node.node_type}): Verbindungsstärke {strength:.2f}"
            )
        
        return connection_descriptions
    
    def export_network(self) -> Dict[str, Any]:
        """
        Exportiert das Netzwerk als Wörterbuch
        
        Returns:
            Wörterbuch mit dem Netzwerk
        """
        return {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "node_name": node.node_name,
                    "node_description": node.node_description,
                    "trust_score": node.trust_score,
                    "first_seen": node.first_seen.isoformat(),
                    "last_seen": node.last_seen.isoformat(),
                    "context_clusters": list(node.context_clusters),
                    "connections": node.connections
                }
                for node in self.nodes.values()
            ]
        }
    
    def import_network(self, network_data: Dict[str, Any]) -> None:
        """
        Importiert ein Netzwerk aus einem Wörterbuch
        
        Args:
            network_data: Wörterbuch mit dem Netzwerk
        """
        if "nodes" not in network_data:
            logger.error("Ungültiges Netzwerk-Datenformat: 'nodes' nicht gefunden")
            return
        
        # Lösche vorhandenes Netzwerk
        self.nodes = {}
        
        # Importiere Knoten
        for node_data in network_data["nodes"]:
            try:
                node_id = node_data["node_id"]
                self.nodes[node_id] = NetworkNode(
                    node_id=node_id,
                    node_type=node_data["node_type"],
                    node_name=node_data["node_name"],
                    node_description=node_data["node_description"],
                    trust_score=node_data["trust_score"],
                    first_seen=datetime.datetime.fromisoformat(node_data["first_seen"]),
                    last_seen=datetime.datetime.fromisoformat(node_data["last_seen"]),
                    context_clusters=set(node_data.get("context_clusters", [])),
                    connections=node_data.get("connections", {})
                )
            except (KeyError, ValueError) as e:
                logger.error(f"Fehler beim Importieren des Knotens: {e}")
        
        logger.info(f"{len(self.nodes)} Knoten importiert")
