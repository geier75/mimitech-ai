#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - PRISM-VisualizationEngine

Visualisierungsmodul für die PRISM-Simulation mit Unterstützung für 2D/3D-Darstellungen,
interaktive Zeitlinien und Echtzeit-Datenvisualisierung.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import json
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum, auto
import uuid
from dataclasses import dataclass, field
import threading
import queue

# Konfiguriere Logging
logger = logging.getLogger("MISO.simulation.visualization")

# Prüfen, ob Apple Silicon verfügbar ist und entsprechende Optimierungen aktivieren
is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine

# Import von internen Modulen
try:
    from miso.timeline.echo_prime import TimeNode, Timeline, Trigger, TemporalEvent
    from miso.simulation.event_generator import SimulationEvent, EventType
    HAS_DEPENDENCIES = True
except ImportError:
    logger.warning("Einige Abhängigkeiten konnten nicht importiert werden. VisualizationEngine läuft im eingeschränkten Modus.")
    HAS_DEPENDENCIES = False


class VisualizationType(Enum):
    """Typen von Visualisierungen"""
    TIMELINE_2D = auto()        # 2D-Zeitlinienvisualisierung
    TIMELINE_3D = auto()        # 3D-Zeitlinienvisualisierung
    MATRIX_2D = auto()          # 2D-Matrixvisualisierung
    MATRIX_3D = auto()          # 3D-Matrixvisualisierung
    EVENT_GRAPH = auto()        # Ereignisgraph
    PROBABILITY_HEATMAP = auto() # Wahrscheinlichkeits-Heatmap
    PARADOX_VISUALIZATION = auto() # Paradoxvisualisierung
    CUSTOM = auto()             # Benutzerdefinierte Visualisierung


@dataclass
class VisualizationConfig:
    """Konfiguration für Visualisierungen"""
    type: VisualizationType = VisualizationType.TIMELINE_2D
    title: str = "PRISM-Simulation"
    width: int = 1200
    height: int = 800
    dpi: int = 100
    background_color: str = "#1E1E1E"
    foreground_color: str = "#FFFFFF"
    accent_color: str = "#00BFFF"
    node_color: str = "#32CD32"
    edge_color: str = "#FF4500"
    timeline_colors: List[str] = field(default_factory=lambda: [
        "#FF4500", "#00BFFF", "#32CD32", "#FFD700", "#9370DB",
        "#FF69B4", "#20B2AA", "#FF6347", "#4682B4", "#FFA500"
    ])
    show_labels: bool = True
    show_legend: bool = True
    show_grid: bool = True
    interactive: bool = True
    animation_speed: float = 1.0
    max_elements: int = 1000
    custom_params: Dict[str, Any] = field(default_factory=dict)


class VisualizationEngine:
    """
    Visualisierungsmodul für die PRISM-Simulation mit Unterstützung für 2D/3D-Darstellungen,
    interaktive Zeitlinien und Echtzeit-Datenvisualisierung.
    """
    
    def __init__(self, config: Dict[str, Any] = None, prism_engine = None):
        """
        Initialisiert die VisualizationEngine
        
        Args:
            config: Konfigurationsparameter
            prism_engine: Instanz der PRISM-Engine für die Integration
        """
        self.config = config or {}
        self.default_vis_config = VisualizationConfig()
        self.active_visualizations = {}
        self.data_buffers = {}
        self.update_queues = {}
        self.animation_threads = {}
        self.prism_engine = prism_engine
        
        # Initialisiere erweiterte Visualisierungsfunktionen, wenn PRISM-Engine verfügbar ist
        if self.prism_engine is not None:
            logger.info("VisualizationEngine mit PRISM-Engine initialisiert")
            self._initialize_prism_visualization()
        self.is_running = False
        
        # Matplotlib-Konfiguration
        plt.style.use('dark_background' if self.config.get("dark_mode", True) else 'default')
        
        # Initialisiere Farbpaletten
        self.color_palettes = {
            "default": self.default_vis_config.timeline_colors,
            "probability": plt.cm.viridis(np.linspace(0, 1, 10)),
            "divergent": plt.cm.coolwarm(np.linspace(0, 1, 10)),
            "sequential": plt.cm.plasma(np.linspace(0, 1, 10)),
            "categorical": plt.cm.tab10(np.linspace(0, 1, 10)),
            "paradox": ["#FF0000", "#FF4500", "#FFA500", "#FFD700", "#ADFF2F", 
                       "#32CD32", "#00BFFF", "#0000FF", "#8A2BE2", "#FF00FF"]
        }
        
        logger.info("VisualizationEngine initialisiert")
    
    def create_visualization(self, vis_type: VisualizationType, 
                           data: Any = None,
                           config: Optional[VisualizationConfig] = None) -> str:
        """
        Erstellt eine neue Visualisierung
        
        Args:
            vis_type: Typ der Visualisierung
            data: Daten für die Visualisierung
            config: Konfiguration für die Visualisierung
            
        Returns:
            ID der erstellten Visualisierung
        """
        # Verwende Standardkonfiguration, falls keine angegeben
        if config is None:
            config = self.default_vis_config
            config.type = vis_type
        
        # Generiere eine eindeutige ID für die Visualisierung
        vis_id = str(uuid.uuid4())
        
        # Erstelle Figur und Achsen
        fig, ax = plt.subplots(figsize=(config.width / config.dpi, config.height / config.dpi), 
                              dpi=config.dpi, 
                              facecolor=config.background_color)
        
        if vis_type in [VisualizationType.TIMELINE_3D, VisualizationType.MATRIX_3D]:
            ax = fig.add_subplot(111, projection='3d')
        
        # Konfiguriere Achsen
        ax.set_title(config.title, color=config.foreground_color)
        ax.tick_params(colors=config.foreground_color)
        
        # Konfiguriere Gitter
        if config.show_grid:
            ax.grid(True, linestyle='--', alpha=0.3)
        
        # Initialisiere Daten-Buffer
        self.data_buffers[vis_id] = data if data is not None else []
        
        # Erstelle Update-Queue für Echtzeit-Updates
        self.update_queues[vis_id] = queue.Queue()
        
        # Speichere Visualisierung
        self.active_visualizations[vis_id] = {
            "type": vis_type,
            "config": config,
            "figure": fig,
            "axes": ax,
            "created_at": time.time(),
            "last_updated": time.time(),
            "plots": {}
        }
        
        # Initialisiere Visualisierung basierend auf Typ
        self._initialize_visualization(vis_id)
        
        logger.info(f"Visualisierung erstellt: {vis_id}, Typ: {vis_type.name}")
        
        return vis_id
    
    def _initialize_visualization(self, vis_id: str):
        """
        Initialisiert eine Visualisierung basierend auf ihrem Typ
        
        Args:
            vis_id: ID der Visualisierung
        """
        vis = self.active_visualizations[vis_id]
        vis_type = vis["type"]
        
        if vis_type == VisualizationType.TIMELINE_2D:
            self._init_timeline_2d(vis_id)
        elif vis_type == VisualizationType.TIMELINE_3D:
            self._init_timeline_3d(vis_id)
        elif vis_type == VisualizationType.MATRIX_2D:
            self._init_matrix_2d(vis_id)
        elif vis_type == VisualizationType.MATRIX_3D:
            self._init_matrix_3d(vis_id)
        elif vis_type == VisualizationType.EVENT_GRAPH:
            self._init_event_graph(vis_id)
        elif vis_type == VisualizationType.PROBABILITY_HEATMAP:
            self._init_probability_heatmap(vis_id)
        elif vis_type == VisualizationType.PARADOX_VISUALIZATION:
            self._init_paradox_visualization(vis_id)
    
    def _init_timeline_2d(self, vis_id: str):
        """Initialisiert eine 2D-Zeitlinienvisualisierung"""
        vis = self.active_visualizations[vis_id]
        ax = vis["axes"]
        config = vis["config"]
        
        # Setze Achsenbeschriftungen
        ax.set_xlabel("Zeit", color=config.foreground_color)
        ax.set_ylabel("Zeitlinien", color=config.foreground_color)
        
        # Initialisiere leere Plots
        vis["plots"] = {
            "timelines": [],
            "nodes": []
        }
    
    def _init_timeline_3d(self, vis_id: str):
        """Initialisiert eine 3D-Zeitlinienvisualisierung"""
        vis = self.active_visualizations[vis_id]
        ax = vis["axes"]
        config = vis["config"]
        
        # Setze Achsenbeschriftungen
        ax.set_xlabel("Zeit", color=config.foreground_color)
        ax.set_ylabel("Zeitlinien", color=config.foreground_color)
        ax.set_zlabel("Wahrscheinlichkeit", color=config.foreground_color)
        
        # Initialisiere leere Plots
        vis["plots"] = {
            "timelines": [],
            "nodes": []
        }
    
    def _init_matrix_2d(self, vis_id: str):
        """Initialisiert eine 2D-Matrixvisualisierung"""
        vis = self.active_visualizations[vis_id]
        ax = vis["axes"]
        config = vis["config"]
        
        # Setze Achsenbeschriftungen
        ax.set_xlabel("Dimension 1", color=config.foreground_color)
        ax.set_ylabel("Dimension 2", color=config.foreground_color)
        
        # Initialisiere Heatmap
        heatmap = ax.imshow(np.zeros((10, 10)), cmap='viridis', 
                           interpolation='nearest', aspect='auto')
        
        # Füge Farbbalken hinzu
        cbar = vis["figure"].colorbar(heatmap, ax=ax)
        cbar.set_label("Wert", color=config.foreground_color)
        
        # Speichere Plots
        vis["plots"] = {
            "heatmap": heatmap,
            "colorbar": cbar
        }
    
    def _init_matrix_3d(self, vis_id: str):
        """Initialisiert eine 3D-Matrixvisualisierung"""
        vis = self.active_visualizations[vis_id]
        ax = vis["axes"]
        config = vis["config"]
        
        # Setze Achsenbeschriftungen
        ax.set_xlabel("Dimension 1", color=config.foreground_color)
        ax.set_ylabel("Dimension 2", color=config.foreground_color)
        ax.set_zlabel("Dimension 3", color=config.foreground_color)
        
        # Initialisiere 3D-Scatter
        scatter = ax.scatter([], [], [], c=[], cmap='viridis', s=30, alpha=0.8)
        
        # Speichere Plots
        vis["plots"] = {
            "scatter": scatter
        }
    
    def update_visualization(self, vis_id: str, data: Any):
        """
        Aktualisiert eine Visualisierung mit neuen Daten
        
        Args:
            vis_id: ID der Visualisierung
            data: Neue Daten für die Visualisierung
        """
        if vis_id not in self.active_visualizations:
            logger.warning(f"Visualisierung {vis_id} nicht gefunden")
            return
        
        # Füge Daten zur Update-Queue hinzu
        self.update_queues[vis_id].put(data)
        
        # Aktualisiere Zeitstempel
        self.active_visualizations[vis_id]["last_updated"] = time.time()
        
        # Aktualisiere Daten direkt, wenn keine Animation läuft
        if vis_id not in self.animation_threads:
            self._update_visualization_data(vis_id, data)
    
    def start_animation(self, vis_id: str, interval: int = 100):
        """
        Startet eine Animation für eine Visualisierung
        
        Args:
            vis_id: ID der Visualisierung
            interval: Aktualisierungsintervall in Millisekunden
        """
        if vis_id not in self.active_visualizations:
            logger.warning(f"Visualisierung {vis_id} nicht gefunden")
            return
        
        vis = self.active_visualizations[vis_id]
        
        # Erstelle Animationsfunktion
        def animate(frame):
            # Verarbeite alle ausstehenden Updates
            while not self.update_queues[vis_id].empty():
                data = self.update_queues[vis_id].get()
                self._update_visualization_data(vis_id, data)
            
            return []
        
        # Erstelle Animation
        animation = FuncAnimation(
            vis["figure"], 
            animate, 
            interval=interval, 
            blit=True
        )
        
        # Speichere Animation
        vis["animation"] = animation
        
        # Starte Animationsthread
        def animation_thread():
            plt.show()
        
        thread = threading.Thread(target=animation_thread)
        thread.daemon = True
        thread.start()
        
        self.animation_threads[vis_id] = thread
        
        logger.info(f"Animation gestartet für Visualisierung {vis_id}")
    
    def save_visualization(self, vis_id: str, filepath: str, format: str = "png"):
        """
        Speichert eine Visualisierung als Datei
        
        Args:
            vis_id: ID der Visualisierung
            filepath: Pfad zur Zieldatei
            format: Dateiformat (png, pdf, svg, etc.)
        """
        if vis_id not in self.active_visualizations:
            logger.warning(f"Visualisierung {vis_id} nicht gefunden")
            return
        
        vis = self.active_visualizations[vis_id]
        config = vis["config"]
        
        # Speichere Figur
        vis["figure"].savefig(
            filepath,
            format=format,
            dpi=config.dpi,
            bbox_inches="tight",
            facecolor=config.background_color
        )
        
        logger.info(f"Visualisierung {vis_id} gespeichert als {filepath}")
    
    def close_visualization(self, vis_id: str):
        """
        Schließt eine Visualisierung
        
        Args:
            vis_id: ID der Visualisierung
        """
        if vis_id not in self.active_visualizations:
            logger.warning(f"Visualisierung {vis_id} nicht gefunden")
            return
        
        # Schließe Figur
        plt.close(self.active_visualizations[vis_id]["figure"])
        
        # Entferne Visualisierung aus aktiven Visualisierungen
        del self.active_visualizations[vis_id]
        del self.data_buffers[vis_id]
        del self.update_queues[vis_id]
        
        if vis_id in self.animation_threads:
            del self.animation_threads[vis_id]
        
        logger.info(f"Visualisierung {vis_id} geschlossen")
    
    def close_all_visualizations(self):
        """Schließt alle Visualisierungen"""
        vis_ids = list(self.active_visualizations.keys())
        for vis_id in vis_ids:
            self.close_visualization(vis_id)
        
        logger.info("Alle Visualisierungen geschlossen")
        
    def _initialize_prism_visualization(self):
        """Initialisiert erweiterte Visualisierungsfunktionen für die PRISM-Engine"""
        if not self.prism_engine:
            return
            
        try:
            # Registriere Callbacks für PRISM-Engine-Ereignisse
            if hasattr(self.prism_engine, 'register_visualization_callback'):
                self.prism_engine.register_visualization_callback(self._handle_prism_event)
                
            # Erweitere die Visualisierungskonfiguration mit PRISM-spezifischen Optionen
            prism_vis_types = {
                'PRISM_MATRIX': 'Visualisierung der PRISM-Matrix',
                'PROBABILITY_FLOW': 'Wahrscheinlichkeitsfluss-Visualisierung',
                'REALITY_MODULATION': 'Realitätsmodulations-Visualisierung',
                'TIMELINE_BRANCHING': 'Zeitlinienverzweigungs-Visualisierung'
            }
            
            # Speichere PRISM-spezifische Visualisierungstypen
            self.prism_visualization_types = prism_vis_types
            
            # Initialisiere Datenbuffer für PRISM-Visualisierungen
            for vis_type in prism_vis_types.keys():
                self.data_buffers[vis_type] = []
                self.update_queues[vis_type] = queue.Queue()
                
            logger.info(f"PRISM-Visualisierungsfunktionen initialisiert: {', '.join(prism_vis_types.keys())}")
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung der PRISM-Visualisierungsfunktionen: {e}")
            
    def _handle_prism_event(self, event_type, event_data):
        """Verarbeitet Ereignisse von der PRISM-Engine für die Visualisierung
        
        Args:
            event_type: Typ des Ereignisses
            event_data: Ereignisdaten
        """
        try:
            # Füge Ereignisdaten zur entsprechenden Warteschlange hinzu
            if event_type in self.update_queues:
                self.update_queues[event_type].put(event_data)
                
            # Aktualisiere aktive Visualisierungen, falls vorhanden
            if event_type in self.active_visualizations:
                for vis_id, vis_info in list(self.active_visualizations.items()):
                    if vis_info.get('type') == event_type:
                        self._update_visualization_data(vis_id, event_data)
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung eines PRISM-Ereignisses: {e}")
            
    def visualize_prism_matrix(self, matrix_data=None, config=None):
        """Erstellt eine Visualisierung der PRISM-Matrix
        
        Args:
            matrix_data: Daten der PRISM-Matrix, falls nicht direkt aus der PRISM-Engine geholt
            config: Konfiguration für die Visualisierung
            
        Returns:
            ID der erstellten Visualisierung
        """
        if not self.prism_engine and not matrix_data:
            logger.error("Keine PRISM-Engine verfügbar und keine Matrix-Daten angegeben")
            return None
            
        # Hole Matrix-Daten aus der PRISM-Engine, falls nicht angegeben
        if not matrix_data and hasattr(self.prism_engine, 'matrix'):
            try:
                matrix_data = self.prism_engine.matrix.get_visualization_data()
            except Exception as e:
                logger.error(f"Fehler beim Abrufen der Matrix-Daten: {e}")
                return None
                
        # Erstelle Visualisierung
        vis_config = config or VisualizationConfig(
            type=VisualizationType.MATRIX_3D,
            title="PRISM-Matrix Visualisierung",
            custom_params={
                "colormap": "viridis",
                "show_probability_values": True,
                "show_coordinates": True,
                "rotation_speed": 0.5
            }
        )
        
        return self.create_visualization(VisualizationType.MATRIX_3D, matrix_data, vis_config)
        
    def visualize_timeline_with_prism(self, timeline_id, config=None):
        """Erstellt eine Visualisierung einer Zeitlinie mit PRISM-Integration
        
        Args:
            timeline_id: ID der zu visualisierenden Zeitlinie
            config: Konfiguration für die Visualisierung
            
        Returns:
            ID der erstellten Visualisierung
        """
        if not self.prism_engine:
            logger.warning("Keine PRISM-Engine verfügbar, erstelle Standard-Zeitlinienvisualisierung")
            # Fallback auf Standard-Zeitlinienvisualisierung
            return self.create_visualization(VisualizationType.TIMELINE_2D, {"timeline_id": timeline_id}, config)
            
        # Hole Zeitliniendaten aus der PRISM-Engine
        timeline_data = None
        try:
            if hasattr(self.prism_engine, 'time_scope') and hasattr(self.prism_engine.time_scope, 'get_timeline_data'):
                timeline_data = self.prism_engine.time_scope.get_timeline_data(timeline_id)
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Zeitliniendaten: {e}")
            
        # Erstelle erweiterte Visualisierungskonfiguration
        vis_config = config or VisualizationConfig(
            type=VisualizationType.TIMELINE_3D,
            title=f"PRISM-Zeitlinie: {timeline_id}",
            custom_params={
                "show_probability_flow": True,
                "highlight_paradoxes": True,
                "show_reality_modulation": True,
                "interactive_nodes": True
            }
        )
        
        # Kombiniere Zeitliniendaten mit PRISM-Daten
        combined_data = {
            "timeline_id": timeline_id,
            "prism_data": timeline_data or {},
            "probability_metrics": self._get_probability_metrics_for_timeline(timeline_id) if self.prism_engine else {}
        }
        
        return self.create_visualization(VisualizationType.TIMELINE_3D, combined_data, vis_config)
        
    def _get_probability_metrics_for_timeline(self, timeline_id):
        """Holt Wahrscheinlichkeitsmetriken für eine Zeitlinie aus der PRISM-Engine
        
        Args:
            timeline_id: ID der Zeitlinie
            
        Returns:
            Dictionary mit Wahrscheinlichkeitsmetriken
        """
        if not self.prism_engine:
            return {}
            
        try:
            # Versuche, Wahrscheinlichkeitsmetriken aus der PRISM-Engine zu holen
            if hasattr(self.prism_engine, 'get_probability_metrics_for_timeline'):
                return self.prism_engine.get_probability_metrics_for_timeline(timeline_id)
                
            # Alternative Methode, falls die direkte Methode nicht verfügbar ist
            if hasattr(self.prism_engine, 'matrix') and hasattr(self.prism_engine.matrix, 'get_metrics_for_stream'):
                return self.prism_engine.matrix.get_metrics_for_stream(f"timeline_{timeline_id}")
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Wahrscheinlichkeitsmetriken: {e}")
            
        return {}
