#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Predictive Stream Analyzer

Kontinuierliche Analyse von Trends, Mustern und Bewegungen für die PRISM-Engine.
Nutzt LSTM-Netze und Reinforcement Patterns zur Trenddetektion.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import json
import time
from collections import deque

# Konfiguriere Logging
logger = logging.getLogger("MISO.simulation.predictive_stream")

# Prüfen, ob Apple Silicon verfügbar ist und entsprechende Optimierungen aktivieren
is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine
if is_apple_silicon:
    # Apple Neural Engine Optimierungen
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import von internen Modulen
try:
    from miso.math.tensor_ops import MISOTensor, MLXTensor, TorchTensor
    HAS_TENSOR_OPS = True
except ImportError:
    logger.warning("Tensor-Operationen konnten nicht importiert werden. Verwende Standard-Implementierung.")
    HAS_TENSOR_OPS = False


class LSTMPredictor(nn.Module):
    """LSTM-Netzwerk für Sequenzvorhersage"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        """
        Initialisiert das LSTM-Netzwerk
        
        Args:
            input_size: Größe des Eingabevektors
            hidden_size: Größe des versteckten Zustands
            num_layers: Anzahl der LSTM-Schichten
            output_size: Größe des Ausgabevektors
        """
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Vorwärtsdurchlauf durch das Netzwerk
        
        Args:
            x: Eingabetensor der Form (batch_size, sequence_length, input_size)
            
        Returns:
            Ausgabetensor der Form (batch_size, output_size)
        """
        # Initialisiere versteckten Zustand
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Vorwärtsdurchlauf durch LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Verwende nur den letzten Zeitschritt
        out = self.fc(out[:, -1, :])
        return out


class PatternDetector:
    """Erkennt Muster in Datenströmen"""
    
    def __init__(self, pattern_length: int = 5, similarity_threshold: float = 0.8):
        """
        Initialisiert den PatternDetector
        
        Args:
            pattern_length: Länge der zu erkennenden Muster
            similarity_threshold: Schwellenwert für die Ähnlichkeit
        """
        self.pattern_length = pattern_length
        self.similarity_threshold = similarity_threshold
        self.known_patterns = []
    
    def detect_patterns(self, data_stream: List[float]) -> List[Dict[str, Any]]:
        """
        Erkennt Muster in einem Datenstrom
        
        Args:
            data_stream: Liste von Datenpunkten
            
        Returns:
            Liste von erkannten Mustern
        """
        if len(data_stream) < self.pattern_length:
            return []
        
        detected_patterns = []
        
        # Extrahiere alle möglichen Muster der angegebenen Länge
        for i in range(len(data_stream) - self.pattern_length + 1):
            pattern = data_stream[i:i+self.pattern_length]
            
            # Prüfe, ob das Muster bereits bekannt ist
            is_known = False
            for known_pattern in self.known_patterns:
                similarity = self._calculate_similarity(pattern, known_pattern["pattern"])
                if similarity >= self.similarity_threshold:
                    is_known = True
                    known_pattern["occurrences"] += 1
                    known_pattern["last_seen"] = i
                    break
            
            # Wenn das Muster neu ist, füge es hinzu
            if not is_known:
                self.known_patterns.append({
                    "pattern": pattern,
                    "occurrences": 1,
                    "first_seen": i,
                    "last_seen": i
                })
        
        # Filtere Muster mit mehr als einer Vorkommen
        for pattern in self.known_patterns:
            if pattern["occurrences"] > 1:
                detected_patterns.append({
                    "pattern": pattern["pattern"],
                    "occurrences": pattern["occurrences"],
                    "first_seen": pattern["first_seen"],
                    "last_seen": pattern["last_seen"],
                    "length": self.pattern_length
                })
        
        return detected_patterns
    
    def _calculate_similarity(self, pattern1: List[float], pattern2: List[float]) -> float:
        """
        Berechnet die Ähnlichkeit zwischen zwei Mustern
        
        Args:
            pattern1: Erstes Muster
            pattern2: Zweites Muster
            
        Returns:
            Ähnlichkeitswert zwischen 0 und 1
        """
        if len(pattern1) != len(pattern2):
            return 0.0
        
        # Berechne die euklidische Distanz
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(pattern1, pattern2)))
        
        # Normalisiere die Distanz auf einen Ähnlichkeitswert zwischen 0 und 1
        max_distance = np.sqrt(len(pattern1) * 4)  # Annahme: Werte liegen im Bereich [-1, 1]
        similarity = 1.0 - (distance / max_distance)
        
        return max(0.0, min(1.0, similarity))


class PredictiveStreamAnalyzer:
    """
    Kontinuierliche Analyse von Trends, Mustern, Bewegungen
    Nutzt LSTM-Netze + Reinforcement Patterns zur Trenddetektion
    """
    
    def __init__(self, sequence_length: int = 10, prediction_horizon: int = 5):
        """
        Initialisiert den PredictiveStreamAnalyzer
        
        Args:
            sequence_length: Länge der Sequenz für die Vorhersage
            prediction_horizon: Anzahl der vorherzusagenden Zeitschritte
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.lstm_model = None
        self.pattern_detector = PatternDetector()
        self.data_buffer = {}  # Puffer für verschiedene Datenströme
        self.models = {}  # LSTM-Modelle für verschiedene Datenströme
        
        logger.info(f"PredictiveStreamAnalyzer initialisiert mit Sequenzlänge {sequence_length} "
                   f"und Vorhersagehorizont {prediction_horizon}")
    
    def register_data_stream(self, stream_id: str, initial_data: List[float] = None):
        """
        Registriert einen neuen Datenstrom
        
        Args:
            stream_id: ID des Datenstroms
            initial_data: Initiale Daten für den Strom
        """
        if stream_id in self.data_buffer:
            logger.warning(f"Datenstrom {stream_id} ist bereits registriert")
            return
        
        self.data_buffer[stream_id] = deque(maxlen=1000)  # Begrenze die Größe des Puffers
        
        if initial_data:
            self.data_buffer[stream_id].extend(initial_data)
        
        logger.info(f"Datenstrom {stream_id} registriert")
    
    def add_data_point(self, stream_id: str, value: float):
        """
        Fügt einen Datenpunkt zu einem Datenstrom hinzu
        
        Args:
            stream_id: ID des Datenstroms
            value: Datenpunkt
        """
        if stream_id not in self.data_buffer:
            self.register_data_stream(stream_id)
        
        self.data_buffer[stream_id].append(value)
    
    def train_model(self, stream_id: str, epochs: int = 100, learning_rate: float = 0.001):
        """
        Trainiert ein LSTM-Modell für einen Datenstrom
        
        Args:
            stream_id: ID des Datenstroms
            epochs: Anzahl der Trainingsepochen
            learning_rate: Lernrate
            
        Returns:
            Trainingsergebnis
        """
        if stream_id not in self.data_buffer:
            logger.error(f"Datenstrom {stream_id} ist nicht registriert")
            return {"status": "error", "message": "Datenstrom nicht registriert"}
        
        if len(self.data_buffer[stream_id]) < self.sequence_length + self.prediction_horizon:
            logger.error(f"Nicht genügend Daten im Strom {stream_id} für das Training")
            return {"status": "error", "message": "Nicht genügend Daten"}
        
        # Konvertiere Daten in Trainingsformat
        data = list(self.data_buffer[stream_id])
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            X.append(data[i:i+self.sequence_length])
            y.append(data[i+self.sequence_length:i+self.sequence_length+self.prediction_horizon])
        
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(2).to(device)  # (batch_size, sequence_length, 1)
        y = torch.tensor(y, dtype=torch.float32).to(device)  # (batch_size, prediction_horizon)
        
        # Erstelle und trainiere das Modell
        model = LSTMPredictor(input_size=1, hidden_size=50, num_layers=2, output_size=self.prediction_horizon)
        model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoche {epoch+1}/{epochs}, Verlust: {loss.item():.4f}")
        
        # Speichere das trainierte Modell
        self.models[stream_id] = model
        
        return {
            "status": "success",
            "stream_id": stream_id,
            "epochs": epochs,
            "final_loss": loss.item()
        }
    
    def predict_next_values(self, stream_id: str, steps: int = None) -> List[float]:
        """
        Sagt die nächsten Werte in einem Datenstrom voraus
        
        Args:
            stream_id: ID des Datenstroms
            steps: Anzahl der vorherzusagenden Schritte (Standard: prediction_horizon)
            
        Returns:
            Liste von vorhergesagten Werten
        """
        if stream_id not in self.data_buffer:
            logger.error(f"Datenstrom {stream_id} ist nicht registriert")
            return []
        
        if stream_id not in self.models:
            logger.error(f"Kein trainiertes Modell für Datenstrom {stream_id}")
            return []
        
        steps = steps or self.prediction_horizon
        
        # Hole die letzten sequence_length Datenpunkte
        data = list(self.data_buffer[stream_id])[-self.sequence_length:]
        
        if len(data) < self.sequence_length:
            logger.error(f"Nicht genügend Daten im Strom {stream_id} für die Vorhersage")
            return []
        
        # Konvertiere Daten in Vorhersageformat
        X = torch.tensor([data], dtype=torch.float32).unsqueeze(2).to(device)  # (1, sequence_length, 1)
        
        # Vorhersage
        model = self.models[stream_id]
        model.eval()
        with torch.no_grad():
            predictions = model(X).cpu().numpy()[0]
        
        return predictions.tolist()
    
    def analyze_stream(self, data_stream: List[Any]) -> Dict[str, Any]:
        """
        Analysiert einen Datenstrom und erkennt Trends und Muster
        
        Args:
            data_stream: Liste von Datenpunkten
            
        Returns:
            Analyseergebnis
        """
        if not data_stream:
            return {"status": "error", "message": "Leerer Datenstrom"}
        
        # Konvertiere Datenstrom in numerisches Format, falls nötig
        numeric_stream = []
        for item in data_stream:
            if isinstance(item, (int, float)):
                numeric_stream.append(float(item))
            elif isinstance(item, dict) and "value" in item and isinstance(item["value"], (int, float)):
                numeric_stream.append(float(item["value"]))
            else:
                logger.warning(f"Nicht-numerischer Wert im Datenstrom: {item}")
        
        if not numeric_stream:
            return {"status": "error", "message": "Keine numerischen Daten im Strom"}
        
        # Registriere temporären Datenstrom
        temp_stream_id = f"temp_{time.time()}"
        self.register_data_stream(temp_stream_id, numeric_stream)
        
        # Erkenne Trends
        trend_result = self.detect_trend(numeric_stream)
        
        # Erkenne Muster
        patterns = self.pattern_detector.detect_patterns(numeric_stream)
        
        # Trainiere Modell und mache Vorhersage, wenn genügend Daten vorhanden sind
        prediction_result = {"status": "not_available"}
        if len(numeric_stream) >= self.sequence_length + self.prediction_horizon:
            training_result = self.train_model(temp_stream_id, epochs=50)
            
            if training_result["status"] == "success":
                predictions = self.predict_next_values(temp_stream_id)
                prediction_result = {
                    "status": "success",
                    "predictions": predictions,
                    "horizon": self.prediction_horizon
                }
        
        # Lösche temporären Datenstrom
        del self.data_buffer[temp_stream_id]
        if temp_stream_id in self.models:
            del self.models[temp_stream_id]
        
        return {
            "status": "success",
            "data_length": len(numeric_stream),
            "trend": trend_result,
            "patterns": patterns,
            "prediction": prediction_result,
            "statistics": {
                "mean": np.mean(numeric_stream),
                "std": np.std(numeric_stream),
                "min": min(numeric_stream),
                "max": max(numeric_stream)
            }
        }
    
    def detect_trend(self, data_stream: List[Any]) -> Dict[str, Any]:
        """
        Erkennt Trends in einem Datenstrom
        
        Args:
            data_stream: Liste von Datenpunkten
            
        Returns:
            Trendinformationen
        """
        if not data_stream or len(data_stream) < 3:
            return {"status": "error", "message": "Zu wenig Daten für Trenderkennung"}
        
        # Konvertiere in numerisches Format, falls nötig
        numeric_stream = []
        for item in data_stream:
            if isinstance(item, (int, float)):
                numeric_stream.append(float(item))
            elif isinstance(item, dict) and "value" in item and isinstance(item["value"], (int, float)):
                numeric_stream.append(float(item["value"]))
            else:
                logger.warning(f"Nicht-numerischer Wert im Datenstrom: {item}")
        
        if not numeric_stream:
            return {"status": "error", "message": "Keine numerischen Daten im Strom"}
        
        # Berechne lineare Regression
        x = np.arange(len(numeric_stream))
        y = np.array(numeric_stream)
        
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Berechne R²
        y_pred = m * x + c
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # Bestimme Trendrichtung und -stärke
        trend_direction = "steigend" if m > 0 else "fallend" if m < 0 else "stabil"
        trend_strength = abs(m) * len(numeric_stream) / (max(numeric_stream) - min(numeric_stream)) if max(numeric_stream) != min(numeric_stream) else 0
        
        # Klassifiziere Trendstärke
        if trend_strength < 0.1:
            trend_strength_category = "sehr schwach"
        elif trend_strength < 0.3:
            trend_strength_category = "schwach"
        elif trend_strength < 0.5:
            trend_strength_category = "moderat"
        elif trend_strength < 0.7:
            trend_strength_category = "stark"
        else:
            trend_strength_category = "sehr stark"
        
        return {
            "status": "success",
            "direction": trend_direction,
            "slope": m,
            "intercept": c,
            "r_squared": r_squared,
            "strength": trend_strength,
            "strength_category": trend_strength_category,
            "confidence": r_squared  # Verwende R² als Konfidenzmaß
        }


# Beispiel für die Verwendung des PredictiveStreamAnalyzers
if __name__ == "__main__":
    # Konfiguriere Logging für Standalone-Ausführung
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Erstelle einen PredictiveStreamAnalyzer
    analyzer = PredictiveStreamAnalyzer(sequence_length=10, prediction_horizon=5)
    
    # Generiere einen Beispiel-Datenstrom (Sinus-Welle mit Rauschen)
    x = np.linspace(0, 4 * np.pi, 100)
    data_stream = np.sin(x) + np.random.normal(0, 0.1, 100)
    
    # Analysiere den Datenstrom
    result = analyzer.analyze_stream(data_stream)
    
    # Zeige Ergebnis
    print(f"Analyseergebnis: {json.dumps(result, indent=2)}")
