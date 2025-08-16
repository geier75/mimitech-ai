#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LogFilter-Komponente für VXOR AI Blackbox.

Implementiert Filter zum Schutz sensibler Daten in Log-Einträgen, um
zu verhindern, dass vertrauliche Informationen wie Passwörter, API-Schlüssel
oder persönliche Daten in den Logs erscheinen.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Set, Pattern

# Globale Liste der sensitiven Datenmuster
_sensitive_patterns = {
    # Passwörter und Zugangsdaten
    "password": re.compile(r'(?i)(password|passwd|pwd)[\s]*[=:][^,}\]]*', re.IGNORECASE),
    "api_key": re.compile(r'(?i)(api[_-]?key|access[_-]?key|secret[_-]?key)[\s]*[=:][^,}\]]*', re.IGNORECASE),
    "token": re.compile(r'(?i)(token|auth[_-]?token|bearer)[\s]*[=:][^,}\]]*', re.IGNORECASE),
    
    # Persönliche Daten
    "email": re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'),
    "credit_card": re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
    
    # Kryptographische Daten
    "private_key": re.compile(r'(?i)(private[_-]?key|signing[_-]?key)[\s]*[=:][^,}\]]*', re.IGNORECASE),
    "seed_phrase": re.compile(r'(?i)(seed[_-]?phrase|mnemonic|recovery[_-]?phrase)[\s]*[=:][^,}\]]*', re.IGNORECASE),
    
    # Spezifische VXOR AI Patterns für T-Mathematics Engine und M-LINGUA
    "mlx_tensor_key": re.compile(r'(?i)(mlx[_-]?key|neural[_-]?engine[_-]?key)[\s]*[=:][^,}\]]*', re.IGNORECASE),
    "torch_tensor_key": re.compile(r'(?i)(torch[_-]?key|mps[_-]?key)[\s]*[=:][^,}\]]*', re.IGNORECASE),
    "m_lingua_key": re.compile(r'(?i)(m[_-]?lingua[_-]?key|lingua[_-]?api[_-]?key)[\s]*[=:][^,}\]]*', re.IGNORECASE),
    
    # Netzwerkinformationen
    "internal_ip": re.compile(r'\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b'),
}

# Ersetzungstext für maskierte Daten
MASK_REPLACEMENT = "[REDACTED]"


class SensitiveDataFilter(logging.Filter):
    """
    Ein Log-Filter, der sensitive Daten in Log-Einträgen maskiert.
    
    Unterstützt:
    - Standardmuster für Passwörter, API-Schlüssel, etc.
    - Benutzerdefinierte Muster für spezifische sensitive Daten
    - Kontextbewusste Filterung
    - Spezielle Unterstützung für VXOR AI-spezifische Daten (MLXTensor, TorchTensor, M-LINGUA)
    """
    
    def __init__(self, name: str = "", patterns: Optional[Dict[str, Pattern]] = None,
                enable_all: bool = True, disable_patterns: Optional[List[str]] = None):
        """
        Initialisiert den SensitiveDataFilter.
        
        Args:
            name: Name des Filters
            patterns: Optionale zusätzliche Muster für sensitive Daten
            enable_all: Wenn True, werden alle Standardmuster aktiviert
            disable_patterns: Liste von Musternamen, die deaktiviert werden sollen
        """
        super().__init__(name)
        self.patterns = dict(_sensitive_patterns)
        
        # Füge benutzerdefinierte Muster hinzu
        if patterns:
            self.patterns.update(patterns)
        
        # Deaktiviere bestimmte Muster, falls angegeben
        self.active_patterns = set(self.patterns.keys())
        if not enable_all:
            self.active_patterns.clear()
        
        if disable_patterns:
            for pattern_name in disable_patterns:
                if pattern_name in self.active_patterns:
                    self.active_patterns.remove(pattern_name)
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filtert sensitive Daten aus einem Log-Eintrag.
        
        Args:
            record: Der zu filternde Log-Eintrag
            
        Returns:
            True (der Eintrag wird nach der Filterung immer protokolliert)
        """
        if isinstance(record.msg, str):
            # Wende alle aktiven Muster auf die Nachricht an
            for pattern_name in self.active_patterns:
                if pattern_name in self.patterns:
                    record.msg = self.patterns[pattern_name].sub(
                        f"\\1={MASK_REPLACEMENT}" if "=" in record.msg else MASK_REPLACEMENT,
                        record.msg
                    )
        
        # Filter auch die args, falls vorhanden
        if record.args:
            record.args = self._filter_args(record.args)
        
        return True
    
    def _filter_args(self, args: Any) -> Any:
        """
        Filtert sensitive Daten aus den args eines Log-Eintrags.
        
        Args:
            args: Die zu filternden args
            
        Returns:
            Die gefilterten args
        """
        if isinstance(args, dict):
            # Filtere Dictionaries
            return {k: self._filter_args(v) for k, v in args.items()}
        
        elif isinstance(args, (list, tuple)):
            # Filtere Listen und Tupel
            return type(args)(self._filter_args(arg) for arg in args)
        
        elif isinstance(args, str):
            # Filtere Strings
            filtered = args
            for pattern_name in self.active_patterns:
                if pattern_name in self.patterns:
                    filtered = self.patterns[pattern_name].sub(
                        f"\\1={MASK_REPLACEMENT}" if "=" in filtered else MASK_REPLACEMENT,
                        filtered
                    )
            return filtered
        
        # Andere Typen unverändert zurückgeben
        return args


def add_sensitive_pattern(pattern_name: str, pattern: str, flags: int = re.IGNORECASE) -> bool:
    """
    Fügt ein neues Muster für sensitive Daten hinzu.
    
    Args:
        pattern_name: Name des Musters
        pattern: Regulärer Ausdruck für das Muster
        flags: Flags für den regulären Ausdruck
        
    Returns:
        True, wenn das Muster erfolgreich hinzugefügt wurde
    """
    if pattern_name in _sensitive_patterns:
        return False
    
    try:
        compiled_pattern = re.compile(pattern, flags)
        _sensitive_patterns[pattern_name] = compiled_pattern
        return True
    except re.error:
        return False


def remove_sensitive_pattern(pattern_name: str) -> bool:
    """
    Entfernt ein Muster für sensitive Daten.
    
    Args:
        pattern_name: Name des zu entfernenden Musters
        
    Returns:
        True, wenn das Muster erfolgreich entfernt wurde
    """
    if pattern_name in _sensitive_patterns:
        del _sensitive_patterns[pattern_name]
        return True
    return False


def get_all_patterns() -> Dict[str, str]:
    """
    Gibt alle verfügbaren Muster für sensitive Daten zurück.
    
    Returns:
        Dictionary mit Musternamen und Mustern als Strings
    """
    return {name: pattern.pattern for name, pattern in _sensitive_patterns.items()}


def apply_sensitive_data_filter(logger: logging.Logger, 
                              enable_all: bool = True, 
                              disable_patterns: Optional[List[str]] = None) -> logging.Logger:
    """
    Wendet einen SensitiveDataFilter auf einen Logger an.
    
    Args:
        logger: Der Logger, auf den der Filter angewendet werden soll
        enable_all: Wenn True, werden alle Standardmuster aktiviert
        disable_patterns: Liste von Musternamen, die deaktiviert werden sollen
        
    Returns:
        Der Logger mit angewendetem Filter
    """
    filter = SensitiveDataFilter(enable_all=enable_all, disable_patterns=disable_patterns)
    logger.addFilter(filter)
    return logger
