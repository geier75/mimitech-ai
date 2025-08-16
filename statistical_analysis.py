"""
Statistical-Analysis für MISO

Implementiert statistische Analysefunktionen für MISO.
ZTM-Verifiziert: Dieses Modul unterliegt der ZTM-Policy und wird entsprechend überwacht.
"""

import logging
import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Union, Optional, Any, Dict

# Konfiguration des Loggings
logger = logging.getLogger("miso.math.statistical_analysis")

class StatisticalAnalysis:
    """
    Implementiert statistische Analysefunktionen für MISO.
    """
    
    def __init__(self, precision: int = 6):
        """
        Initialisiert die StatisticalAnalysis.
        
        Args:
            precision: Präzision für Berechnungen
        """
        self.precision = precision
        logger.info(f"[ZTM VERIFIED] StatisticalAnalysis initialisiert (Präzision: {precision})")
    
    def mean(self, data: List[float]) -> float:
        """
        Berechnet den Mittelwert einer Datenreihe.
        
        Args:
            data: Datenreihe
            
        Returns:
            float: Mittelwert
        """
        if not data:
            logger.warning("Versuch, Mittelwert einer leeren Datenreihe zu berechnen")
            return 0.0
        
        result = sum(data) / len(data)
        return round(result, self.precision)
    
    def median(self, data: List[float]) -> float:
        """
        Berechnet den Median einer Datenreihe.
        
        Args:
            data: Datenreihe
            
        Returns:
            float: Median
        """
        if not data:
            logger.warning("Versuch, Median einer leeren Datenreihe zu berechnen")
            return 0.0
        
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        if n % 2 == 0:
            # Gerade Anzahl: Durchschnitt der beiden mittleren Werte
            result = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        else:
            # Ungerade Anzahl: Mittlerer Wert
            result = sorted_data[n // 2]
        
        return round(result, self.precision)
    
    def mode(self, data: List[float]) -> List[float]:
        """
        Berechnet den Modus einer Datenreihe.
        
        Args:
            data: Datenreihe
            
        Returns:
            List[float]: Modus (kann mehrere Werte enthalten)
        """
        if not data:
            logger.warning("Versuch, Modus einer leeren Datenreihe zu berechnen")
            return []
        
        # Zähle die Häufigkeit jedes Werts
        counts = {}
        for value in data:
            rounded_value = round(value, self.precision)
            counts[rounded_value] = counts.get(rounded_value, 0) + 1
        
        # Finde den/die häufigsten Wert(e)
        max_count = max(counts.values())
        modes = [value for value, count in counts.items() if count == max_count]
        
        return modes
    
    def variance(self, data: List[float], ddof: int = 0) -> float:
        """
        Berechnet die Varianz einer Datenreihe.
        
        Args:
            data: Datenreihe
            ddof: Delta Degrees of Freedom (0 für Populationsvarianz, 1 für Stichprobenvarianz)
            
        Returns:
            float: Varianz
        """
        if not data or len(data) <= ddof:
            logger.warning(f"Versuch, Varianz einer zu kleinen Datenreihe zu berechnen (n={len(data)}, ddof={ddof})")
            return 0.0
        
        mean = self.mean(data)
        squared_diff = [(x - mean) ** 2 for x in data]
        result = sum(squared_diff) / (len(data) - ddof)
        
        return round(result, self.precision)
    
    def std_dev(self, data: List[float], ddof: int = 0) -> float:
        """
        Berechnet die Standardabweichung einer Datenreihe.
        
        Args:
            data: Datenreihe
            ddof: Delta Degrees of Freedom (0 für Populationsstandardabweichung, 1 für Stichprobenstandardabweichung)
            
        Returns:
            float: Standardabweichung
        """
        variance = self.variance(data, ddof)
        result = variance ** 0.5
        
        return round(result, self.precision)
    
    def covariance(self, data1: List[float], data2: List[float], ddof: int = 0) -> float:
        """
        Berechnet die Kovarianz zweier Datenreihen.
        
        Args:
            data1: Erste Datenreihe
            data2: Zweite Datenreihe
            ddof: Delta Degrees of Freedom
            
        Returns:
            float: Kovarianz
        """
        if len(data1) != len(data2):
            logger.error(f"Datenreihen haben unterschiedliche Längen: {len(data1)} und {len(data2)}")
            raise ValueError("Datenreihen müssen dieselbe Länge haben")
        
        if not data1 or len(data1) <= ddof:
            logger.warning(f"Versuch, Kovarianz zu kleiner Datenreihen zu berechnen (n={len(data1)}, ddof={ddof})")
            return 0.0
        
        mean1 = self.mean(data1)
        mean2 = self.mean(data2)
        
        cov_sum = sum((x - mean1) * (y - mean2) for x, y in zip(data1, data2))
        result = cov_sum / (len(data1) - ddof)
        
        return round(result, self.precision)
    
    def correlation(self, data1: List[float], data2: List[float]) -> float:
        """
        Berechnet den Korrelationskoeffizienten zweier Datenreihen.
        
        Args:
            data1: Erste Datenreihe
            data2: Zweite Datenreihe
            
        Returns:
            float: Korrelationskoeffizient
        """
        if len(data1) != len(data2):
            logger.error(f"Datenreihen haben unterschiedliche Längen: {len(data1)} und {len(data2)}")
            raise ValueError("Datenreihen müssen dieselbe Länge haben")
        
        std_dev1 = self.std_dev(data1, ddof=1)
        std_dev2 = self.std_dev(data2, ddof=1)
        
        if std_dev1 == 0 or std_dev2 == 0:
            logger.warning("Standardabweichung ist null, Korrelation nicht definiert")
            return 0.0
        
        cov = self.covariance(data1, data2, ddof=1)
        result = cov / (std_dev1 * std_dev2)
        
        return round(result, self.precision)
    
    def t_test(self, data1: List[float], data2: List[float], equal_var: bool = True) -> Tuple[float, float]:
        """
        Führt einen t-Test für zwei Datenreihen durch.
        
        Args:
            data1: Erste Datenreihe
            data2: Zweite Datenreihe
            equal_var: Ob gleiche Varianzen angenommen werden sollen
            
        Returns:
            Tuple[float, float]: t-Statistik und p-Wert
        """
        try:
            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
            return round(float(t_stat), self.precision), round(float(p_value), self.precision)
        except Exception as e:
            logger.error(f"Fehler beim t-Test: {e}")
            raise ValueError(f"Fehler beim t-Test: {e}")
    
    def anova(self, *data_groups: List[float]) -> Tuple[float, float]:
        """
        Führt eine einfaktorielle Varianzanalyse (ANOVA) durch.
        
        Args:
            *data_groups: Datengruppen
            
        Returns:
            Tuple[float, float]: F-Statistik und p-Wert
        """
        try:
            f_stat, p_value = stats.f_oneway(*data_groups)
            return round(float(f_stat), self.precision), round(float(p_value), self.precision)
        except Exception as e:
            logger.error(f"Fehler bei ANOVA: {e}")
            raise ValueError(f"Fehler bei ANOVA: {e}")
    
    def chi_square_test(self, observed: List[int], expected: List[float]) -> Tuple[float, float]:
        """
        Führt einen Chi-Quadrat-Test durch.
        
        Args:
            observed: Beobachtete Häufigkeiten
            expected: Erwartete Häufigkeiten
            
        Returns:
            Tuple[float, float]: Chi-Quadrat-Statistik und p-Wert
        """
        if len(observed) != len(expected):
            logger.error(f"Beobachtete und erwartete Häufigkeiten haben unterschiedliche Längen: {len(observed)} und {len(expected)}")
            raise ValueError("Beobachtete und erwartete Häufigkeiten müssen dieselbe Länge haben")
        
        try:
            chi2_stat, p_value = stats.chisquare(observed, expected)
            return round(float(chi2_stat), self.precision), round(float(p_value), self.precision)
        except Exception as e:
            logger.error(f"Fehler beim Chi-Quadrat-Test: {e}")
            raise ValueError(f"Fehler beim Chi-Quadrat-Test: {e}")
    
    def linear_regression(self, x: List[float], y: List[float]) -> Tuple[float, float, float, float]:
        """
        Führt eine lineare Regression durch.
        
        Args:
            x: Unabhängige Variable
            y: Abhängige Variable
            
        Returns:
            Tuple[float, float, float, float]: Steigung, y-Achsenabschnitt, R^2, p-Wert
        """
        if len(x) != len(y):
            logger.error(f"x und y haben unterschiedliche Längen: {len(x)} und {len(y)}")
            raise ValueError("x und y müssen dieselbe Länge haben")
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2
            
            return (
                round(float(slope), self.precision),
                round(float(intercept), self.precision),
                round(float(r_squared), self.precision),
                round(float(p_value), self.precision)
            )
        except Exception as e:
            logger.error(f"Fehler bei linearer Regression: {e}")
            raise ValueError(f"Fehler bei linearer Regression: {e}")
    
    def z_score(self, data: List[float]) -> List[float]:
        """
        Berechnet die Z-Scores einer Datenreihe.
        
        Args:
            data: Datenreihe
            
        Returns:
            List[float]: Z-Scores
        """
        if not data:
            logger.warning("Versuch, Z-Scores einer leeren Datenreihe zu berechnen")
            return []
        
        mean = self.mean(data)
        std_dev = self.std_dev(data, ddof=0)
        
        if std_dev == 0:
            logger.warning("Standardabweichung ist null, Z-Scores nicht definiert")
            return [0.0] * len(data)
        
        z_scores = [(x - mean) / std_dev for x in data]
        return [round(z, self.precision) for z in z_scores]
    
    def percentile(self, data: List[float], p: float) -> float:
        """
        Berechnet das p-te Perzentil einer Datenreihe.
        
        Args:
            data: Datenreihe
            p: Perzentil (0-100)
            
        Returns:
            float: Perzentil
        """
        if not data:
            logger.warning("Versuch, Perzentil einer leeren Datenreihe zu berechnen")
            return 0.0
        
        if p < 0 or p > 100:
            logger.error(f"Perzentil muss zwischen 0 und 100 liegen, erhalten: {p}")
            raise ValueError("Perzentil muss zwischen 0 und 100 liegen")
        
        result = np.percentile(data, p)
        return round(float(result), self.precision)
    
    def iqr(self, data: List[float]) -> float:
        """
        Berechnet den Interquartilsabstand einer Datenreihe.
        
        Args:
            data: Datenreihe
            
        Returns:
            float: Interquartilsabstand
        """
        q1 = self.percentile(data, 25)
        q3 = self.percentile(data, 75)
        result = q3 - q1
        
        return round(result, self.precision)
    
    def outliers(self, data: List[float], method: str = "iqr") -> List[float]:
        """
        Identifiziert Ausreißer in einer Datenreihe.
        
        Args:
            data: Datenreihe
            method: Methode zur Identifikation von Ausreißern ("iqr" oder "z_score")
            
        Returns:
            List[float]: Ausreißer
        """
        if not data:
            logger.warning("Versuch, Ausreißer einer leeren Datenreihe zu identifizieren")
            return []
        
        if method == "iqr":
            q1 = self.percentile(data, 25)
            q3 = self.percentile(data, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = [x for x in data if x < lower_bound or x > upper_bound]
            
        elif method == "z_score":
            z_scores = self.z_score(data)
            outliers = [data[i] for i, z in enumerate(z_scores) if abs(z) > 3]
            
        else:
            logger.error(f"Unbekannte Methode zur Identifikation von Ausreißern: {method}")
            raise ValueError("Methode muss 'iqr' oder 'z_score' sein")
        
        return outliers
