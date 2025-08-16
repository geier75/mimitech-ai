"""
DEEP-STATE-MODUL – Social Media Analyse-Komponente

Diese Komponente implementiert die Social Media Analyse-Funktionalität des DEEP-STATE-MODULS,
mit Fokus auf die Erkennung von Trends, Stimmungen und Desinformationskampagnen.
"""

from typing import List, Dict, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from datetime import datetime
import random
import json
import logging

# Import aus dem Deep-State-Modul
from miso.strategic.deep_state import EscalationLevel
from miso.strategic.ztm_policy import ZTMPolicy, ztm_decorator


class SentimentLevel(Enum):
    """
    Stimmungslevel für Social Media Inhalte.
    """
    SEHR_NEGATIV = auto()
    NEGATIV = auto()
    NEUTRAL = auto()
    POSITIV = auto()
    SEHR_POSITIV = auto()


class ContentType(Enum):
    """
    Arten von Social Media Inhalten.
    """
    TEXT = auto()
    BILD = auto()
    VIDEO = auto()
    AUDIO = auto()
    LINK = auto()
    MEME = auto()


class SocialMediaContent:
    """
    Repräsentiert einen Social Media Inhalt.
    """
    def __init__(self,
                 content_id: str,
                 platform: str,
                 content_type: ContentType,
                 topic: str,
                 sentiment: SentimentLevel,
                 reach: int,
                 engagement: int,
                 timestamp: datetime,
                 hashtags: List[str] = None,
                 keywords: List[str] = None,
                 is_potential_disinformation: bool = False):
        self.content_id = content_id
        self.platform = platform
        self.content_type = content_type
        self.topic = topic
        self.sentiment = sentiment
        self.reach = reach
        self.engagement = engagement
        self.timestamp = timestamp
        self.hashtags = hashtags or []
        self.keywords = keywords or []
        self.is_potential_disinformation = is_potential_disinformation


class SocialMediaAnalyzer:
    """
    Analysiert Social Media Trends und Stimmungen.
    """
    
    def __init__(self):
        # Logger konfigurieren
        self.logger = logging.getLogger('SocialMediaAnalyzer')
        
        # ZTM-Policy initialisieren
        self.ztm_policy = ZTMPolicy("SocialMediaAnalyzer")
        
        # Plattformen
        self.platforms = [
            "twitter", "facebook", "instagram", "tiktok", "reddit", 
            "youtube", "linkedin", "telegram", "whatsapp", "discord"
        ]
        
        # Themen
        self.topics = [
            "politik", "wirtschaft", "gesundheit", "technologie", "umwelt", 
            "unterhaltung", "sport", "bildung", "sicherheit", "gesellschaft"
        ]
        
        # Aktuelle Inhalte
        self.current_contents: List[SocialMediaContent] = []
        
        # Historische Inhalte
        self.historical_contents: Dict[str, List[SocialMediaContent]] = {}
        
        # Aktuelle Trends nach Plattform
        self.current_trends: Dict[str, Dict[str, float]] = {platform: {} for platform in self.platforms}
        
        # Schwellenwerte für Desinformation
        self.disinformation_thresholds = {
            "engagement_anomaly": 2.5,  # Standardabweichungen über dem Durchschnitt
            "rapid_spread": 1000,  # Anzahl der Shares pro Stunde
            "sentiment_extremity": 0.8,  # Extrem positive oder negative Stimmung
            "coordinated_activity": 0.7  # Wahrscheinlichkeit koordinierter Aktivität
        }
    
    @ztm_decorator
    def register_content(self, platform: str, content_type: str, topic: str, 
                        sentiment_value: float, reach: int, engagement: int,
                        hashtags: List[str] = None, keywords: List[str] = None) -> SocialMediaContent:
        """
        Registriert einen neuen Social Media Inhalt.
        """
        try:
            # ZTM-Verifizierung der Eingabeparameter
            if self.ztm_policy.ztm_active:
                input_verification = {
                    "platform": platform,
                    "content_type": content_type,
                    "topic": topic,
                    "sentiment_value": sentiment_value,
                    "reach": reach,
                    "engagement": engagement,
                    "timestamp": datetime.now()
                }
                self.ztm_policy.verify_action("SOCIAL_MEDIA_CONTENT_INPUT", input_verification)
                
                # Überprüfe, ob Plattform gültig ist
                if platform not in self.platforms:
                    self.ztm_policy.handle_error(ValueError(f"Ungültige Plattform: {platform}"), "register_content")
                    raise ValueError(f"Ungültige Plattform: {platform}")
                
                # Überprüfe, ob Thema gültig ist
                if topic not in self.topics:
                    self.ztm_policy.handle_error(ValueError(f"Ungültiges Thema: {topic}"), "register_content")
                    raise ValueError(f"Ungültiges Thema: {topic}")
                
                # Überprüfe, ob Content-Typ gültig ist
                try:
                    content_type_enum = ContentType[content_type.upper()]
                except KeyError:
                    self.ztm_policy.handle_error(ValueError(f"Ungültiger Content-Typ: {content_type}"), "register_content")
                    raise ValueError(f"Ungültiger Content-Typ: {content_type}")
            else:
                # Wenn ZTM nicht aktiv ist, trotzdem Enum-Konvertierung durchführen
                try:
                    content_type_enum = ContentType[content_type.upper()]
                except KeyError:
                    raise ValueError(f"Ungültiger Content-Typ: {content_type}")
            
            # Bestimme Sentiment-Level
            sentiment_level = self._determine_sentiment_level(sentiment_value)
            
            # Überprüfe auf potenzielle Desinformation
            is_disinformation = self._check_potential_disinformation(
                platform, topic, sentiment_value, reach, engagement
            )
            
            # ZTM-Verifizierung der Desinformationserkennung
            if self.ztm_policy.ztm_active and is_disinformation:
                disinfo_verification = {
                    "platform": platform,
                    "topic": topic,
                    "sentiment_value": sentiment_value,
                    "reach": reach,
                    "engagement": engagement,
                    "is_disinformation": True,
                    "timestamp": datetime.now()
                }
                self.ztm_policy.verify_action("DISINFORMATION_DETECTED", disinfo_verification)
                self.logger.warning(f"{self.ztm_policy.status} Potenzielle Desinformation erkannt: {topic} auf {platform}")
            
            # Erstelle Content-ID
            content_id = f"{platform}_{topic}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Erstelle Content-Objekt
            content = SocialMediaContent(
                content_id=content_id,
                platform=platform,
                content_type=content_type_enum,
                topic=topic,
                sentiment=sentiment_level,
                reach=reach,
                engagement=engagement,
                timestamp=datetime.now(),
                hashtags=hashtags,
                keywords=keywords,
                is_potential_disinformation=is_disinformation
            )
            
            # Speichere Content
            self.current_contents.append(content)
            
            if platform not in self.historical_contents:
                self.historical_contents[platform] = []
            
            self.historical_contents[platform].append(content)
            
            # Aktualisiere Trends
            self._update_trends(platform, topic, sentiment_level, hashtags)
            
            # ZTM-Verifizierung des erstellten Contents
            if self.ztm_policy.ztm_active:
                content_verification = {
                    "content_id": content_id,
                    "platform": platform,
                    "topic": topic,
                    "sentiment": str(sentiment_level),
                    "is_disinformation": is_disinformation,
                    "timestamp": datetime.now()
                }
                self.ztm_policy.verify_action("SOCIAL_MEDIA_CONTENT_CREATED", content_verification)
            
            return content
            
        except Exception as e:
            if self.ztm_policy.ztm_active:
                self.ztm_policy.handle_error(e, "register_content")
            self.logger.error(f"Fehler bei der Registrierung eines Social Media Inhalts: {str(e)}")
            raise
    
    def _determine_sentiment_level(self, sentiment_value: float) -> SentimentLevel:
        """
        Bestimmt das Sentiment-Level basierend auf einem numerischen Wert (-1 bis 1).
        """
        if sentiment_value < -0.6:
            return SentimentLevel.SEHR_NEGATIV
        elif sentiment_value < -0.2:
            return SentimentLevel.NEGATIV
        elif sentiment_value <= 0.2:
            return SentimentLevel.NEUTRAL
        elif sentiment_value <= 0.6:
            return SentimentLevel.POSITIV
        else:
            return SentimentLevel.SEHR_POSITIV
    
    def _check_potential_disinformation(self, platform: str, topic: str, 
                                      sentiment_value: float, reach: int, 
                                      engagement: int) -> bool:
        """
        Überprüft, ob ein Inhalt potenzielle Desinformation sein könnte.
        """
        # Engagement-Anomalie prüfen
        engagement_rate = engagement / max(1, reach)
        avg_engagement_rate = 0.02  # Beispielwert, in der Realität aus Daten berechnen
        std_engagement_rate = 0.01  # Beispielwert, in der Realität aus Daten berechnen
        
        engagement_anomaly = (engagement_rate - avg_engagement_rate) / std_engagement_rate
        
        # Sentiment-Extremität prüfen
        sentiment_extremity = abs(sentiment_value)
        
        # Einfache Heuristik für Desinformationserkennung
        if (engagement_anomaly > self.disinformation_thresholds["engagement_anomaly"] and
            sentiment_extremity > self.disinformation_thresholds["sentiment_extremity"]):
            return True
        
        # Weitere Faktoren könnten hier berücksichtigt werden
        
        return False
    
    def _update_trends(self, platform: str, topic: str, sentiment: SentimentLevel, 
                     hashtags: List[str] = None):
        """
        Aktualisiert die aktuellen Trends basierend auf neuem Inhalt.
        """
        # Aktualisiere Topic-Trend
        if topic not in self.current_trends[platform]:
            self.current_trends[platform][topic] = 1.0
        else:
            self.current_trends[platform][topic] += 1.0
        
        # Aktualisiere Hashtag-Trends
        if hashtags:
            for hashtag in hashtags:
                if hashtag not in self.current_trends[platform]:
                    self.current_trends[platform][hashtag] = 0.5
                else:
                    self.current_trends[platform][hashtag] += 0.5
        
        # Normalisiere Trends (optional)
        total = sum(self.current_trends[platform].values())
        for key in self.current_trends[platform]:
            self.current_trends[platform][key] /= total
    
    @ztm_decorator
    def analyze_platform_trends(self, platform: str) -> Dict[str, Any]:
        """
        Analysiert Trends auf einer bestimmten Plattform.
        """
        try:
            # ZTM-Verifizierung der Eingabeparameter
            if self.ztm_policy.ztm_active:
                input_verification = {
                    "platform": platform,
                    "timestamp": datetime.now()
                }
                self.ztm_policy.verify_action("PLATFORM_TREND_ANALYSIS_INPUT", input_verification)
                
                # Überprüfe, ob Plattform gültig ist
                if platform not in self.platforms:
                    self.ztm_policy.handle_error(ValueError(f"Ungültige Plattform: {platform}"), "analyze_platform_trends")
                    raise ValueError(f"Ungültige Plattform: {platform}")
            
            # Sammle Inhalte für die Plattform
            platform_contents = [content for content in self.current_contents if content.platform == platform]
            
            # Berechne durchschnittliches Sentiment
            sentiment_values = []
            for content in platform_contents:
                if content.sentiment == SentimentLevel.SEHR_NEGATIV:
                    sentiment_values.append(-1.0)
                elif content.sentiment == SentimentLevel.NEGATIV:
                    sentiment_values.append(-0.5)
                elif content.sentiment == SentimentLevel.NEUTRAL:
                    sentiment_values.append(0.0)
                elif content.sentiment == SentimentLevel.POSITIV:
                    sentiment_values.append(0.5)
                elif content.sentiment == SentimentLevel.SEHR_POSITIV:
                    sentiment_values.append(1.0)
            
            avg_sentiment = sum(sentiment_values) / max(1, len(sentiment_values))
            
            # Zähle Themen
            topic_counts = {}
            for content in platform_contents:
                if content.topic not in topic_counts:
                    topic_counts[content.topic] = 0
                topic_counts[content.topic] += 1
            
            # Bestimme Top-Themen
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Sammle Hashtags
            all_hashtags = []
            for content in platform_contents:
                all_hashtags.extend(content.hashtags)
            
            # Zähle Hashtags
            hashtag_counts = {}
            for hashtag in all_hashtags:
                if hashtag not in hashtag_counts:
                    hashtag_counts[hashtag] = 0
                hashtag_counts[hashtag] += 1
            
            # Bestimme Top-Hashtags
            top_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Zähle potenzielle Desinformation
            disinfo_count = sum(1 for content in platform_contents if content.is_potential_disinformation)
            
            # Erstelle Analyseergebnis
            analysis_result = {
                "platform": platform,
                "content_count": len(platform_contents),
                "average_sentiment": avg_sentiment,
                "top_topics": dict(top_topics),
                "top_hashtags": dict(top_hashtags),
                "disinformation_count": disinfo_count,
                "disinformation_percentage": disinfo_count / max(1, len(platform_contents)) * 100,
                "timestamp": datetime.now()
            }
            
            # ZTM-Verifizierung des Analyseergebnisses
            if self.ztm_policy.ztm_active:
                analysis_verification = {
                    "platform": platform,
                    "content_count": len(platform_contents),
                    "average_sentiment": avg_sentiment,
                    "disinformation_count": disinfo_count,
                    "timestamp": datetime.now()
                }
                self.ztm_policy.verify_action("PLATFORM_TREND_ANALYSIS_RESULT", analysis_verification)
                
                # Bei hohem Desinformationsanteil zusätzliche Verifizierung
                disinfo_percentage = disinfo_count / max(1, len(platform_contents)) * 100
                if disinfo_percentage > 20:  # Schwellenwert: 20%
                    high_disinfo_verification = {
                        "platform": platform,
                        "disinformation_percentage": disinfo_percentage,
                        "requires_immediate_attention": True,
                        "timestamp": datetime.now()
                    }
                    self.ztm_policy.verify_action("HIGH_DISINFORMATION_LEVEL", high_disinfo_verification)
                    self.logger.warning(f"{self.ztm_policy.status} Hoher Desinformationsanteil erkannt: {platform} mit {disinfo_percentage:.1f}%")
            
            return analysis_result
            
        except Exception as e:
            if self.ztm_policy.ztm_active:
                self.ztm_policy.handle_error(e, "analyze_platform_trends")
            self.logger.error(f"Fehler bei der Analyse von Plattform-Trends: {str(e)}")
            raise
    
    @ztm_decorator
    def detect_coordinated_campaigns(self) -> List[Dict[str, Any]]:
        """
        Erkennt koordinierte Kampagnen über verschiedene Plattformen hinweg.
        """
        try:
            # ZTM-Verifizierung des Methodenaufrufs
            if self.ztm_policy.ztm_active:
                self.ztm_policy.verify_action("CAMPAIGN_DETECTION_START", {
                    "content_count": len(self.current_contents),
                    "timestamp": datetime.now()
                })
            
            # Gruppiere Inhalte nach Themen
            topic_contents = {}
            for content in self.current_contents:
                if content.topic not in topic_contents:
                    topic_contents[content.topic] = []
                topic_contents[content.topic].append(content)
            
            # Suche nach Kampagnen
            campaigns = []
            
            for topic, contents in topic_contents.items():
                # Mindestens 10 Inhalte für eine potenzielle Kampagne
                if len(contents) < 10:
                    continue
                
                # Überprüfe Plattformverteilung
                platforms = set(content.platform for content in contents)
                if len(platforms) < 3:  # Mindestens 3 Plattformen für eine Kampagne
                    continue
                
                # Überprüfe Zeitverteilung (innerhalb von 24 Stunden)
                timestamps = [content.timestamp for content in contents]
                min_time = min(timestamps)
                max_time = max(timestamps)
                time_span = (max_time - min_time).total_seconds() / 3600  # in Stunden
                
                if time_span > 24:  # Kampagne innerhalb von 24 Stunden
                    continue
                
                # Überprüfe gemeinsame Hashtags/Keywords
                all_hashtags = []
                for content in contents:
                    all_hashtags.extend(content.hashtags)
                
                hashtag_counts = {}
                for hashtag in all_hashtags:
                    if hashtag not in hashtag_counts:
                        hashtag_counts[hashtag] = 0
                    hashtag_counts[hashtag] += 1
                
                # Finde gemeinsame Hashtags (in mindestens 30% der Inhalte)
                common_hashtags = [h for h, count in hashtag_counts.items() 
                                if count >= len(contents) * 0.3]
                
                if not common_hashtags:  # Mindestens ein gemeinsamer Hashtag
                    continue
                
                # Überprüfe Sentiment-Verteilung
                sentiment_counts = {level: 0 for level in SentimentLevel}
                for content in contents:
                    sentiment_counts[content.sentiment] += 1
                
                # Prüfe, ob ein Sentiment dominiert (>70%)
                max_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])
                if max_sentiment[1] / len(contents) < 0.7:
                    continue
                
                # Überprüfe Desinformationsanteil
                disinfo_count = sum(1 for content in contents if content.is_potential_disinformation)
                disinfo_percentage = disinfo_count / len(contents) * 100
                
                # Erstelle Kampagnenobjekt
                campaign = {
                    "campaign_id": f"campaign_{topic}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "topic": topic,
                    "content_count": len(contents),
                    "platforms": list(platforms),
                    "time_span_hours": time_span,
                    "common_hashtags": common_hashtags,
                    "dominant_sentiment": str(max_sentiment[0]),
                    "disinformation_percentage": disinfo_percentage,
                    "is_concerning": disinfo_percentage > 50 or len(platforms) > 5,
                    "timestamp": datetime.now()
                }
                
                campaigns.append(campaign)
                
                # ZTM-Verifizierung der erkannten Kampagne
                if self.ztm_policy.ztm_active:
                    campaign_verification = {
                        "campaign_id": campaign["campaign_id"],
                        "topic": topic,
                        "content_count": len(contents),
                        "platforms_count": len(platforms),
                        "disinformation_percentage": disinfo_percentage,
                        "is_concerning": campaign["is_concerning"],
                        "timestamp": datetime.now()
                    }
                    self.ztm_policy.verify_action("CAMPAIGN_DETECTED", campaign_verification)
                    
                    # Bei besorgniserregenden Kampagnen zusätzliche Verifizierung
                    if campaign["is_concerning"]:
                        concerning_campaign_verification = {
                            "campaign_id": campaign["campaign_id"],
                            "topic": topic,
                            "disinformation_percentage": disinfo_percentage,
                            "platforms_count": len(platforms),
                            "requires_immediate_action": True,
                            "timestamp": datetime.now()
                        }
                        self.ztm_policy.verify_action("CONCERNING_CAMPAIGN_DETECTED", concerning_campaign_verification)
                        self.logger.warning(f"{self.ztm_policy.status} Besorgniserregende Kampagne erkannt: {topic} über {len(platforms)} Plattformen mit {disinfo_percentage:.1f}% Desinformation")
            
            # ZTM-Verifizierung des Methodenabschlusses
            if self.ztm_policy.ztm_active:
                self.ztm_policy.verify_action("CAMPAIGN_DETECTION_COMPLETE", {
                    "campaigns_detected": len(campaigns),
                    "timestamp": datetime.now()
                })
            
            return campaigns
            
        except Exception as e:
            if self.ztm_policy.ztm_active:
                self.ztm_policy.handle_error(e, "detect_coordinated_campaigns")
            self.logger.error(f"Fehler bei der Erkennung koordinierter Kampagnen: {str(e)}")
            raise
