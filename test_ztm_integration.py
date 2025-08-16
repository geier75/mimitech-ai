"""
MISO - ZTM-Integrationstests

Diese Testklasse überprüft die korrekte Integration der Zero-Tolerance Mode (ZTM)
in allen MISO-Komponenten.
"""

import unittest
import logging
from datetime import datetime

# Importiere alle Komponenten mit ZTM-Integration
from miso.strategic.ztm_policy import ZTMPolicy
from miso.strategic.market_observer import MarketObserver
from miso.strategic.threat_analyzer import ThreatAnalyzer
from miso.strategic.geopolitical_analyzer import GeopoliticalAnalyzer
from miso.strategic.social_media_analyzer import SocialMediaAnalyzer, ContentType, SentimentLevel
from miso.strategic.economic_analyzer import EconomicAnalyzer
from miso.strategic.paradox_resolver import ParadoxResolver, ParadoxType
from miso.strategic.deep_state import DeepStateModul


class TestZTMIntegration(unittest.TestCase):
    """
    Testet die ZTM-Integration in allen MISO-Komponenten.
    """
    
    def setUp(self):
        """
        Testumgebung einrichten.
        """
        # Konfiguriere Logging für Tests
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('TestZTMIntegration')
        
        # Initialisiere alle Komponenten
        self.market_observer = MarketObserver()
        self.threat_analyzer = ThreatAnalyzer()
        self.geopolitical_analyzer = GeopoliticalAnalyzer()
        self.social_media_analyzer = SocialMediaAnalyzer()
        self.economic_analyzer = EconomicAnalyzer()
        self.paradox_resolver = ParadoxResolver()
        self.deep_state = DeepStateModul()
        
        # Aktiviere ZTM in allen Komponenten
        self._activate_ztm_in_all_components()
    
    def _activate_ztm_in_all_components(self):
        """
        Aktiviert den ZTM in allen Komponenten.
        """
        # Aktiviere ZTM in jeder Komponente
        self.market_observer.ztm_policy.activate_ztm()
        self.threat_analyzer.ztm_policy.activate_ztm()
        self.geopolitical_analyzer.ztm_policy.activate_ztm()
        self.social_media_analyzer.ztm_policy.activate_ztm()
        self.economic_analyzer.ztm_policy.activate_ztm()
        self.paradox_resolver.ztm_policy.activate_ztm()
        self.deep_state.activate_ztm()
        
        # Überprüfe, ob ZTM in allen Komponenten aktiv ist
        self.assertTrue(self.market_observer.ztm_policy.ztm_active)
        self.assertTrue(self.threat_analyzer.ztm_policy.ztm_active)
        self.assertTrue(self.geopolitical_analyzer.ztm_policy.ztm_active)
        self.assertTrue(self.social_media_analyzer.ztm_policy.ztm_active)
        self.assertTrue(self.economic_analyzer.ztm_policy.ztm_active)
        self.assertTrue(self.paradox_resolver.ztm_policy.ztm_active)
        self.assertTrue(self.deep_state.ztm_mode)
    
    def test_market_observer_ztm(self):
        """
        Testet die ZTM-Integration im MarketObserver.
        """
        self.logger.info("Teste ZTM-Integration im MarketObserver...")
        
        # Überprüfe ZTM-Aktivierung
        self.assertTrue(self.market_observer.ztm_policy.ztm_active)
        
        # Teste Methode mit ZTM-Decorator
        try:
            self.market_observer.update_market_data()
            self.logger.info("MarketObserver ZTM-Test erfolgreich")
        except Exception as e:
            self.fail(f"MarketObserver ZTM-Test fehlgeschlagen: {str(e)}")
    
    def test_threat_analyzer_ztm(self):
        """
        Testet die ZTM-Integration im ThreatAnalyzer.
        """
        self.logger.info("Teste ZTM-Integration im ThreatAnalyzer...")
        
        # Überprüfe ZTM-Aktivierung
        self.assertTrue(self.threat_analyzer.ztm_policy.ztm_active)
        
        # Teste Methode mit ZTM-Decorator
        try:
            threat = self.threat_analyzer.detect_threat(
                threat_type="cyber",
                source="unknown_actor",
                target="financial_system",
                indicators=["unusual_traffic", "encryption_patterns"]
            )
            self.assertIsNotNone(threat)
            self.logger.info("ThreatAnalyzer ZTM-Test erfolgreich")
        except Exception as e:
            self.fail(f"ThreatAnalyzer ZTM-Test fehlgeschlagen: {str(e)}")
    
    def test_geopolitical_analyzer_ztm(self):
        """
        Testet die ZTM-Integration im GeopoliticalAnalyzer.
        """
        self.logger.info("Teste ZTM-Integration im GeopoliticalAnalyzer...")
        
        # Überprüfe ZTM-Aktivierung
        self.assertTrue(self.geopolitical_analyzer.ztm_policy.ztm_active)
        
        # Teste Methode mit ZTM-Decorator
        try:
            event = self.geopolitical_analyzer.register_event(
                event_type="diplomatic_tension",
                region="europe",
                countries=["germany", "france"],
                intensity=0.7,
                description="Diplomatische Spannungen zwischen Deutschland und Frankreich",
                tags=["diplomacy", "europe", "tension"]
            )
            self.assertIsNotNone(event)
            
            # Teste weitere Methode mit ZTM-Decorator
            analysis = self.geopolitical_analyzer.analyze_regional_trends("europe")
            self.assertIsNotNone(analysis)
            
            self.logger.info("GeopoliticalAnalyzer ZTM-Test erfolgreich")
        except Exception as e:
            self.fail(f"GeopoliticalAnalyzer ZTM-Test fehlgeschlagen: {str(e)}")
    
    def test_social_media_analyzer_ztm(self):
        """
        Testet die ZTM-Integration im SocialMediaAnalyzer.
        """
        self.logger.info("Teste ZTM-Integration im SocialMediaAnalyzer...")
        
        # Überprüfe ZTM-Aktivierung
        self.assertTrue(self.social_media_analyzer.ztm_policy.ztm_active)
        
        # Teste Methode mit ZTM-Decorator
        try:
            content = self.social_media_analyzer.register_content(
                platform="twitter",
                content_type="TEXT",
                topic="politik",
                sentiment_value=0.3,
                reach=5000,
                engagement=500,
                hashtags=["election", "politics"],
                keywords=["vote", "democracy"]
            )
            self.assertIsNotNone(content)
            
            # Teste weitere Methode mit ZTM-Decorator
            analysis = self.social_media_analyzer.analyze_platform_trends("twitter")
            self.assertIsNotNone(analysis)
            
            self.logger.info("SocialMediaAnalyzer ZTM-Test erfolgreich")
        except Exception as e:
            self.fail(f"SocialMediaAnalyzer ZTM-Test fehlgeschlagen: {str(e)}")
    
    def test_economic_analyzer_ztm(self):
        """
        Testet die ZTM-Integration im EconomicAnalyzer.
        """
        self.logger.info("Teste ZTM-Integration im EconomicAnalyzer...")
        
        # Überprüfe ZTM-Aktivierung
        self.assertTrue(self.economic_analyzer.ztm_policy.ztm_active)
        
        # Teste Methode mit ZTM-Decorator
        try:
            event = self.economic_analyzer.register_event(
                event_type="market_crash",
                region="north_america",
                sector="finance",
                impact=0.8,
                description="Börsencrash an der Wall Street"
            )
            self.assertIsNotNone(event)
            
            # Teste weitere Methode mit ZTM-Decorator
            analysis = self.economic_analyzer.analyze_sector("finance")
            self.assertIsNotNone(analysis)
            
            self.logger.info("EconomicAnalyzer ZTM-Test erfolgreich")
        except Exception as e:
            self.fail(f"EconomicAnalyzer ZTM-Test fehlgeschlagen: {str(e)}")
    
    def test_paradox_resolver_ztm(self):
        """
        Testet die ZTM-Integration im ParadoxResolver.
        """
        self.logger.info("Teste ZTM-Integration im ParadoxResolver...")
        
        # Überprüfe ZTM-Aktivierung
        self.assertTrue(self.paradox_resolver.ztm_policy.ztm_active)
        
        # Teste Methode mit ZTM-Decorator
        try:
            paradox = self.paradox_resolver.detect_paradox(
                description="Temporale Paradoxie in der Zeitlinie",
                paradox_type="TEMPORAL",
                complexity=0.7,
                stability=0.4
            )
            self.assertIsNotNone(paradox)
            
            # Teste weitere Methode mit ZTM-Decorator
            resolution_result = self.paradox_resolver.resolve_paradox(paradox.paradox_id)
            self.assertIsNotNone(resolution_result)
            
            # Teste Paradoxvorhersage
            predictions = self.paradox_resolver.predict_paradoxes()
            
            self.logger.info("ParadoxResolver ZTM-Test erfolgreich")
        except Exception as e:
            self.fail(f"ParadoxResolver ZTM-Test fehlgeschlagen: {str(e)}")
    
    def test_deep_state_ztm(self):
        """
        Testet die ZTM-Integration im DeepStateModul.
        """
        self.logger.info("Teste ZTM-Integration im DeepStateModul...")
        
        # Überprüfe ZTM-Aktivierung
        self.assertTrue(self.deep_state.ztm_mode)
        
        # Teste ZTM-Methoden
        try:
            # Deaktiviere ZTM
            self.deep_state.deactivate_ztm()
            self.assertFalse(self.deep_state.ztm_mode)
            
            # Aktiviere ZTM wieder
            self.deep_state.activate_ztm()
            self.assertTrue(self.deep_state.ztm_mode)
            
            self.logger.info("DeepStateModul ZTM-Test erfolgreich")
        except Exception as e:
            self.fail(f"DeepStateModul ZTM-Test fehlgeschlagen: {str(e)}")
    
    def test_error_handling_with_ztm(self):
        """
        Testet die Fehlerbehandlung mit aktiviertem ZTM.
        """
        self.logger.info("Teste Fehlerbehandlung mit aktiviertem ZTM...")
        
        # Teste Fehlerbehandlung im ParadoxResolver
        try:
            # Versuche, ein nicht existierendes Paradox aufzulösen
            with self.assertRaises(ValueError):
                self.paradox_resolver.resolve_paradox("nicht_existierende_id")
            
            self.logger.info("ZTM-Fehlerbehandlungstest erfolgreich")
        except Exception as e:
            self.fail(f"ZTM-Fehlerbehandlungstest fehlgeschlagen: {str(e)}")
    
    def test_ztm_verification_actions(self):
        """
        Testet die Verifizierungsaktionen des ZTM.
        """
        self.logger.info("Teste ZTM-Verifizierungsaktionen...")
        
        # Erstelle eine separate ZTMPolicy für Tests
        test_policy = ZTMPolicy("TestComponent")
        test_policy.activate_ztm()
        
        # Teste Verifizierungsaktionen
        try:
            # Verifiziere eine Aktion
            test_policy.verify_action("TEST_ACTION", {
                "test_param": "test_value",
                "timestamp": datetime.now()
            })
            
            # Verifiziere eine Eingabe
            input_data = {
                "field1": "value1",
                "field2": 42
            }
            verified_input = test_policy.verify_input(input_data, ["field1", "field2"])
            self.assertEqual(verified_input["field1"], "value1")
            self.assertEqual(verified_input["field2"], 42)
            
            # Teste Fehlerbehandlung
            test_error = ValueError("Testfehler")
            test_policy.handle_error(test_error, "test_method")
            
            self.logger.info("ZTM-Verifizierungsaktionstest erfolgreich")
        except Exception as e:
            self.fail(f"ZTM-Verifizierungsaktionstest fehlgeschlagen: {str(e)}")
    
    def tearDown(self):
        """
        Testumgebung aufräumen.
        """
        # Deaktiviere ZTM in allen Komponenten
        self.market_observer.ztm_policy.deactivate_ztm()
        self.threat_analyzer.ztm_policy.deactivate_ztm()
        self.geopolitical_analyzer.ztm_policy.deactivate_ztm()
        self.social_media_analyzer.ztm_policy.deactivate_ztm()
        self.economic_analyzer.ztm_policy.deactivate_ztm()
        self.paradox_resolver.ztm_policy.deactivate_ztm()
        self.deep_state.deactivate_ztm()


if __name__ == "__main__":
    unittest.main()
