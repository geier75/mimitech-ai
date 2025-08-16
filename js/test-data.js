/**
 * test-data.js
 * Mock-Daten für das VXOR Benchmark Dashboard Testing
 * 
 * Stellt Testdaten bereit, um die Module zu testen, wenn keine echten API-Endpunkte verfügbar sind.
 * Diese Datei sollte NUR für Testzwecke verwendet werden.
 */

'use strict';

const VXORTestData = {
    // Mock-Komponenten
    components: [
        {
            id: 'matrix-core',
            name: 'Matrix Core',
            description: 'Hochleistungs-Matrix-Berechnungsmodul',
            status: 'ready',
            benchmark_ready: true
        },
        {
            id: 'quantum-sim',
            name: 'Quantum Simulator',
            description: 'Quanten-Simulationsmodul',
            status: 'ready',
            benchmark_ready: true
        },
        {
            id: 'tmath-engine',
            name: 'TMath Engine',
            description: 'Tensor-Mathematik-Engine',
            status: 'ready',
            benchmark_ready: true
        },
        {
            id: 'mlperf-module',
            name: 'MLPerf Module',
            description: 'Machine Learning Benchmark-Modul',
            status: 'ready',
            benchmark_ready: true
        },
        {
            id: 'swe-analyzer',
            name: 'SWE Analyzer',
            description: 'Software Engineering Benchmark-Modul',
            status: 'ready',
            benchmark_ready: true
        },
        {
            id: 'security-scanner',
            name: 'Security Scanner',
            description: 'Sicherheits-Benchmark-Modul',
            status: 'ready',
            benchmark_ready: true
        }
    ],
    
    // Mock-Hardware-Metriken
    hardware: {
        cpu_usage: 45.6,
        memory_usage: 4235.7,
        runtime: 7200 // 2 Stunden in Sekunden
    },
    
    // Mock-Benchmark-Daten
    benchmarks: {
        all: {
            performance: {
                labels: [
                    'Matrix-Mult (1024x1024)', 
                    'Matrix-Vektor (1024x1024)', 
                    'Singulärwertzerlegung (1024x1024)',
                    'Quantum-Gatter H', 
                    'Quantum-Gatter CNOT', 
                    'Polynom-Auswertung',
                    'Integration'
                ],
                values: [145.3, 42.1, 278.9, 5.4, 8.7, 32.1, 67.8]
            },
            memory: {
                timestamps: [
                    new Date(Date.now() - 50000).toISOString(),
                    new Date(Date.now() - 40000).toISOString(),
                    new Date(Date.now() - 30000).toISOString(),
                    new Date(Date.now() - 20000).toISOString(),
                    new Date(Date.now() - 10000).toISOString(),
                    new Date().toISOString()
                ],
                values: [2345, 2478, 2512, 2498, 2456, 2433]
            },
            results: [
                { component: 'Matrix Core', metric: 'Matrix-Mult (1024x1024)', value: 145.3, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Matrix-Mult (2048x2048)', value: 723.6, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Matrix-Mult (4096x4096)', value: 4568.2, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Matrix-Vektor (1024x1024)', value: 42.1, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Singulärwertzerlegung (1024x1024)', value: 278.9, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Singulärwertzerlegung (2048x2048)', value: 1245.7, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Eigenwert (1024x1024)', value: 198.3, unit: 'ms' },
                { component: 'Matrix Core', metric: 'LU-Zerlegung (1024x1024)', value: 112.4, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Inversion (1024x1024)', value: 156.8, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Inversion (2048x2048)', value: 864.2, unit: 'ms' },
                { component: 'Matrix Core', metric: 'FLOPS', value: 125.7, unit: 'GFLOPS' },
                { component: 'Quantum Simulator', metric: 'Quantum-Gatter H', value: 5.4, unit: 'ms' },
                { component: 'Quantum Simulator', metric: 'Quantum-Gatter CNOT', value: 8.7, unit: 'ms' },
                { component: 'Quantum Simulator', metric: 'Deutsch-Jozsa', value: 78.3, unit: 'ms' },
                { component: 'Quantum Simulator', metric: 'Grover', value: 142.6, unit: 'ms' },
                { component: 'Quantum Simulator', metric: 'QFT', value: 95.4, unit: 'ms' },
                { component: 'Quantum Simulator', metric: 'VQE', value: 287.3, unit: 'ms' },
                { component: 'Quantum Simulator', metric: 'Gate-Fidelity', value: 99.92, unit: '%' },
                { component: 'Quantum Simulator', metric: 'Max. simulierbare Qubits', value: 28, unit: 'Qubits' },
                { component: 'Quantum Simulator', metric: 'Verschränkungs-Tiefe', value: 18, unit: 'Ebenen' },
                { component: 'TMath Engine', metric: 'Polynom-Auswertung', value: 32.1, unit: 'ms' },
                { component: 'TMath Engine', metric: 'Nullstellen', value: 45.6, unit: 'ms' },
                { component: 'TMath Engine', metric: 'Integration', value: 67.8, unit: 'ms' },
                { component: 'TMath Engine', metric: 'Differentiation', value: 23.5, unit: 'ms' }
            ]
        },
        
        // Matrix-spezifische Benchmark-Daten
        matrix: {
            performance: {
                labels: [
                    'Matrix-Mult (1024x1024)', 
                    'Matrix-Vektor (1024x1024)', 
                    'Singulärwertzerlegung (1024x1024)',
                    'Eigenwert (1024x1024)',
                    'LU-Zerlegung (1024x1024)',
                    'Inversion (1024x1024)'
                ],
                values: [145.3, 42.1, 278.9, 198.3, 112.4, 156.8]
            },
            memory: {
                timestamps: [
                    new Date(Date.now() - 50000).toISOString(),
                    new Date(Date.now() - 40000).toISOString(),
                    new Date(Date.now() - 30000).toISOString(),
                    new Date(Date.now() - 20000).toISOString(),
                    new Date(Date.now() - 10000).toISOString(),
                    new Date().toISOString()
                ],
                values: [2345, 2478, 2512, 2498, 2456, 2433]
            },
            results: [
                { component: 'Matrix Core', metric: 'Matrix-Mult (1024x1024)', value: 145.3, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Matrix-Mult (2048x2048)', value: 723.6, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Matrix-Mult (4096x4096)', value: 4568.2, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Matrix-Vektor (1024x1024)', value: 42.1, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Singulärwertzerlegung (1024x1024)', value: 278.9, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Singulärwertzerlegung (2048x2048)', value: 1245.7, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Eigenwert (1024x1024)', value: 198.3, unit: 'ms' },
                { component: 'Matrix Core', metric: 'LU-Zerlegung (1024x1024)', value: 112.4, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Inversion (1024x1024)', value: 156.8, unit: 'ms' },
                { component: 'Matrix Core', metric: 'Inversion (2048x2048)', value: 864.2, unit: 'ms' },
                { component: 'Matrix Core', metric: 'FLOPS', value: 125.7, unit: 'GFLOPS' }
            ]
        },
        
        // Quantum-spezifische Benchmark-Daten
        quantum: {
            performance: {
                labels: [
                    'Quantum-Gatter H', 
                    'Quantum-Gatter CNOT', 
                    'Quantum-Gatter X',
                    'Quantum-Gatter Y',
                    'Quantum-Gatter Z'
                ],
                values: [5.4, 8.7, 5.2, 5.3, 5.1]
            },
            results: [
                { component: 'Quantum Simulator', metric: 'Quantum-Gatter H', value: 5.4, unit: 'ms' },
                { component: 'Quantum Simulator', metric: 'Quantum-Gatter CNOT', value: 8.7, unit: 'ms' },
                { component: 'Quantum Simulator', metric: 'Quantum-Gatter X', value: 5.2, unit: 'ms' },
                { component: 'Quantum Simulator', metric: 'Quantum-Gatter Y', value: 5.3, unit: 'ms' },
                { component: 'Quantum Simulator', metric: 'Quantum-Gatter Z', value: 5.1, unit: 'ms' },
                { component: 'Quantum Simulator', metric: 'Deutsch-Jozsa', value: 78.3, unit: 'ms' },
                { component: 'Quantum Simulator', metric: 'Grover', value: 142.6, unit: 'ms' },
                { component: 'Quantum Simulator', metric: 'QFT', value: 95.4, unit: 'ms' },
                { component: 'Quantum Simulator', metric: 'VQE', value: 287.3, unit: 'ms' },
                { component: 'Quantum Simulator', metric: 'Gate-Fidelity', value: 99.92, unit: '%' },
                { component: 'Quantum Simulator', metric: 'Max. simulierbare Qubits', value: 28, unit: 'Qubits' },
                { component: 'Quantum Simulator', metric: 'Verschränkungs-Tiefe', value: 18, unit: 'Ebenen' }
            ]
        },
        
        // TMath-spezifische Benchmark-Daten
        tmaths: {
            performance: {
                labels: [
                    'Polynom-Auswertung', 
                    'Nullstellen', 
                    'Integration',
                    'Differentiation'
                ],
                values: [32.1, 45.6, 67.8, 23.5]
            },
            results: [
                { component: 'TMath Engine', metric: 'Polynom-Auswertung', value: 32.1, unit: 'ms' },
                { component: 'TMath Engine', metric: 'Nullstellen', value: 45.6, unit: 'ms' },
                { component: 'TMath Engine', metric: 'Integration', value: 67.8, unit: 'ms' },
                { component: 'TMath Engine', metric: 'Differentiation', value: 23.5, unit: 'ms' }
            ]
        },
        
        // MLPerf-spezifische Benchmark-Daten
        mlperf: {
            performance: {
                labels: [
                    'Inferenz ResNet-50', 
                    'Inferenz BERT',
                    'Training ResNet',
                    'Training MobileNet'
                ],
                values: [12.3, 28.7, 245.6, 189.2]
            },
            results: [
                // Inferenz-Latenz (in ms)
                { component: 'MLPerf Module', metric: 'Inferenz-Latenz ResNet-50 Batch=1', value: 12.3, unit: 'ms' },
                { component: 'MLPerf Module', metric: 'Inferenz-Latenz ResNet-50 Batch=8', value: 45.8, unit: 'ms' },
                { component: 'MLPerf Module', metric: 'Inferenz-Latenz ResNet-50 Batch=32', value: 167.3, unit: 'ms' },
                { component: 'MLPerf Module', metric: 'Inferenz-Latenz BERT-Base Batch=1', value: 28.7, unit: 'ms' },
                { component: 'MLPerf Module', metric: 'Inferenz-Latenz BERT-Base Batch=8', value: 98.4, unit: 'ms' },
                { component: 'MLPerf Module', metric: 'Inferenz-Latenz BERT-Base Batch=32', value: 342.6, unit: 'ms' },
                { component: 'MLPerf Module', metric: 'Inferenz-Latenz MobileNet Batch=1', value: 5.4, unit: 'ms' },
                { component: 'MLPerf Module', metric: 'Inferenz-Latenz MobileNet Batch=8', value: 19.2, unit: 'ms' },
                { component: 'MLPerf Module', metric: 'Inferenz-Latenz MobileNet Batch=32', value: 62.3, unit: 'ms' },
                { component: 'MLPerf Module', metric: 'Inferenz-Latenz YOLOv5 Batch=1', value: 21.2, unit: 'ms' },
                { component: 'MLPerf Module', metric: 'Inferenz-Latenz YOLOv5 Batch=8', value: 76.9, unit: 'ms' },
                { component: 'MLPerf Module', metric: 'Inferenz-Latenz YOLOv5 Batch=32', value: 232.7, unit: 'ms' },
                
                // Inferenz-Durchsatz (Bilder/Sek)
                { component: 'MLPerf Module', metric: 'Inferenz-Durchsatz ResNet-50 Batch=1', value: 81.3, unit: 'Bilder/Sek' },
                { component: 'MLPerf Module', metric: 'Inferenz-Durchsatz ResNet-50 Batch=8', value: 174.7, unit: 'Bilder/Sek' },
                { component: 'MLPerf Module', metric: 'Inferenz-Durchsatz ResNet-50 Batch=32', value: 191.3, unit: 'Bilder/Sek' },
                { component: 'MLPerf Module', metric: 'Inferenz-Durchsatz BERT-Base Batch=1', value: 34.8, unit: 'Samples/Sek' },
                { component: 'MLPerf Module', metric: 'Inferenz-Durchsatz BERT-Base Batch=8', value: 81.3, unit: 'Samples/Sek' },
                { component: 'MLPerf Module', metric: 'Inferenz-Durchsatz BERT-Base Batch=32', value: 93.4, unit: 'Samples/Sek' },
                
                // Training-Performance (in ms/Batch)
                { component: 'MLPerf Module', metric: 'Training-Verlust ResNet-50 Epoch=1', value: 2.34, unit: '' },
                { component: 'MLPerf Module', metric: 'Training-Verlust ResNet-50 Epoch=2', value: 1.87, unit: '' },
                { component: 'MLPerf Module', metric: 'Training-Verlust ResNet-50 Epoch=5', value: 1.12, unit: '' },
                { component: 'MLPerf Module', metric: 'Training-Verlust ResNet-50 Epoch=10', value: 0.78, unit: '' },
                { component: 'MLPerf Module', metric: 'Validierungs-Genauigkeit ResNet-50 Epoch=1', value: 0.45, unit: '' },
                { component: 'MLPerf Module', metric: 'Validierungs-Genauigkeit ResNet-50 Epoch=2', value: 0.63, unit: '' },
                { component: 'MLPerf Module', metric: 'Validierungs-Genauigkeit ResNet-50 Epoch=5', value: 0.82, unit: '' },
                { component: 'MLPerf Module', metric: 'Validierungs-Genauigkeit ResNet-50 Epoch=10', value: 0.89, unit: '' },
                
                // Precision-Daten
                { component: 'MLPerf Module', metric: 'Präzision ResNet-50 Batch=1', value: 'FP32', unit: '' },
                { component: 'MLPerf Module', metric: 'Präzision ResNet-50 Batch=8', value: 'FP16', unit: '' },
                { component: 'MLPerf Module', metric: 'Präzision ResNet-50 Batch=32', value: 'INT8', unit: '' }
            ]
        },
        
        // SWE-Bench-spezifische Benchmark-Daten
        swe: {
            performance: {
                labels: [
                    'Code-Generierung',
                    'Bugfixing',
                    'Code-Review',
                    'Refactoring'
                ],
                values: [25.6, 42.3, 19.8, 37.2]
            },
            results: [
                // Generierungszeit für verschiedene Komplexitäten und Sprachen
                { component: 'SWE Analyzer', metric: 'Generierungszeit Algorithmus-Implementation Einfach Python', value: 12.4, unit: 'Sek' },
                { component: 'SWE Analyzer', metric: 'Generierungszeit Algorithmus-Implementation Mittel Python', value: 25.6, unit: 'Sek' },
                { component: 'SWE Analyzer', metric: 'Generierungszeit Algorithmus-Implementation Komplex Python', value: 48.3, unit: 'Sek' },
                { component: 'SWE Analyzer', metric: 'Generierungszeit Algorithmus-Implementation Sehr komplex Python', value: 92.7, unit: 'Sek' },
                
                { component: 'SWE Analyzer', metric: 'Code-Qualität Algorithmus-Implementation Einfach Python', value: 8.7, unit: '/10' },
                { component: 'SWE Analyzer', metric: 'Code-Qualität Algorithmus-Implementation Mittel Python', value: 8.2, unit: '/10' },
                { component: 'SWE Analyzer', metric: 'Code-Qualität Algorithmus-Implementation Komplex Python', value: 7.4, unit: '/10' },
                { component: 'SWE Analyzer', metric: 'Code-Qualität Algorithmus-Implementation Sehr komplex Python', value: 6.8, unit: '/10' },
                
                // Generierungszeit für JavaScript
                { component: 'SWE Analyzer', metric: 'Generierungszeit Algorithmus-Implementation Einfach JavaScript', value: 14.2, unit: 'Sek' },
                { component: 'SWE Analyzer', metric: 'Generierungszeit Algorithmus-Implementation Mittel JavaScript', value: 29.8, unit: 'Sek' },
                { component: 'SWE Analyzer', metric: 'Generierungszeit Algorithmus-Implementation Komplex JavaScript', value: 52.1, unit: 'Sek' },
                { component: 'SWE Analyzer', metric: 'Generierungszeit Algorithmus-Implementation Sehr komplex JavaScript', value: 98.4, unit: 'Sek' },
                
                // Bugfixing-Daten für Python
                { component: 'SWE Analyzer', metric: 'Zeit bis Fix Syntax Python', value: 5.2, unit: 'Sek' },
                { component: 'SWE Analyzer', metric: 'Zeit bis Fix Logic Python', value: 18.7, unit: 'Sek' },
                { component: 'SWE Analyzer', metric: 'Zeit bis Fix Performance Python', value: 25.3, unit: 'Sek' },
                { component: 'SWE Analyzer', metric: 'Zeit bis Fix Security Python', value: 34.8, unit: 'Sek' },
                { component: 'SWE Analyzer', metric: 'Zeit bis Fix Edge Case Python', value: 42.3, unit: 'Sek' },
                
                { component: 'SWE Analyzer', metric: 'Erfolgsrate Syntax Python', value: 0.98, unit: '' },
                { component: 'SWE Analyzer', metric: 'Erfolgsrate Logic Python', value: 0.87, unit: '' },
                { component: 'SWE Analyzer', metric: 'Erfolgsrate Performance Python', value: 0.82, unit: '' },
                { component: 'SWE Analyzer', metric: 'Erfolgsrate Security Python', value: 0.76, unit: '' },
                { component: 'SWE Analyzer', metric: 'Erfolgsrate Edge Case Python', value: 0.68, unit: '' },
                
                // Metriktabelle Daten
                { component: 'SWE Analyzer', metric: 'Erfolgsrate Algorithmus-Implementation Python', value: 0.94, unit: '' },
                { component: 'SWE Analyzer', metric: 'Zeit bis Lösung Algorithmus-Implementation Python', value: 45.2, unit: 'Sek' },
                { component: 'SWE Analyzer', metric: 'Codequalität Algorithmus-Implementation Python', value: 8.4, unit: '/10' },
                
                { component: 'SWE Analyzer', metric: 'Erfolgsrate Feature-Erweiterung Python', value: 0.89, unit: '' },
                { component: 'SWE Analyzer', metric: 'Zeit bis Lösung Feature-Erweiterung Python', value: 78.6, unit: 'Sek' },
                { component: 'SWE Analyzer', metric: 'Codequalität Feature-Erweiterung Python', value: 7.9, unit: '/10' },
                
                { component: 'SWE Analyzer', metric: 'Erfolgsrate Bugfixing Python', value: 0.92, unit: '' },
                { component: 'SWE Analyzer', metric: 'Zeit bis Lösung Bugfixing Python', value: 35.7, unit: 'Sek' },
                { component: 'SWE Analyzer', metric: 'Codequalität Bugfixing Python', value: 8.1, unit: '/10' }
            ]
        },
        
        // Security-spezifische Benchmark-Daten
        security: {
            performance: {
                labels: [
                    'Vulnerability Scanning',
                    'Code Security Review',
                    'Penetration Testing',
                    'Threat Modeling'
                ],
                values: [86.4, 72.3, 91.7, 84.2]
            },
            results: [
                // Erkennungs- und Präventionsraten für Web
                { component: 'Security Scanner', metric: 'Erkennungsrate Web Injection', value: 9.3, unit: '/10' },
                { component: 'Security Scanner', metric: 'Erkennungsrate Web XSS', value: 9.5, unit: '/10' },
                { component: 'Security Scanner', metric: 'Erkennungsrate Web Access Control', value: 8.7, unit: '/10' },
                { component: 'Security Scanner', metric: 'Erkennungsrate Web Cryptographic', value: 8.4, unit: '/10' },
                { component: 'Security Scanner', metric: 'Erkennungsrate Web Configuration', value: 9.1, unit: '/10' },
                { component: 'Security Scanner', metric: 'Erkennungsrate Web Authentication', value: 8.9, unit: '/10' },
                
                { component: 'Security Scanner', metric: 'Präventionsrate Web Injection', value: 8.4, unit: '/10' },
                { component: 'Security Scanner', metric: 'Präventionsrate Web XSS', value: 8.2, unit: '/10' },
                { component: 'Security Scanner', metric: 'Präventionsrate Web Access Control', value: 7.8, unit: '/10' },
                { component: 'Security Scanner', metric: 'Präventionsrate Web Cryptographic', value: 7.6, unit: '/10' },
                { component: 'Security Scanner', metric: 'Präventionsrate Web Configuration', value: 8.3, unit: '/10' },
                { component: 'Security Scanner', metric: 'Präventionsrate Web Authentication', value: 7.7, unit: '/10' },
                
                // Erkennungs- und Präventionsraten für Mobile
                { component: 'Security Scanner', metric: 'Erkennungsrate Mobile Injection', value: 8.1, unit: '/10' },
                { component: 'Security Scanner', metric: 'Erkennungsrate Mobile XSS', value: 8.5, unit: '/10' },
                { component: 'Security Scanner', metric: 'Erkennungsrate Mobile Access Control', value: 7.9, unit: '/10' },
                { component: 'Security Scanner', metric: 'Erkennungsrate Mobile Cryptographic', value: 9.1, unit: '/10' },
                { component: 'Security Scanner', metric: 'Erkennungsrate Mobile Configuration', value: 8.4, unit: '/10' },
                { component: 'Security Scanner', metric: 'Erkennungsrate Mobile Authentication', value: 9.3, unit: '/10' },
                
                // Werkzeug-Effizienz für Web
                { component: 'Security Scanner', metric: 'Erkennungseffizienz Web Statische Analyse', value: 8.7, unit: '/10' },
                { component: 'Security Scanner', metric: 'Erkennungseffizienz Web Dynamische Analyse', value: 9.2, unit: '/10' },
                { component: 'Security Scanner', metric: 'Erkennungseffizienz Web Fuzzing', value: 7.8, unit: '/10' },
                { component: 'Security Scanner', metric: 'Erkennungseffizienz Web Penetration Testing', value: 9.6, unit: '/10' },
                { component: 'Security Scanner', metric: 'Erkennungseffizienz Web Compositional Analysis', value: 8.3, unit: '/10' },
                
                { component: 'Security Scanner', metric: 'False Positives Web Statische Analyse', value: 5.2, unit: '/10' },
                { component: 'Security Scanner', metric: 'False Positives Web Dynamische Analyse', value: 3.8, unit: '/10' },
                { component: 'Security Scanner', metric: 'False Positives Web Fuzzing', value: 6.1, unit: '/10' },
                { component: 'Security Scanner', metric: 'False Positives Web Penetration Testing', value: 2.4, unit: '/10' },
                { component: 'Security Scanner', metric: 'False Positives Web Compositional Analysis', value: 4.7, unit: '/10' }
            ],
            
            // Beispielhafte Sicherheitsmatrix
            securityMatrix: {
                Web: {
                    Authentifizierung: { Critical: 1, High: 3, Medium: 5, Low: 7 },
                    Zugriffskontrolle: { Critical: 2, High: 4, Medium: 6, Low: 3 },
                    Datenvalidierung: { Critical: 3, High: 5, Medium: 2, Low: 4 },
                    Kryptographie: { Critical: 0, High: 2, Medium: 4, Low: 6 },
                    Konfiguration: { Critical: 1, High: 2, Medium: 5, Low: 7 }
                },
                Mobile: {
                    Authentifizierung: { Critical: 2, High: 3, Medium: 4, Low: 6 },
                    Zugriffskontrolle: { Critical: 1, High: 2, Medium: 5, Low: 4 },
                    Datenvalidierung: { Critical: 0, High: 3, Medium: 6, Low: 8 },
                    Kryptographie: { Critical: 2, High: 4, Medium: 3, Low: 5 },
                    Konfiguration: { Critical: 1, High: 3, Medium: 4, Low: 6 }
                },
                Cloud: {
                    Authentifizierung: { Critical: 1, High: 4, Medium: 5, Low: 3 },
                    Zugriffskontrolle: { Critical: 2, High: 6, Medium: 3, Low: 2 },
                    Datenvalidierung: { Critical: 0, High: 2, Medium: 4, Low: 5 },
                    Kryptographie: { Critical: 1, High: 3, Medium: 5, Low: 4 },
                    Konfiguration: { Critical: 3, High: 5, Medium: 4, Low: 2 }
                },
                IoT: {
                    Authentifizierung: { Critical: 2, High: 5, Medium: 6, Low: 3 },
                    Zugriffskontrolle: { Critical: 1, High: 4, Medium: 5, Low: 2 },
                    Datenvalidierung: { Critical: 1, High: 3, Medium: 4, Low: 6 },
                    Kryptographie: { Critical: 3, High: 6, Medium: 4, Low: 2 },
                    Konfiguration: { Critical: 2, High: 5, Medium: 7, Low: 4 }
                },
                Netzwerk: {
                    Authentifizierung: { Critical: 1, High: 3, Medium: 4, Low: 5 },
                    Zugriffskontrolle: { Critical: 2, High: 5, Medium: 3, Low: 2 },
                    Datenvalidierung: { Critical: 0, High: 2, Medium: 5, Low: 7 },
                    Kryptographie: { Critical: 1, High: 3, Medium: 4, Low: 5 },
                    Konfiguration: { Critical: 2, High: 4, Medium: 6, Low: 3 }
                }
            }
        }
    }
};



// Mock-API-Endpunkte
if (typeof VXORUtils !== 'undefined') {
    // Überschreibe die API-Funktionen mit Mock-Versionen
    const originalFetchComponent = VXORUtils.fetchComponentStatus;
    const originalFetchHardware = VXORUtils.fetchHardwareMetrics;
    const originalFetchBenchmark = VXORUtils.fetchBenchmarkData;
    
    // Mock-Implementierungen
    VXORUtils.fetchComponentStatus = async function() {
        console.log('[TEST] Verwende Mock-Komponenten-Daten');
        return Promise.resolve({ components: VXORTestData.components });
    };
    
    VXORUtils.fetchHardwareMetrics = async function() {
        console.log('[TEST] Verwende Mock-Hardware-Daten');
        return Promise.resolve(VXORTestData.hardware);
    };
    
    VXORUtils.fetchBenchmarkData = async function(component, category = 'all') {
        console.log(`[TEST] Verwende Mock-Benchmark-Daten für Kategorie: ${category}`);
        return Promise.resolve(VXORTestData.benchmarks[category] || VXORTestData.benchmarks.all);
    };
    
    // Methode zum Zurücksetzen auf Original-Implementierungen
    VXORUtils.resetToOriginalImplementation = function() {
        VXORUtils.fetchComponentStatus = originalFetchComponent;
        VXORUtils.fetchHardwareMetrics = originalFetchHardware;
        VXORUtils.fetchBenchmarkData = originalFetchBenchmark;
        console.log('[TEST] Zurückgesetzt auf Original-API-Implementierungen');
    };
    
    console.log('[TEST] Mock-API-Implementierungen wurden aktiviert');
}