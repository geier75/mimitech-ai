#!/usr/bin/env python3
"""
MISO Ultimate Quick Secret Sweep - Fokussiert auf kritische Bereiche
"""

import re
import os
import json
from pathlib import Path

class QuickSecretSweeper:
    def __init__(self):
        self.root = Path('/Volumes/My Book/MISO_Ultimate 15.32.28')
        
        # Nur kritische Verzeichnisse scannen
        self.priority_dirs = [
            'scripts/',
            'miso/',
            'training/', 
            'security/',
            '.github/',
            '*.py',
            '*.json',
            '*.yml',
            '*.yaml'
        ]
        
        # Kritische Secret-Patterns
        self.patterns = {
            'api_key': r'(api[_-]?key|API[_-]?KEY)["\s]*[=:]["\s]*["\'][a-zA-Z0-9]{16,}["\']',
            'secret': r'(secret|SECRET|password|PASSWORD)["\s]*[=:]["\s]*["\'][^"\']{8,}["\']',
            'token': r'(token|TOKEN)["\s]*[=:]["\s]*["\'][a-zA-Z0-9_\-]{20,}["\']',
            'ssh_key': r'-----BEGIN.*PRIVATE KEY-----',
            'github_pat': r'gh[pousr]_[a-zA-Z0-9]{36}',
            'email_real': r'[a-zA-Z0-9._%+-]+@(?!example\.com|test\.com|localhost)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        }

    def quick_scan(self):
        findings = []
        files_scanned = 0
        
        print("🔍 Quick Secret Sweep - Kritische Bereiche")
        
        # Scan root-level Python files
        for py_file in self.root.glob('*.py'):
            findings.extend(self.scan_file(py_file))
            files_scanned += 1
        
        # Scan priority directories
        for dir_pattern in ['scripts', 'miso', 'training', 'security', '.github']:
            dir_path = self.root / dir_pattern
            if dir_path.exists():
                for file_path in dir_path.rglob('*'):
                    if (file_path.is_file() and 
                        file_path.suffix in {'.py', '.json', '.yml', '.yaml', '.sh', '.md'} and
                        not any(skip in str(file_path) for skip in ['__pycache__', '.git', 'test_', '_test'])):
                        
                        findings.extend(self.scan_file(file_path))
                        files_scanned += 1
                        
                        if files_scanned % 10 == 0:
                            print(f"  📁 {files_scanned} Dateien gescannt...")
        
        return {
            'timestamp': '2025-08-19T03:44:57+02:00',
            'files_scanned': files_scanned,
            'findings': findings,
            'summary': {
                'secrets': len([f for f in findings if f['type'] == 'SECRET']),
                'suspicious': len([f for f in findings if f['type'] == 'SUSPICIOUS'])
            }
        }

    def scan_file(self, file_path):
        findings = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            for name, pattern in self.patterns.items():
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    findings.append({
                        'type': 'SECRET' if name in ['api_key', 'secret', 'token', 'ssh_key', 'github_pat'] else 'SUSPICIOUS',
                        'pattern': name,
                        'file': str(file_path.relative_to(self.root)),
                        'line': line_num,
                        'severity': 'HIGH' if name != 'email_real' else 'MEDIUM'
                    })
        except:
            pass
        return findings

if __name__ == '__main__':
    scanner = QuickSecretSweeper()
    result = scanner.quick_scan()
    
    # Speichere Ergebnis
    with open('/Volumes/My Book/MISO_Ultimate 15.32.28/security/QUARANTINE_ZONE/quick_sweep_report.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ QUICK SWEEP COMPLETE")
    print(f"📊 {result['files_scanned']} Dateien gescannt")
    print(f"🚨 {result['summary']['secrets']} Secrets gefunden")
    print(f"⚠️  {result['summary']['suspicious']} Verdächtige Patterns")
    
    if result['summary']['secrets'] > 0:
        print(f"\n🚨 AKTION ERFORDERLICH: {result['summary']['secrets']} potentielle Secrets!")
    else:
        print(f"\n✅ Keine kritischen Secrets in Kern-Bereichen gefunden")
