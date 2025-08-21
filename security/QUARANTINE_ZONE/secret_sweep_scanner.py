#!/usr/bin/env python3
"""
MISO Ultimate Secret & PII Sweep Scanner
Scans codebase for potential secrets, API keys, personal information
"""

import re
import os
import json
from pathlib import Path
from typing import List, Dict, Set

class SecretSweeper:
    def __init__(self, scan_root: str):
        self.scan_root = Path(scan_root)
        self.findings = []
        
        # Patterns for common secrets
        self.secret_patterns = {
            'api_key': r'api[_-]?key["\s]*[=:]["\s]*[a-zA-Z0-9]{20,}',
            'secret_key': r'secret[_-]?key["\s]*[=:]["\s]*[a-zA-Z0-9]{20,}',
            'password': r'password["\s]*[=:]["\s]*["\'][^"\']{8,}["\']',
            'token': r'token["\s]*[=:]["\s]*[a-zA-Z0-9_\-]{20,}',
            'ssh_key': r'-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----',
            'github_token': r'gh[pousr]_[a-zA-Z0-9]{36}',
            'aws_key': r'AKIA[0-9A-Z]{16}',
        }
        
        # Patterns for PII
        self.pii_patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'(\+49|0)[1-9][0-9]{1,4}[0-9]{6,8}',
            'user_home': r'/Users/[a-zA-Z0-9_-]+',
            'username': r'(user|username)["\s]*[=:]["\s]*[a-zA-Z0-9_-]{3,}["\s]*',
        }
        
        # File extensions to scan
        self.scan_extensions = {'.py', '.js', '.json', '.yaml', '.yml', '.md', '.txt', '.sh'}
        
        # Paths to skip
        self.skip_paths = {
            '.git', '__pycache__', '.venv', 'node_modules', 
            'whisper.cpp', '.benchmarks', 'venv'
        }

    def should_scan_file(self, file_path: Path) -> bool:
        """Check if file should be scanned"""
        if file_path.suffix not in self.scan_extensions:
            return False
            
        # Skip files in excluded directories
        for part in file_path.parts:
            if part in self.skip_paths:
                return False
        
        return True

    def scan_file(self, file_path: Path) -> List[Dict]:
        """Scan individual file for secrets and PII"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            lines = content.split('\n')
            
            # Scan for secrets
            for pattern_name, pattern in self.secret_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    findings.append({
                        'type': 'SECRET',
                        'pattern': pattern_name,
                        'file': str(file_path.relative_to(self.scan_root)),
                        'line': line_num,
                        'context': lines[line_num - 1].strip() if line_num <= len(lines) else '',
                        'severity': 'HIGH'
                    })
            
            # Scan for PII
            for pattern_name, pattern in self.pii_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    # Skip obvious false positives for emails
                    if pattern_name == 'email':
                        matched_text = match.group()
                        if any(domain in matched_text.lower() for domain in 
                              ['example.com', 'test.com', 'localhost', 'placeholder']):
                            continue
                    
                    findings.append({
                        'type': 'PII',
                        'pattern': pattern_name,
                        'file': str(file_path.relative_to(self.scan_root)),
                        'line': line_num,
                        'context': lines[line_num - 1].strip() if line_num <= len(lines) else '',
                        'severity': 'MEDIUM'
                    })
                    
        except Exception as e:
            findings.append({
                'type': 'ERROR',
                'pattern': 'scan_error',
                'file': str(file_path.relative_to(self.scan_root)),
                'line': 0,
                'context': f'Error scanning file: {str(e)}',
                'severity': 'LOW'
            })
            
        return findings

    def scan_directory(self) -> Dict:
        """Scan entire directory tree"""
        print(f"🔍 Starting secret & PII sweep of {self.scan_root}")
        
        scanned_files = 0
        total_findings = []
        
        for file_path in self.scan_root.rglob('*'):
            if file_path.is_file() and self.should_scan_file(file_path):
                findings = self.scan_file(file_path)
                total_findings.extend(findings)
                scanned_files += 1
                
                if scanned_files % 50 == 0:
                    print(f"  📁 Scanned {scanned_files} files...")
        
        # Categorize findings
        secrets = [f for f in total_findings if f['type'] == 'SECRET']
        pii = [f for f in total_findings if f['type'] == 'PII']
        errors = [f for f in total_findings if f['type'] == 'ERROR']
        
        report = {
            'scan_metadata': {
                'timestamp': '2025-08-19T01:41:17+02:00',
                'scan_root': str(self.scan_root),
                'files_scanned': scanned_files,
                'total_findings': len(total_findings)
            },
            'summary': {
                'secrets_found': len(secrets),
                'pii_found': len(pii),
                'scan_errors': len(errors)
            },
            'findings': {
                'secrets': secrets,
                'pii': pii,
                'errors': errors
            }
        }
        
        return report

if __name__ == '__main__':
    scanner = SecretSweeper('/Volumes/My Book/MISO_Ultimate 15.32.28')
    report = scanner.scan_directory()
    
    # Save report
    output_file = Path('/Volumes/My Book/MISO_Ultimate 15.32.28/security/QUARANTINE_ZONE/secret_sweep_report.json')
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n🔍 SECRET & PII SWEEP COMPLETE")
    print(f"📊 Files scanned: {report['scan_metadata']['files_scanned']}")
    print(f"🚨 Secrets found: {report['summary']['secrets_found']}")
    print(f"⚠️  PII found: {report['summary']['pii_found']}")
    print(f"❌ Errors: {report['summary']['scan_errors']}")
    print(f"📄 Full report: {output_file}")
    
    if report['summary']['secrets_found'] > 0:
        print(f"\n🚨 CRITICAL: {report['summary']['secrets_found']} potential secrets found!")
        print("   ACTION REQUIRED: Review and remediate before any external sharing")
