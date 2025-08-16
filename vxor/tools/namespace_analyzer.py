#!/usr/bin/env python3
"""
MISO → vXor Namespace Migration Analyzer
Identifiziert und katalogisiert alle MISO-Namensraum-Referenzen im Projekt
"""

import os
import re
import json
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Set

# Konfiguration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "migration_analysis_results.json")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "migration_analysis_results.csv")
EXCLUDED_DIRS = [
    ".git", "__pycache__", "venv", "env", ".env", 
    "node_modules", "dist", "build", ".idea", ".vscode"
]

# Regex-Muster für verschiedene Arten von MISO-Referenzen
IMPORT_PATTERNS = [
    r"import\s+(miso\.[\w\.]+)(?:\s+as\s+(\w+))?",
    r"from\s+(miso\.[\w\.]+)\s+import\s+(.+)"
]
USAGE_PATTERNS = [
    r"(miso\.[\w\.]+)\(",
    r"(miso\.[\w\.]+)\.",
    r"(miso\.[\w\.]+)\["
]
CLASS_PATTERNS = [
    r"class\s+\w+\(.*?(miso\.[\w\.]+).*?\):",
]

def should_exclude(path: str) -> bool:
    """Überprüft, ob ein Pfad ausgeschlossen werden soll."""
    return any(excluded in path for excluded in EXCLUDED_DIRS)

def find_python_files() -> List[str]:
    """Findet alle Python-Dateien im Projekt."""
    python_files = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Überspringe ausgeschlossene Verzeichnisse
        dirs[:] = [d for d in dirs if not should_exclude(d)]
        
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    return python_files

def analyze_file(filepath: str) -> Dict:
    """Analysiert eine Python-Datei auf MISO-Referenzen."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    relative_path = os.path.relpath(filepath, PROJECT_ROOT)
    
    result = {
        "file": relative_path,
        "imports": [],
        "usages": [],
        "class_inheritance": []
    }
    
    # Imports analysieren
    for pattern in IMPORT_PATTERNS:
        for match in re.finditer(pattern, content):
            if pattern == IMPORT_PATTERNS[0]:  # import miso.x [as y]
                module = match.group(1)
                alias = match.group(2) if match.group(2) else None
                result["imports"].append({
                    "type": "import",
                    "module": module,
                    "alias": alias,
                    "line": content[:match.start()].count('\n') + 1
                })
            else:  # from miso.x import y
                module = match.group(1)
                imports = match.group(2)
                result["imports"].append({
                    "type": "from_import",
                    "module": module,
                    "imports": [i.strip() for i in imports.split(',')],
                    "line": content[:match.start()].count('\n') + 1
                })
    
    # Verwendungen analysieren
    for pattern in USAGE_PATTERNS:
        for match in re.finditer(pattern, content):
            module = match.group(1)
            result["usages"].append({
                "module": module,
                "line": content[:match.start()].count('\n') + 1
            })
    
    # Vererbungen analysieren
    for pattern in CLASS_PATTERNS:
        for match in re.finditer(pattern, content):
            module = match.group(1)
            result["class_inheritance"].append({
                "module": module,
                "line": content[:match.start()].count('\n') + 1
            })
    
    return result

def analyze_all_files() -> Dict:
    """Analysiert alle Python-Dateien und erstellt einen Bericht."""
    files = find_python_files()
    results = []
    
    total_files = len(files)
    processed = 0
    
    print(f"Analysiere {total_files} Python-Dateien...")
    
    for file in files:
        result = analyze_file(file)
        results.append(result)
        
        processed += 1
        if processed % 50 == 0 or processed == total_files:
            print(f"Fortschritt: {processed}/{total_files} Dateien ({processed/total_files*100:.1f}%)")
    
    # Statistiken erstellen
    stats = generate_statistics(results)
    
    full_report = {
        "statistics": stats,
        "file_results": results
    }
    
    return full_report

def generate_statistics(results: List[Dict]) -> Dict:
    """Generiert Statistiken aus den Analyseergebnissen."""
    stats = {
        "total_files": len(results),
        "files_with_miso_refs": 0,
        "total_miso_imports": 0,
        "total_miso_usages": 0,
        "total_miso_inheritance": 0,
        "most_imported_modules": defaultdict(int),
        "most_used_modules": defaultdict(int)
    }
    
    for result in results:
        has_miso = False
        
        if result["imports"]:
            stats["total_miso_imports"] += len(result["imports"])
            has_miso = True
            for imp in result["imports"]:
                if imp["type"] == "import":
                    stats["most_imported_modules"][imp["module"]] += 1
                else:  # from_import
                    stats["most_imported_modules"][imp["module"]] += len(imp["imports"])
        
        if result["usages"]:
            stats["total_miso_usages"] += len(result["usages"])
            has_miso = True
            for usage in result["usages"]:
                stats["most_used_modules"][usage["module"]] += 1
        
        if result["class_inheritance"]:
            stats["total_miso_inheritance"] += len(result["class_inheritance"])
            has_miso = True
        
        if has_miso:
            stats["files_with_miso_refs"] += 1
    
    # Sortiere die häufigsten Module
    stats["most_imported_modules"] = dict(sorted(
        stats["most_imported_modules"].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:20])  # Top 20
    
    stats["most_used_modules"] = dict(sorted(
        stats["most_used_modules"].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:20])  # Top 20
    
    return stats

def export_to_csv(data: Dict) -> None:
    """Exportiert die Ergebnisse in eine CSV-Datei."""
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['Datei', 'Referenz-Typ', 'Modul', 'Zeile', 'Details'])
        
        # Daten
        for file_result in data["file_results"]:
            file_path = file_result["file"]
            
            for imp in file_result["imports"]:
                if imp["type"] == "import":
                    details = f"alias: {imp['alias']}" if imp["alias"] else ""
                    writer.writerow([file_path, 'Import', imp["module"], imp["line"], details])
                else:
                    details = f"imports: {', '.join(imp['imports'])}"
                    writer.writerow([file_path, 'From Import', imp["module"], imp["line"], details])
            
            for usage in file_result["usages"]:
                writer.writerow([file_path, 'Verwendung', usage["module"], usage["line"], ""])
            
            for inheritance in file_result["class_inheritance"]:
                writer.writerow([file_path, 'Vererbung', inheritance["module"], inheritance["line"], ""])

def main():
    print("MISO → vXor Namespace Migration Analyzer")
    print("=======================================")
    
    results = analyze_all_files()
    
    # Speichere Ergebnisse als JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Exportiere als CSV für bessere Lesbarkeit
    export_to_csv(results)
    
    print("\nAnalyse abgeschlossen!")
    print(f"JSON-Bericht gespeichert unter: {OUTPUT_JSON}")
    print(f"CSV-Bericht gespeichert unter: {OUTPUT_CSV}")
    
    # Zeige Zusammenfassung
    stats = results["statistics"]
    print("\nZUSAMMENFASSUNG:")
    print(f"Insgesamt analysiert: {stats['total_files']} Python-Dateien")
    print(f"Dateien mit MISO-Referenzen: {stats['files_with_miso_refs']} ({stats['files_with_miso_refs']/stats['total_files']*100:.1f}%)")
    print(f"MISO-Imports gefunden: {stats['total_miso_imports']}")
    print(f"MISO-Verwendungen gefunden: {stats['total_miso_usages']}")
    print(f"MISO-Vererbungen gefunden: {stats['total_miso_inheritance']}")
    
    print("\nTop 5 importierte MISO-Module:")
    for i, (module, count) in enumerate(list(stats["most_imported_modules"].items())[:5]):
        print(f"{i+1}. {module}: {count} Mal")
    
    print("\nTop 5 verwendete MISO-Module:")
    for i, (module, count) in enumerate(list(stats["most_used_modules"].items())[:5]):
        print(f"{i+1}. {module}: {count} Mal")

if __name__ == "__main__":
    main()
