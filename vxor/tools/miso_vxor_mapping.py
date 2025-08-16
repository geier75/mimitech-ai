#!/usr/bin/env python3
"""
MISO → vXor Mapping-Generator
Erstellt eine Mapping-Tabelle zwischen MISO-Namensräumen und vXor-Äquivalenten
"""

import os
import json
import csv
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Konfiguration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_JSON = os.path.join(PROJECT_ROOT, "migration_analysis_results.json")
MAPPING_CSV = os.path.join(PROJECT_ROOT, "miso_vxor_mapping.csv")
MAPPING_JSON = os.path.join(PROJECT_ROOT, "miso_vxor_mapping.json")
MIGRATION_PLAN_MD = os.path.join(PROJECT_ROOT, "namespace_migration_plan.md")

# Vorschlagsregeln für vXor-Mapping
# Diese Regeln definieren, wie MISO-Module nach vXor umbenannt werden sollen
MAPPING_RULES = [
    # Spezifische Module-Mappings
    {"from": r"miso\.security\.vxor_blackbox", "to": "vxor.security"},
    {"from": r"miso\.timeline\.echo_prime", "to": "vxor.chronos"},
    {"from": r"miso\.code\.m_code", "to": "vxor.code"},
    {"from": r"miso\.lang\.mlingua", "to": "vxor.lingua"},
    {"from": r"miso\.logic\.qlogik", "to": "vxor.logik"},
    {"from": r"miso\.math\.t_mathematics", "to": "vxor.math.tensor"},
    {"from": r"miso\.math\.mprime", "to": "vxor.math.symbol"},
    {"from": r"miso\.simulation", "to": "vxor.sim"},
    {"from": r"miso\.monitoring", "to": "vxor.ztm"},
    {"from": r"miso\.omega", "to": "vxor.core"},
    {"from": r"miso\.nexus", "to": "vxor.nexus"},
    {"from": r"miso\.vision", "to": "vxor.vision"},
    
    # Standardregel für alle anderen
    {"from": r"miso\.(\w+)", "to": r"vxor.\1"}
]

def load_analysis_data() -> Dict:
    """Lädt die Analyseergebnisse aus der JSON-Datei."""
    try:
        with open(ANALYSIS_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Fehler: Analysedatei {ANALYSIS_JSON} nicht gefunden.")
        print("Bitte führen Sie zuerst namespace_analyzer.py aus.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Fehler: Die Datei {ANALYSIS_JSON} enthält ungültiges JSON.")
        exit(1)

def extract_modules(data: Dict) -> Set[str]:
    """Extrahiert alle eindeutigen MISO-Module aus den Analyseergebnissen."""
    modules = set()
    
    for file_result in data["file_results"]:
        for imp in file_result["imports"]:
            if imp["type"] == "import":
                modules.add(imp["module"])
            else:  # from_import
                modules.add(imp["module"])
        
        for usage in file_result["usages"]:
            modules.add(usage["module"])
        
        for inheritance in file_result["class_inheritance"]:
            modules.add(inheritance["module"])
    
    return modules

def suggest_vxor_mapping(module: str) -> str:
    """Schlägt ein vXor-Äquivalent für ein MISO-Modul vor."""
    for rule in MAPPING_RULES:
        if re.match(rule["from"], module):
            # Wenn es ein Regex-Ersetzungsmuster ist
            if "\\1" in rule["to"]:
                return re.sub(rule["from"], rule["to"], module)
            else:
                # Wenn der Modulpfad nach dem Präfix beibehalten werden soll
                module_suffix = module[module.find(".")+1:]
                base_path = rule["from"].rstrip("\\.*")
                suffix_path = module[len(base_path):]
                return rule["to"] + suffix_path
    
    # Fallback: Ersetze nur den Präfix
    return module.replace("miso.", "vxor.")

def create_mapping(modules: Set[str]) -> Dict[str, str]:
    """Erstellt ein Mapping von MISO-Modulen zu vXor-Modulen."""
    mapping = {}
    for module in sorted(modules):
        mapping[module] = suggest_vxor_mapping(module)
    return mapping

def export_mapping_to_csv(mapping: Dict[str, str]) -> None:
    """Exportiert das Mapping in eine CSV-Datei."""
    with open(MAPPING_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['MISO-Modul', 'vXor-Modul', 'Bestätigt'])
        
        for miso_module, vxor_module in sorted(mapping.items()):
            writer.writerow([miso_module, vxor_module, 'Nein'])  # 'Nein' bedeutet, dass das Mapping noch überprüft werden muss

def export_mapping_to_json(mapping: Dict[str, str]) -> None:
    """Exportiert das Mapping in eine JSON-Datei."""
    mapping_data = {
        "mapping": [
            {"miso_module": miso, "vxor_module": vxor, "confirmed": False}
            for miso, vxor in mapping.items()
        ]
    }
    
    with open(MAPPING_JSON, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2)

def create_migration_plan(mapping: Dict[str, str], stats: Dict) -> None:
    """Erstellt einen Migrationsplan als Markdown-Datei."""
    with open(MIGRATION_PLAN_MD, 'w', encoding='utf-8') as f:
        f.write("# Namespace-Migrationsplan: MISO → vXor\n\n")
        
        # Übersicht
        f.write("## Übersicht\n\n")
        f.write(f"- **Gesamtzahl der Python-Dateien:** {stats['total_files']}\n")
        f.write(f"- **Dateien mit MISO-Referenzen:** {stats['files_with_miso_refs']} ({stats['files_with_miso_refs']/stats['total_files']*100:.1f}%)\n")
        f.write(f"- **MISO-Imports gefunden:** {stats['total_miso_imports']}\n")
        f.write(f"- **MISO-Verwendungen gefunden:** {stats['total_miso_usages']}\n")
        f.write(f"- **Eindeutige MISO-Module gefunden:** {len(mapping)}\n\n")
        
        # Top-Module
        f.write("## Häufigste MISO-Module\n\n")
        f.write("### Top 10 importierte MISO-Module\n\n")
        f.write("| MISO-Modul | Anzahl | Geplantes vXor-Modul |\n")
        f.write("|------------|--------|---------------------|\n")
        
        for i, (module, count) in enumerate(list(stats["most_imported_modules"].items())[:10]):
            vxor_module = mapping.get(module, "Nicht definiert")
            f.write(f"| {module} | {count} | {vxor_module} |\n")
        
        f.write("\n### Top 10 verwendete MISO-Module\n\n")
        f.write("| MISO-Modul | Anzahl | Geplantes vXor-Modul |\n")
        f.write("|------------|--------|---------------------|\n")
        
        for i, (module, count) in enumerate(list(stats["most_used_modules"].items())[:10]):
            vxor_module = mapping.get(module, "Nicht definiert")
            f.write(f"| {module} | {count} | {vxor_module} |\n")
        
        # Vollständiges Mapping
        f.write("\n## Vollständiges Modul-Mapping\n\n")
        f.write("| MISO-Modul | vXor-Modul |\n")
        f.write("|------------|------------|\n")
        
        for miso_module, vxor_module in sorted(mapping.items()):
            f.write(f"| {miso_module} | {vxor_module} |\n")
        
        # Migrationsschritte
        f.write("\n## Migrationsschritte\n\n")
        f.write("1. **Review des Mappings**\n")
        f.write("   - Überprüfung und Anpassung des automatisch generierten Mappings\n")
        f.write("   - Klärung von Abhängigkeiten und möglichen Konflikten\n\n")
        
        f.write("2. **Vorbereitung der vXor-Struktur**\n")
        f.write("   - Erstellen der Verzeichnisstruktur für vXor-Module\n")
        f.write("   - Anlegen der erforderlichen `__init__.py`-Dateien\n\n")
        
        f.write("3. **Entwicklung des Migrationsskripts**\n")
        f.write("   - Auf Basis des bestätigten Mappings\n")
        f.write("   - Integration in die CI-Pipeline\n\n")
        
        f.write("4. **Testphase**\n")
        f.write("   - Migration in einer Testumgebung\n")
        f.write("   - Ausführung aller Tests gegen die neue Struktur\n\n")
        
        f.write("5. **Migration in Produktion**\n")
        f.write("   - Schrittweise Migration nach Modulen\n")
        f.write("   - Kontinuierliche Tests und Validierung\n\n")
        
        f.write("6. **Rollback-Strategie**\n")
        f.write("   - Vorübergehende Kompatibilitätsschicht für alte Importe\n")
        f.write("   - Fallback-Mechanismus bei Problemen\n\n")
        
        # Zeitplan
        f.write("## Zeitplan\n\n")
        f.write("| Meilenstein | Datum | Verantwortlich |\n")
        f.write("|-------------|-------|----------------|\n")
        f.write("| Mapping Review abgeschlossen | 2025-07-19 | Lead Dev |\n")
        f.write("| vXor-Struktur vorbereitet | 2025-07-20 | Dev-Team |\n")
        f.write("| Migrationsskript entwickelt | 2025-07-21 | DevOps |\n")
        f.write("| Testphase abgeschlossen | 2025-07-25 | QA |\n")
        f.write("| Migration abgeschlossen | 2025-07-30 | Dev-Team |\n")

def main():
    print("MISO → vXor Mapping-Generator")
    print("============================")
    
    # Lade Analysedaten
    print("Lade Analyseergebnisse...")
    analysis_data = load_analysis_data()
    stats = analysis_data["statistics"]
    
    # Extrahiere Module
    print("Extrahiere eindeutige MISO-Module...")
    modules = extract_modules(analysis_data)
    print(f"{len(modules)} eindeutige MISO-Module gefunden.")
    
    # Erstelle Mapping
    print("Erstelle vXor-Mapping...")
    mapping = create_mapping(modules)
    
    # Exportiere Mapping
    print("Exportiere Mapping...")
    export_mapping_to_csv(mapping)
    export_mapping_to_json(mapping)
    print(f"Mapping-CSV gespeichert unter: {MAPPING_CSV}")
    print(f"Mapping-JSON gespeichert unter: {MAPPING_JSON}")
    
    # Erstelle Migrationsplan
    print("Erstelle Migrationsplan...")
    create_migration_plan(mapping, stats)
    print(f"Migrationsplan gespeichert unter: {MIGRATION_PLAN_MD}")
    
    print("\nMapping-Generierung abgeschlossen!")
    print(f"Bitte überprüfen und bestätigen Sie das Mapping in: {MAPPING_CSV}")

if __name__ == "__main__":
    main()
