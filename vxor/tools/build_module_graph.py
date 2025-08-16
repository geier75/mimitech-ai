#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR_AI 15.32.28 - Module Graph Generator
Erzeugt einen vollständigen Abhängigkeitsgraphen aller Python-Module im Projekt.
"""

import os
import sys
import importlib
import pkgutil
import inspect
import re
from pathlib import Path
from collections import defaultdict
import argparse
from typing import Dict, List, Set, Tuple, Optional

# Graphviz für die SVG-Generierung
try:
    import graphviz
except ImportError:
    print("Warning: graphviz package not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'graphviz'])
    import graphviz

# Farben für verschiedene Modultypen
COLORS = {
    'core': '#4285F4',       # Blau
    'security': '#DB4437',   # Rot
    'ml': '#F4B400',         # Gelb
    'tests': '#0F9D58',      # Grün
    'tools': '#9C27B0',      # Lila
    'external': '#757575',   # Grau
}


class ModuleGraph:
    """Generiert und visualisiert einen Abhängigkeitsgraphen für Python-Module."""
    
    def __init__(self, root_dir: str, exclude_dirs: List[str] = None):
        """
        Initialisiert den ModuleGraph.
        
        Args:
            root_dir: Wurzelverzeichnis des Projekts
            exclude_dirs: Liste von Verzeichnissen, die ausgeschlossen werden sollen
        """
        self.root_dir = os.path.abspath(root_dir)
        self.exclude_dirs = exclude_dirs or ['venv', '.git', '__pycache__', 'build', 'dist']
        self.modules: Dict[str, Set[str]] = defaultdict(set)
        self.module_types: Dict[str, str] = {}
        self.all_python_files: List[str] = []
        self.external_deps: Dict[str, Set[str]] = defaultdict(set)
    
    def find_all_python_files(self) -> List[str]:
        """Findet alle Python-Dateien im Projekt."""
        python_files = []
        
        for root, dirs, files in os.walk(self.root_dir):
            # Ausgeschlossene Verzeichnisse überspringen
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    rel_path = os.path.relpath(os.path.join(root, file), self.root_dir)
                    python_files.append(rel_path)
        
        return sorted(python_files)
    
    def module_to_node_name(self, module_path: str) -> str:
        """Konvertiert einen Modulpfad in einen Knotennamen für den Graphen."""
        # .py entfernen und / in . umwandeln
        node_name = module_path.replace('/', '.').replace('\\', '.')
        if node_name.endswith('.py'):
            node_name = node_name[:-3]
        return node_name
    
    def determine_module_type(self, module_path: str) -> str:
        """Bestimmt den Typ eines Moduls basierend auf seinem Pfad."""
        if module_path.startswith('security/'):
            return 'security'
        elif module_path.startswith('tests/'):
            return 'tests'
        elif module_path.startswith('tools/'):
            return 'tools'
        elif any(ml_dir in module_path for ml_dir in ['ml/', 'ai/', 'models/']):
            return 'ml'
        else:
            return 'core'
    
    def parse_imports(self, file_path: str) -> Tuple[Set[str], Set[str]]:
        """
        Analysiert eine Python-Datei auf Import-Anweisungen.
        
        Returns:
            Tuple mit (interne_imports, externe_imports)
        """
        internal_imports = set()
        external_imports = set()
        
        try:
            with open(os.path.join(self.root_dir, file_path), 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Reguläre Importe finden: import x, import x.y, import x.y as z
            import_regex = r'^\s*import\s+([\w.]+)(?:\s+as\s+\w+)?'
            # From-Importe finden: from x import y, from x.y import z
            from_regex = r'^\s*from\s+([\w.]+)\s+import'
            
            # Multiline-Kommentare und Strings ignorieren
            content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
            content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)
            
            # Zeile für Zeile analysieren
            for line in content.split('\n'):
                # Einzeilige Kommentare ignorieren
                if '#' in line:
                    line = line.split('#')[0]
                
                # Import-Anweisungen finden
                import_match = re.search(import_regex, line)
                from_match = re.search(from_regex, line)
                
                if import_match:
                    module_name = import_match.group(1)
                    self._categorize_import(module_name, internal_imports, external_imports)
                
                if from_match:
                    module_name = from_match.group(1)
                    self._categorize_import(module_name, internal_imports, external_imports)
        
        except Exception as e:
            print(f"Error parsing {file_path}: {str(e)}", file=sys.stderr)
        
        return internal_imports, external_imports
    
    def _categorize_import(self, module_name: str, internal_imports: Set[str], external_imports: Set[str]):
        """Kategorisiert einen Import als intern oder extern."""
        # Das Toplevel-Modul extrahieren
        top_module = module_name.split('.')[0]
        
        # Prüfen, ob es sich um ein internes Modul handelt
        is_internal = False
        for py_file in self.all_python_files:
            py_module = self.module_to_node_name(py_file)
            if py_module == module_name or py_module.startswith(module_name + '.'):
                internal_imports.add(module_name)
                is_internal = True
                break
            # Auch prüfen, ob das Modul ein Untermodul eines Projektmoduls ist
            elif module_name.startswith(py_module.split('.')[0] + '.'):
                internal_imports.add(module_name)
                is_internal = True
                break
        
        # Wenn nicht intern und nicht in der Standardbibliothek, dann extern
        if not is_internal and not self._is_stdlib_module(top_module):
            external_imports.add(top_module)  # Nur das Toplevel-Modul für externe Abhängigkeiten
    
    def _is_stdlib_module(self, module_name: str) -> bool:
        """Überprüft, ob ein Modul zur Python-Standardbibliothek gehört."""
        # Liste bekannter Standard-Module
        stdlib_modules = set([
            'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'concurrent',
            'contextlib', 'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum',
            'functools', 'glob', 'hashlib', 'hmac', 'http', 'importlib', 'inspect',
            'io', 'itertools', 'json', 'logging', 'math', 'mimetypes', 'multiprocessing',
            'operator', 'os', 'pathlib', 'pickle', 'pkgutil', 'platform', 'pprint',
            'queue', 're', 'shutil', 'signal', 'socket', 'sqlite3', 'stat',
            'string', 'struct', 'subprocess', 'sys', 'tempfile', 'threading',
            'time', 'timeit', 'traceback', 'types', 'typing', 'unittest', 'urllib',
            'uuid', 'warnings', 'weakref', 'xml', 'zipfile', 'zlib', 'configparser',
            'ctypes', 'random', 'ssl'
        ])
        
        return module_name in stdlib_modules
    
    def build_graph(self) -> None:
        """Baut den vollständigen Abhängigkeitsgraphen auf."""
        self.all_python_files = self.find_all_python_files()
        
        for py_file in self.all_python_files:
            module_name = self.module_to_node_name(py_file)
            self.module_types[module_name] = self.determine_module_type(py_file)
            
            internal_deps, external_deps = self.parse_imports(py_file)
            
            for dep in internal_deps:
                self.modules[module_name].add(dep)
            
            self.external_deps[module_name] = external_deps
    
    def generate_graph_viz(self, output_file: Optional[str] = None, format: str = 'svg') -> graphviz.Digraph:
        """
        Generiert eine Visualisierung des Abhängigkeitsgraphen mit Graphviz.
        
        Args:
            output_file: Ausgabedatei (ohne Erweiterung)
            format: Ausgabeformat ('svg', 'png', etc.)
        
        Returns:
            graphviz.Digraph-Objekt
        """
        # Graphviz-Objekt erstellen
        dot = graphviz.Digraph(
            name='MISO Module Dependencies',
            comment='VXOR_AI 15.32.28 Module Dependency Graph',
            format=format,
            engine='dot'
        )
        
        # Graph-Attribute
        dot.attr(rankdir='LR', size='12,12', ratio='compress')
        dot.attr('node', shape='box', style='filled', fontname='Arial', fontsize='10')
        dot.attr('edge', fontname='Arial', fontsize='8')
        
        # Cluster für Modultypen erstellen
        clusters = defaultdict(list)
        for module, module_type in self.module_types.items():
            clusters[module_type].append(module)
        
        # Knoten nach Modultyp gruppieren
        for module_type, modules in clusters.items():
            with dot.subgraph(name=f'cluster_{module_type}') as c:
                c.attr(label=module_type.upper(), fontname='Arial Bold', fontsize='12',
                      style='filled', color=COLORS.get(module_type, '#CCCCCC'),
                      fillcolor=f"{COLORS.get(module_type, '#CCCCCC')}22")
                
                for module in sorted(modules):
                    # Knotennamen kürzen, wenn zu lang
                    node_label = module
                    if len(node_label) > 25:
                        parts = node_label.split('.')
                        if len(parts) > 2:
                            node_label = '.'.join([parts[0], '...', parts[-1]])
                    
                    c.node(module, label=node_label, 
                           color=COLORS.get(module_type, '#CCCCCC'),
                           fillcolor=f"{COLORS.get(module_type, '#CCCCCC')}44")
        
        # Knoten für externe Abhängigkeiten
        with dot.subgraph(name='cluster_external') as c:
            c.attr(label='EXTERNAL DEPENDENCIES', fontname='Arial Bold', fontsize='12',
                  style='filled', color=COLORS['external'], fillcolor=f"{COLORS['external']}22")
            
            # Alle externen Abhängigkeiten sammeln
            all_ext_deps = set()
            for deps in self.external_deps.values():
                all_ext_deps.update(deps)
            
            for ext_dep in sorted(all_ext_deps):
                c.node(f"ext_{ext_dep}", label=ext_dep, 
                       color=COLORS['external'], fillcolor=f"{COLORS['external']}44")
        
        # Kanten für interne Abhängigkeiten hinzufügen
        for module, deps in self.modules.items():
            for dep in deps:
                if dep in self.module_types:  # Nur vorhandene Module verbinden
                    dot.edge(module, dep)
        
        # Kanten für externe Abhängigkeiten hinzufügen
        for module, ext_deps in self.external_deps.items():
            for ext_dep in ext_deps:
                dot.edge(module, f"ext_{ext_dep}", style='dashed')
        
        # Ausgabedatei rendern, wenn angegeben
        if output_file:
            dot.render(output_file, cleanup=True)
        
        return dot
    
    def print_summary(self) -> None:
        """Gibt eine Zusammenfassung des Abhängigkeitsgraphen aus."""
        # Anzahl der Module pro Typ
        module_counts = defaultdict(int)
        for module_type in self.module_types.values():
            module_counts[module_type] += 1
        
        # Externe Abhängigkeiten
        all_ext_deps = set()
        for deps in self.external_deps.values():
            all_ext_deps.update(deps)
        
        # Zusammenfassung ausgeben
        print("\nVXOR_AI 15.32.28 Module Graph Summary")
        print("=" * 50)
        print(f"Total Python Files: {len(self.all_python_files)}")
        print(f"Total Modules: {len(self.module_types)}")
        print("\nModules by Type:")
        for module_type, count in sorted(module_counts.items()):
            print(f"  - {module_type.upper()}: {count}")
        
        print(f"\nExternal Dependencies: {len(all_ext_deps)}")
        for dep in sorted(all_ext_deps):
            # Zählen, wie viele Module von dieser Abhängigkeit abhängen
            dep_count = sum(1 for deps in self.external_deps.values() if dep in deps)
            print(f"  - {dep}: used by {dep_count} modules")
        
        # Module mit den meisten Abhängigkeiten
        print("\nMost Connected Modules:")
        
        # Eingehende Verbindungen
        incoming = defaultdict(int)
        for module, deps in self.modules.items():
            for dep in deps:
                incoming[dep] += 1
        
        # Top 5 der meistgenutzten Module
        top_incoming = sorted(incoming.items(), key=lambda x: x[1], reverse=True)[:5]
        print("  Most Imported Modules:")
        for module, count in top_incoming:
            if module in self.module_types:  # Nur interne Module
                print(f"  - {module}: imported by {count} modules")
        
        # Top 5 der Module mit den meisten Abhängigkeiten
        top_outgoing = sorted([(m, len(d)) for m, d in self.modules.items()], key=lambda x: x[1], reverse=True)[:5]
        print("  Modules with Most Dependencies:")
        for module, count in top_outgoing:
            print(f"  - {module}: imports {count} modules")


def main():
    """Hauptfunktion zum Ausführen des Tools."""
    parser = argparse.ArgumentParser(description='MISO Module Graph Generator')
    parser.add_argument('--root', type=str, default='.', help='Root directory of the project')
    parser.add_argument('--output', type=str, default='module_graph', help='Output file (without extension)')
    parser.add_argument('--format', type=str, default='svg', choices=['svg', 'png', 'pdf'],
                      help='Output format')
    parser.add_argument('--exclude', type=str, nargs='+', default=None,
                      help='Directories to exclude')
    
    args = parser.parse_args()
    
    graph = ModuleGraph(args.root, args.exclude)
    graph.build_graph()
    graph.generate_graph_viz(args.output, args.format)
    graph.print_summary()


if __name__ == '__main__':
    main()
