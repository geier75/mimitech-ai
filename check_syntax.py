#!/usr/bin/env python3
# Script to check for syntax errors in VX-PSI/core.py

import ast
import sys

def check_syntax(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        
    try:
        ast.parse(content)
        print(f"✅ Keine Syntaxfehler in {file_path}")
        return True
    except SyntaxError as e:
        print(f"❌ Syntaxfehler in {file_path} in Zeile {e.lineno}, Spalte {e.offset}")
        print(f"Fehlermeldung: {e}")
        
        # Extract the problematic lines (a few lines before and after the error)
        lines = content.split('\n')
        start_line = max(0, e.lineno - 5)
        end_line = min(len(lines), e.lineno + 5)
        
        print("\nProblematischer Code-Abschnitt:")
        for i in range(start_line, end_line):
            prefix = ">>>" if i + 1 == e.lineno else "   "
            print(f"{prefix} {i+1}: {lines[i]}")
        
        return False

if __name__ == "__main__":
    file_path = "/Volumes/My Book/VXOR.AI/VX-PSI/core.py"
    check_syntax(file_path)
