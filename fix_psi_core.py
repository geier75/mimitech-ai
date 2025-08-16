#!/usr/bin/env python3
# Script to fix the syntax error in VX-PSI/core.py

import os
import re

def fix_psi_core_file():
    file_path = "/Volumes/My Book/VXOR.AI/VX-PSI/core.py"
    backup_path = file_path + ".bak"
    
    # Make backup
    with open(file_path, 'r') as file:
        content = file.read()
    
    with open(backup_path, 'w') as file:
        file.write(content)
    
    print(f"Backup created at: {backup_path}")
    
    # Fix the _clamp_metrics method - complete implementation
    clamp_metrics_pattern = r'def _clamp_metrics\(self\) -> None:\s+""".*?""".*?'
    
    fixed_implementation = '''def _clamp_metrics(self) -> None:
        """Begrenzt alle Metriken auf den Bereich [0.0, 1.0]."""
        for key in self.awareness_metrics:
            self.awareness_metrics[key] = max(0.0, min(1.0, self.awareness_metrics[key]))
'''
    
    # Try to find and replace the broken method
    if re.search(clamp_metrics_pattern, content, re.DOTALL):
        modified_content = re.sub(clamp_metrics_pattern, fixed_implementation, content, flags=re.DOTALL)
        
        # Check if there are any truncated lines at the end
        truncated_pattern = r'\(Content truncated due to size limit\. Use line ranges to read in chunks\)'
        if re.search(truncated_pattern, modified_content):
            modified_content = re.sub(truncated_pattern, '', modified_content)
        
        # Write the fixed content
        with open(file_path, 'w') as file:
            file.write(modified_content)
        
        print(f"Fixed file written to: {file_path}")
        return True
    else:
        print("Could not find the _clamp_metrics method to fix")
        return False

if __name__ == "__main__":
    success = fix_psi_core_file()
    print("Fix completed successfully!" if success else "Fix failed!")
