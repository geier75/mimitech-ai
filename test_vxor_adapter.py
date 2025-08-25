#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-Skript für den VXOR-Adapter
"""

import sys
import os
from vxor.agents.vx_adapter_core import VXORAdapter

# Initialisiere VXOR-Adapter
adapter = VXORAdapter()

# Gib Statusübersicht aus
print("\nVXOR-Module Status:\n" + "-"*20)
module_status = adapter.get_module_status()
for module_name, status_data in module_status.items():
    print(f"{module_name}: {status_data['status']}")

# Teste spezifische Module
print("\nTest VX-CHRONOS:")
try:
    chronos = adapter.get_module("VX-CHRONOS")
    print(f"VX-CHRONOS erfolgreich geladen, verfügbare Klassen: {', '.join(dir(chronos))}")
except Exception as e:
    print(f"Fehler bei VX-CHRONOS: {e}")

print("\nTest VX-GESTALT:")
try:
    gestalt = adapter.get_module("VX-GESTALT")
    print(f"VX-GESTALT erfolgreich geladen, verfügbare Klassen: {', '.join(dir(gestalt))}")
except Exception as e:
    print(f"Fehler bei VX-GESTALT: {e}")
