#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-VISION Batch-Verarbeitungsmodul

Dieses Modul bietet hochoptimierte Funktionen für die Batch-Verarbeitung von Bildern
mit dynamischer Batch-Größenanpassung, effizienter Speicherverwaltung und
Hardware-spezifischen Optimierungen.
"""

from vxor.vision.batch.batch_processor import BatchProcessor, BatchConfig
from vxor.vision.batch.scheduler import AdaptiveBatchScheduler
from vxor.vision.batch.memory_manager import MemoryManager

__all__ = [
    'BatchProcessor',
    'BatchConfig',
    'AdaptiveBatchScheduler',
    'MemoryManager'
]
