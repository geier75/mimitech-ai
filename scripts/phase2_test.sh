#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Volumes/My Book/MISO_Ultimate 15.32.28}"
P2="$ROOT/data/type_training/phase2"
RUNS="$ROOT/runs"
CFG_DIR="$ROOT/training/configs"
MERGED_BASE="$RUNS/distill_mprime_20250821_2331/merged_model_distill_mprime_20250821_2331.json"  # Start-Basis aus Phase-1
SIMULATE="${SIMULATE:-1}"    # 1=trocken/simuliert, 0=echtes Training (erfordert Deps)
STATS_T=85.0; SAFETY_T=90.0; CONTAM_T=70.0

# Test with only first 2 AGI types
order=( \
  "AGI_Type_04_Language_Communication_50K.csv:1.0" \
  "AGI_Type_11_Creative_Problem_Solving_40K.csv:0.9" \
)
