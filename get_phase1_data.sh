#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/MimiTechAi/vxor-training-data.git"

# Staging auf interner SSD (vermeidet I/O-Fehler der externen Platte)
STAGE_DIR="${HOME}/vxor_training_stage"

# Zielpfad auf deiner externen Platte (für Training)
TARGET_ROOT="/Volumes/My Book/MISO_Ultimate 15.32.28/data/type_training/phase1"
TARGET_DIR="${TARGET_ROOT}/raw/agi_types"

# Phase-1 Auswahl
PHASE1_FILES=(
  "AGI_Type_04_Language_Communication_50K.csv"
  "AGI_Type_11_Creative_Problem_Solving_40K.csv"
  "AGI_Type_06_Pattern_Recognition_Analysis_50K.csv"
  "AGI_Type_07_Abstract_Reasoning_50K.csv"
  "AGI_Type_08_Temporal_Sequential_Logic_50K.csv"
  "AGI_Type_10_Probability_Statistics_50K.csv"
  "AGI_Type_02_Mathematics_Logic_50K.csv"
  "AGI_Type_42_Knowledge_Transfer_50K.csv"
  "AGI_Type_18_Hypothesis_Testing_40K.csv"
  "AGI_Type_17_Metacognitive_Reasoning_40K.csv"
  "AGI_Type_41_Self_Reflection_Improvement_50K.csv"
  "AGI_Type_47_Explainable_AI_50K.csv"
  "AGI_Type_48_Bias_Detection_50K.csv"
)

echo "==> 📦 Vorbereitungen"
mkdir -p "${STAGE_DIR}" "${TARGET_DIR}" "${TARGET_ROOT}/checksums" "${TARGET_ROOT}/manifests"

command -v git >/dev/null || { echo "❌ git fehlt"; exit 1; }
command -v git-lfs >/dev/null || { echo "❌ git-lfs fehlt (macOS: brew install git-lfs)"; exit 1; }
git lfs install --skip-repo

echo "==> ⬇️  Sparse-Clone nur für benötigte Dateien"
cd "${STAGE_DIR}"
if [[ ! -d vxor-training-data ]]; then
  git clone --filter=blob:none --no-checkout "${REPO_URL}" vxor-training-data
fi
cd vxor-training-data
git sparse-checkout init --cone
git sparse-checkout set "${PHASE1_FILES[@]}"
git checkout

echo "==> 📥 LFS-Objekte holen"
git lfs fetch --all
git lfs checkout || true
git lfs pull || true

echo "==> 🔍 Verifikation (keine LFS-Pointer, sinnvolle Größe)"
for f in "${PHASE1_FILES[@]}"; do
  [[ -f "$f" ]] || { echo "❌ fehlt: $f"; exit 1; }
  if head -1 "$f" | grep -q "git-lfs.github.com/spec/v1"; then
    echo "❌ $f ist noch ein LFS-Pointer. Prüfe git lfs pull."; exit 1
  fi
  sz=$(wc -c <"$f" | tr -d ' ')
  [[ "$sz" -ge 1024 ]] || { echo "❌ $f zu klein ($sz B)"; exit 1; }
done
echo "✅ Verifikation bestanden."

echo "==> 🧾 Checksums schreiben (SHA256)"
( shasum -a 256 "${PHASE1_FILES[@]}" ) > "${TARGET_ROOT}/checksums/SHA256SUMS_phase1.txt"

echo "==> 📤 Kopiere an Ziel (robust, externes Laufwerk)"
rsync -ah --info=progress2 --inplace --partial "${PHASE1_FILES[@]}" "${TARGET_DIR}/"

echo "==> ✅ Fertig. Zieldateien liegen in:"
echo "    ${TARGET_DIR}"
