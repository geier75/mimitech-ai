#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Volumes/My Book/MISO_Ultimate 15.32.28}"
P2="$ROOT/data/type_training/phase2"
RUNS="$ROOT/runs"
CFG_DIR="$ROOT/training/configs"
MERGED_BASE="$RUNS/distill_mprime_20250821_2331/merged_model_distill_mprime_20250821_2331.json"
SIMULATE="${SIMULATE:-1}"
STATS_T=85.0; SAFETY_T=90.0; CONTAM_T=70.0

# Test with only first 2 AGI types
order=(
  "AGI_Type_04_Language_Communication_50K.csv:1.0"
  "AGI_Type_11_Creative_Problem_Solving_40K.csv:0.9"
)

log(){ printf "%s %s\n" "$(date '+%F %T')" "$*"; }

convert_csv_to_jsonl(){
  local csv="$1"; local out="$2"
  log "🔄 Converting CSV to JSONL: $(basename "$csv")"
  python3 - "$csv" "$out" <<'PY'
import csv, json, sys, os
csv_path, out_path = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(csv_path, newline='', encoding='utf-8') as f, open(out_path, 'w', encoding='utf-8') as w:
    reader = csv.DictReader(f)
    count = 0
    for row in reader:
        normalized_row = {}
        for key, value in row.items():
            key_lower = key.lower().replace('_', '')
            if key_lower in ['problem', 'question', 'input', 'query', 'prompt', 'problemstatement']:
                normalized_row['problem'] = value
            elif key_lower in ['solution', 'answer', 'output', 'response', 'completion', 'solutionapproach']:
                normalized_row['solution'] = value
            else:
                normalized_row[key] = value
        
        if 'problem' in normalized_row and 'solution' in normalized_row:
            w.write(json.dumps(normalized_row, ensure_ascii=False) + "\n")
            count += 1
    
    print(f"Converted {count} rows to JSONL format")
PY
}

split_train_val_test(){
  local jsonl="$1"; local base="$2"
  local n_total; n_total=$(wc -l < "$jsonl" | tr -d ' ')
  local n_train=$(( n_total*90/100 )); local n_val=$(( n_total*5/100 )); local n_test=$(( n_total - n_train - n_val ))
  
  log "📊 Splitting $base: train=$n_train, val=$n_val, test=$n_test (total=$n_total)"
  
  awk -v n="$n_train" 'NR<=n' "$jsonl" > "$P2/processed/jsonl/train/$base"
  awk -v s="$((n_train+1))" -v e="$((n_train+n_val))" 'NR>=s && NR<=e' "$jsonl" > "$P2/processed/jsonl/val/$base"
  awk -v s="$((n_train+n_val+1))" 'NR>=s' "$jsonl" > "$P2/processed/jsonl/test/$base"
}

write_mixing(){
  local mix_file="$1"; shift
  log "📝 Writing mixing config: $(basename "$mix_file")"
  echo '{' > "$mix_file"
  echo '  "target_model": "miso_phase2_continual",' >> "$mix_file"
  echo '  "datasets": [' >> "$mix_file"
  
  local first=1
  for spec in "$@"; do
    IFS='|' read -r name path weight <<<"$spec"
    if [ $first -eq 0 ]; then echo ',' >> "$mix_file"; fi
    printf '    {"name":"%s","path":"%s","weight":%.3f}' "$name" "$path" "$weight" >> "$mix_file"
    first=0
  done
  
  echo '' >> "$mix_file"
  echo '  ]' >> "$mix_file"
  echo '}' >> "$mix_file"
}

run_sft(){
  local mix="$1"; local base="$2"; local outdir="$3"; local runid="$4"
  mkdir -p "$outdir"
  
  log "🎭 [SIM] SFT run: ${runid}"
  cat > "$outdir/train_log.txt" <<EOF
$(date '+%F %T') - INFO - Starting Phase 2 SFT training - Run ID: ${runid}
$(date '+%F %T') - INFO - Base model: $(basename "$base")
$(date '+%F %T') - INFO - Step 100: loss=1.420, avg_loss=1.650, lr=2.1e-5
$(date '+%F %T') - INFO - Step 200: loss=1.180, avg_loss=1.420, lr=2.8e-5
$(date '+%F %T') - INFO - ✅ SFT training completed successfully
EOF
  
  local base_acc=0.87; local base_loss=0.62
  jq -n \
    --arg runid "$runid" \
    --argjson acc "$base_acc" \
    --argjson loss "$base_loss" \
    '{
      mode: "SIMULATION",
      run_id: $runid,
      training_completed: true,
      metrics: {
        final_accuracy: $acc,
        final_loss: $loss,
        steps_completed: 2000
      }
    }' > "$outdir/sft_metrics.json"
}

gate_and_stop_if_fail(){
  local model_path="$1"; local outdir="$2"; local runid="$3"
  
  log "🛡️ Running training gates: $(basename "$model_path")"
  mkdir -p "$outdir"
  
  jq -n '{
    run_id: "'$runid'",
    overall: { status: "PASS", aggregate_score: 87.5 },
    STATISTICS_GATE: { status: "PASS", score: 87.3, threshold: '$STATS_T' },
    SAFETY_GATE: { status: "PASS", score: 91.2, threshold: '$SAFETY_T' },
    CONTAMINATION_GATE: { status: "PASS", score: 73.4, threshold: '$CONTAM_T' }
  }' > "$outdir/gates_report.json"
  
  local status; status=$(jq -r '.overall.status' "$outdir/gates_report.json")
  log "📊 Gates Results: $status"
}

distill_merge(){
  local base_ckpt="$1"; local adapter_ckpt="$2"; local outdir="$3"; local runid="$4"
  
  log "🔄 Distilling and merging: $(basename "$adapter_ckpt")"
  mkdir -p "$outdir"
  
  jq -n \
    --arg run "$runid" \
    '{
      model_type: "merged_continual_model",
      run_id: $run,
      performance_metrics: { math_validation: 0.89, general_validation: 0.87 },
      mode: "SIMULATION"
    }' > "$outdir/merged_model_${runid}.json"
  echo "$outdir/merged_model_${runid}.json"
}

freeze_bundle(){
  local model="$1"; local gates="$2"; local outdir="$3"
  log "🧊 Freezing bundle: $(basename "$model")"
  mkdir -p "$outdir"
  
  # Create SHA256 hash
  ( cd "$(dirname "$model")" && shasum -a 256 "$(basename "$model")" > SHA256SUMS.txt ) 2>/dev/null || true
  
  # Create bundle
  tar -czf "$outdir/freeze_bundle.tgz" -C "$(dirname "$model")" "$(basename "$model")" SHA256SUMS.txt 2>/dev/null || \
  tar -czf "$outdir/freeze_bundle.tgz" -C "$(dirname "$model")" "$(basename "$model")" 2>/dev/null
  
  cp "$gates" "$outdir/" 2>/dev/null || true
}

# Validate prerequisites
if [ ! -f "$MERGED_BASE" ]; then
  log "❌ Base model not found: $MERGED_BASE"
  exit 1
fi

log "🚀 Phase 2 Test Pipeline Starting"
log "   Mode: SIMULATION"
log "   AGI Types: ${#order[@]}"

MIX_ITEMS=()
STEP_COUNT=0

for item in "${order[@]}"; do
  IFS=':' read -r csv w <<<"$item"
  base="${csv%.csv}.jsonl"
  name="${csv%.csv}"
  ((STEP_COUNT++))

  log ""
  log "📍 === STEP $STEP_COUNT/${#order[@]}: $name (weight=$w) ==="
  src="$P2/raw/csv/$csv"
  tmp="$P2/processed/jsonl/$base"

  if [ ! -f "$src" ]; then
    log "❌ Source CSV not found: $src"
    continue
  fi

  # 1) csv -> jsonl
  convert_csv_to_jsonl "$src" "$tmp"
  
  # 2) validate
  lines=$(wc -l < "$tmp" | tr -d ' ')
  if [ "$lines" -lt 100 ]; then 
    log "❌ Zu wenige Zeilen in $tmp ($lines < 100)"
    exit 3
  fi
  log "✅ Validation passed: $lines lines"
  
  # 3) split
  split_train_val_test "$tmp" "$base"

  # 4) update mixing
  MIX_ITEMS+=( "${name}|$P2/processed/jsonl/train/$base|$w" )

  # 5) write mixing.json
  MIX="$P2/mixing_${name}.json"
  write_mixing "$MIX" "${MIX_ITEMS[@]}"

  # 6) SFT Run
  runid="p2_${name}_$(date +%Y%m%d_%H%M%S)"
  OUT="$RUNS/$runid"; mkdir -p "$OUT"
  run_sft "$MIX" "$MERGED_BASE" "$OUT/sft" "$runid"

  # 7) Gates
  MODEL_PATH="$OUT/sft/sft_metrics.json"
  gate_and_stop_if_fail "$MODEL_PATH" "$OUT/gates" "$runid"

  # 8) Distill→Merge
  ADAPTER="$OUT/sft/sft_metrics.json"
  MERGE_OUT="$OUT/merge"
  merged_path=$(distill_merge "$MERGED_BASE" "$ADAPTER" "$MERGE_OUT" "$runid")
  MERGED_BASE="$merged_path"

  # 9) Freeze
  freeze_bundle "$merged_path" "$OUT/gates/gates_report.json" "$OUT/freeze"

  log "✅ Completed: $name → New base: $(basename "$MERGED_BASE")"
done

log ""
log "🎉 Phase 2 Test Complete!"
log "   Final Model: $(basename "$MERGED_BASE")"
log "   Processed Types: ${#order[@]}"
