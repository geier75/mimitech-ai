import json, csv, glob, os, hashlib, random, re, sys
random.seed(42)

ROOT="/Volumes/My Book/MISO_Ultimate 15.32.28/data/type_training/phase1"
SRC=f"{ROOT}/raw/agi_types"
DST_T=f"{ROOT}/processed/jsonl/train"
DST_V=f"{ROOT}/processed/jsonl/val"
DST_S=f"{ROOT}/processed/jsonl/test"
os.makedirs(DST_T, exist_ok=True)
os.makedirs(DST_V, exist_ok=True)
os.makedirs(DST_S, exist_ok=True)

def norm(x): return re.sub(r"\s+"," ", (x or "").strip())
def to_pair(row):
    # Handle standard instruction/output format
    if "instruction" in row and "output" in row:
        u = row["instruction"].strip()
        if row.get("input","").strip():
            u += f"\n\nInput:\n{row['input'].strip()}"
        return u, row["output"].strip()
    # Handle prompt/completion format
    if "prompt" in row and "completion" in row:
        return row["prompt"].strip(), row["completion"].strip()
    
    # Handle different AGI training data formats
    problem_field = None
    solution_field = None
    
    # Detect problem and solution fields (case insensitive)
    problem_fields = ["problem_statement", "Pattern_Description", "Abstract_Problem", "Temporal_Problem", "Statistical_Problem", "Creative_Challenge", "Scenario_Text", "scenario", "task", "question"]
    solution_fields = ["solution_approach", "mathematical_solution", "Detection_Approach", "Expected_Analysis", "Expected_Reasoning", "Conceptual_Framework", "Sequential_Logic", "Mathematical_Framework", "Expected_Process", "Approach", "explanation_method", "expected_solution", "answer", "solution", "completion"]
    
    for field in problem_fields:
        for col in row.keys():
            if col.lower() == field.lower() and row[col].strip():
                problem_field = col
                break
        if problem_field:
            break
    
    for field in solution_fields:
        for col in row.keys():
            if col.lower() == field.lower() and row[col].strip():
                solution_field = col
                break
        if solution_field:
            break
    
    # Alternative: look for steps-based fields
    if not solution_field:
        for field in ["required_steps", "solution_steps", "communication_steps", "reasoning_steps"]:
            if field in row and row[field].strip():
                solution_field = field
                break
    
    if problem_field and solution_field:
        problem = row[problem_field].strip()
        solution = row[solution_field].strip()
        if problem and solution:
            # Build context from available metadata
            context_parts = []
            for ctx_field in ["domain", "reasoning_type", "communication_type", "problem_type", "pattern_type"]:
                if row.get(ctx_field, "").strip():
                    context_parts.append(f"{ctx_field.replace('_', ' ').title()}: {row[ctx_field].strip()}")
            if row.get("difficulty_level", "").strip():
                context_parts.append(f"Difficulty: {row['difficulty_level'].strip()}")
            
            context = " | ".join(context_parts)
            if context:
                prompt = f"[{context}]\n\nProblem: {problem}"
            else:
                prompt = f"Problem: {problem}"
            
            # Add additional context if available
            additional_info = []
            for info_field in ["required_steps", "proof_verification", "alternative_methods"]:
                if info_field != solution_field and row.get(info_field, "").strip():
                    additional_info.append(f"{info_field.replace('_', ' ').title()}: {row[info_field].strip()}")
            
            if additional_info:
                completion = f"Solution: {solution}\n\n" + "\n\n".join(additional_info)
            else:
                completion = f"Solution: {solution}"
                
            return prompt, completion
    
    return None

def key(u,a): return hashlib.sha256((norm(u)+"\n###\n"+norm(a)).encode("utf-8")).hexdigest()

paths = sorted(glob.glob(f"{SRC}/*.csv"))
if not paths:
    print("❌ Keine CSVs gefunden. Erst ./get_phase1_data.sh ausführen.", file=sys.stderr)
    sys.exit(1)

for p in paths:
    base=os.path.splitext(os.path.basename(p))[0]
    seen=set(); bucket=[]
    with open(p, newline='', encoding='utf-8') as f:
        r=csv.DictReader(f)
        for row in r:
            pair=to_pair(row)
            if not pair: continue
            u,a=pair; k=key(u,a)
            if k in seen: continue
            seen.add(k)
            bucket.append({"prompt": u, "completion": a})
    random.shuffle(bucket)
    n=len(bucket); n_val=max(1,int(n*0.02)); n_test=max(1,int(n*0.02))
    train=bucket[:n-n_val-n_test]; val=bucket[n-n_val-n_test:n-n_test]; test=bucket[n-n_test:]
    for arr, dst in ((train,DST_T),(val,DST_V),(test,DST_S)):
        out=f"{dst}/{base}.jsonl"
        with open(out,"w",encoding="utf-8") as g:
            for ex in arr: g.write(json.dumps(ex, ensure_ascii=False)+"\n")
    print(f"✅ {base}: train={len(train)} val={len(val)} test={len(test)} (total={n})")
print("🎉 Konvertierung abgeschlossen.")
