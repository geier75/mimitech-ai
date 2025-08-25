#!/usr/bin/env python3
"""Enhanced Validation Pipeline for Phase 2 Real MPS Training"""

import os, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(os.environ.get("ROOT", "."))
MODEL_DIR = ROOT/"runs/phase2_mps_real"
RESULTS_DIR = ROOT/"runs/validation_results"

def run_comprehensive_validation():
    """Run comprehensive validation suite"""
    print("🚀 Starting Enhanced Validation Pipeline")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(MODEL_DIR),
        "validation_tests": {}
    }
    
    # Test 1: Model Loading
    print("\n📦 Test 1: Model Loading Validation")
    try:
        if MODEL_DIR.exists():
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
            results["validation_tests"]["model_loading"] = {
                "status": "PASS",
                "details": f"Model loaded successfully from {MODEL_DIR}"
            }
            print("✅ Model loading: PASS")
        else:
            results["validation_tests"]["model_loading"] = {
                "status": "SKIP",
                "details": "Model directory not found - training in progress"
            }
            print("⏳ Model loading: SKIP (training in progress)")
    except Exception as e:
        results["validation_tests"]["model_loading"] = {
            "status": "FAIL",
            "details": f"Error: {str(e)}"
        }
        print(f"❌ Model loading: FAIL - {e}")
    
    # Test 2: Training Process Health
    print("\n🔍 Test 2: Training Process Health Check")
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        training_processes = [line for line in result.stdout.split('\n') if 'train_mps.py' in line]
        
        if training_processes:
            process_info = training_processes[0].split()
            cpu_usage = float(process_info[2])
            memory_usage = float(process_info[3])
            
            results["validation_tests"]["training_health"] = {
                "status": "PASS",
                "details": f"Training active - CPU: {cpu_usage}%, Memory: {memory_usage}%",
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage
            }
            print(f"✅ Training health: PASS (CPU: {cpu_usage}%, Memory: {memory_usage}%)")
        else:
            results["validation_tests"]["training_health"] = {
                "status": "FAIL",
                "details": "No active training process found"
            }
            print("❌ Training health: FAIL - No active process")
    except Exception as e:
        results["validation_tests"]["training_health"] = {
            "status": "ERROR",
            "details": f"Health check error: {str(e)}"
        }
        print(f"❌ Training health: ERROR - {e}")
    
    # Test 3: Data Pipeline Validation
    print("\n📊 Test 3: Data Pipeline Validation")
    try:
        mixing_config = ROOT/"training/configs/mixing_phase2.json"
        if mixing_config.exists():
            with open(mixing_config) as f:
                config = json.load(f)
            
            total_datasets = len(config["train"])
            available_datasets = 0
            
            for item in config["train"]:
                jsonl_path = Path(item["path"])
                if not jsonl_path.exists():
                    jsonl_path = ROOT / jsonl_path
                if jsonl_path.exists():
                    available_datasets += 1
            
            results["validation_tests"]["data_pipeline"] = {
                "status": "PASS" if available_datasets == total_datasets else "PARTIAL",
                "details": f"{available_datasets}/{total_datasets} datasets available",
                "total_datasets": total_datasets,
                "available_datasets": available_datasets
            }
            
            if available_datasets == total_datasets:
                print(f"✅ Data pipeline: PASS ({available_datasets}/{total_datasets} datasets)")
            else:
                print(f"⚠️ Data pipeline: PARTIAL ({available_datasets}/{total_datasets} datasets)")
        else:
            results["validation_tests"]["data_pipeline"] = {
                "status": "FAIL",
                "details": "Mixing configuration not found"
            }
            print("❌ Data pipeline: FAIL - Config not found")
    except Exception as e:
        results["validation_tests"]["data_pipeline"] = {
            "status": "ERROR",
            "details": f"Pipeline validation error: {str(e)}"
        }
        print(f"❌ Data pipeline: ERROR - {e}")
    
    # Test 4: VXOR Integration Status
    print("\n🔗 Test 4: VXOR Integration Status")
    try:
        from vxor.agents.vx_adapter_core import VXORAdapter
        adapter = VXORAdapter()
        module_status = adapter.get_module_status()
        
        loaded_modules = [name for name, data in module_status.items() if data['status'] == 'loaded']
        error_modules = [name for name, data in module_status.items() if data['status'] == 'error']
        
        results["validation_tests"]["vxor_integration"] = {
            "status": "PASS" if len(loaded_modules) >= 4 else "PARTIAL",
            "details": f"{len(loaded_modules)} modules loaded, {len(error_modules)} errors",
            "loaded_modules": loaded_modules,
            "error_modules": error_modules
        }
        
        if len(loaded_modules) >= 4:
            print(f"✅ VXOR integration: PASS ({len(loaded_modules)} modules loaded)")
        else:
            print(f"⚠️ VXOR integration: PARTIAL ({len(loaded_modules)} modules loaded)")
            
    except Exception as e:
        results["validation_tests"]["vxor_integration"] = {
            "status": "ERROR",
            "details": f"VXOR integration error: {str(e)}"
        }
        print(f"❌ VXOR integration: ERROR - {e}")
    
    # Test 5: Infrastructure Readiness
    print("\n🏗️ Test 5: Infrastructure Readiness")
    required_scripts = [
        "scripts/validate_gates_real.py",
        "scripts/merge_adapters_real.py",
        "scripts/train_mps.py"
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if not (ROOT / script).exists():
            missing_scripts.append(script)
    
    results["validation_tests"]["infrastructure"] = {
        "status": "PASS" if not missing_scripts else "FAIL",
        "details": f"Required scripts check: {len(required_scripts) - len(missing_scripts)}/{len(required_scripts)} found",
        "missing_scripts": missing_scripts
    }
    
    if not missing_scripts:
        print("✅ Infrastructure: PASS (all required scripts present)")
    else:
        print(f"❌ Infrastructure: FAIL - Missing: {', '.join(missing_scripts)}")
    
    # Save Results
    results_file = RESULTS_DIR / f"validation_report_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("📋 ENHANCED VALIDATION SUMMARY")
    print("="*60)
    
    total_tests = len(results["validation_tests"])
    passed_tests = sum(1 for test in results["validation_tests"].values() if test["status"] == "PASS")
    
    for test_name, test_result in results["validation_tests"].items():
        status_icon = {
            "PASS": "✅",
            "FAIL": "❌", 
            "ERROR": "⚠️",
            "SKIP": "⏳",
            "PARTIAL": "🔶"
        }.get(test_result["status"], "❓")
        
        print(f"{status_icon} {test_name.replace('_', ' ').title()}: {test_result['status']}")
    
    print(f"\n🏆 Overall Score: {passed_tests}/{total_tests} tests passed")
    print(f"📊 Report saved: {results_file}")
    
    return results

if __name__ == "__main__":
    run_comprehensive_validation()
