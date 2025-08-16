#!/usr/bin/env python3
"""
🔥 MISO Ultimate MCP Smoke Test Suite
Automatisierte Tests aller MCP-Tools mit echten API-Calls
"""
import json, sys, requests, concurrent.futures, pathlib, time
from datetime import datetime

def load_templates():
    """Load MCP prompt templates"""
    try:
        templates_file = pathlib.Path("mcp_prompt_templates.json")
        if not templates_file.exists():
            print("❌ mcp_prompt_templates.json not found!")
            return []
        
        data = json.loads(templates_file.read_text())
        templates = []
        
        for tool_name, template in data["templates"].items():
            templates.append({
                "tool_name": tool_name,
                "mcp_payload": template["mcp_payload"]
            })
        
        return templates
    except Exception as e:
        print(f"❌ Error loading templates: {e}")
        return []

def fire_call(tpl):
    """Execute single MCP tool call"""
    url = "http://127.0.0.1:8003/mcp"
    start_time = time.perf_counter()
    
    try:
        resp = requests.post(url, json=tpl["mcp_payload"], timeout=60)
        duration = time.perf_counter() - start_time
        status = resp.status_code
        
        if status == 200:
            result = resp.json()
            ok = "✅"
            detail = f"→ {status} ({duration:.2f}s)"
            if "result" in result and "test_id" in result["result"]:
                detail += f" [ID: {result['result']['test_id'][:8]}...]"
        else:
            ok = "❌"
            detail = f"→ {status} ({duration:.2f}s) - {resp.text[:100]}"
            
    except requests.exceptions.Timeout:
        ok = "⏰"
        detail = "→ TIMEOUT (60s)"
    except Exception as e:
        ok = "💥"
        detail = f"→ ERROR: {str(e)[:50]}"
    
    return f"{ok} {tpl['tool_name']:25} {detail}"

def main():
    """Main smoke test execution"""
    print("🔥 MISO Ultimate MCP Smoke Test Suite")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().isoformat()}")
    print()
    
    # Load templates
    templates = load_templates()
    if not templates:
        print("❌ No templates loaded - aborting!")
        return 1
    
    print(f"📦 Loaded {len(templates)} templates")
    print("🚀 Executing smoke tests...")
    print()
    
    # Execute tests
    t0 = time.perf_counter()
    success_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fire_call, templates, chunksize=1))
        
        for result in results:
            print(result)
            if result.startswith("✅"):
                success_count += 1
    
    # Summary
    total_time = time.perf_counter() - t0
    success_rate = (success_count / len(templates)) * 100
    
    print()
    print("=" * 60)
    print(f"⏱️  Completed in {total_time:.2f}s")
    print(f"✅ Success: {success_count}/{len(templates)} ({success_rate:.1f}%)")
    
    if success_count == len(templates):
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("⚠️  Some tests failed - check output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
