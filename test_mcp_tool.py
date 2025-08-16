#!/usr/bin/env python3
"""
🧪 MCP Tool Test Script
======================

Test script to verify the MCP benchmark tool works with real API calls.

Author: MISO Ultimate Team
Date: 30.07.2025
"""

import asyncio
import json
import httpx
from datetime import datetime


async def test_mcp_benchmark_tool():
    """Test the MCP benchmark tool with real API calls"""
    
    print("🧪 Testing MCP Benchmark Tool")
    print("=" * 50)
    
    # Test payload
    test_payload = {
        "tool": "benchmark_matrix",
        "params": {
            "preset": "quick",
            "iterations": 50,
            "matrix_sizes": [64, 128],
            "backend_preference": "auto",
            "timeout_seconds": 120
        }
    }
    
    print(f"📤 Sending MCP request:")
    print(json.dumps(test_payload, indent=2))
    print()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Send MCP request
            response = await client.post(
                "http://127.0.0.1:8001/mcp",
                json=test_payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📥 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ MCP Tool Response:")
                print(json.dumps(result, indent=2))
                
                # Check if it's a real API response
                if "api_response" in result.get("result", {}):
                    api_resp = result["result"]["api_response"]
                    if "test_id" in api_resp:
                        print(f"\n🎯 Real API Response Detected!")
                        print(f"   Test ID: {api_resp['test_id']}")
                        print(f"   Status: {api_resp.get('status', 'unknown')}")
                        print(f"   Message: {api_resp.get('message', 'N/A')}")
                        return True
                    else:
                        print("\n❌ No test_id in API response - might be dummy data")
                        return False
                else:
                    print("\n❌ No api_response field - not calling real API")
                    return False
            else:
                print(f"❌ HTTP Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def test_tool_listing():
    """Test the tool listing endpoint"""
    print("\n🔍 Testing Tool Listing")
    print("=" * 30)
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://127.0.0.1:8001/tools")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Available Tools:")
                for tool in result.get("tools", []):
                    print(f"   - {tool['name']}: {tool['description']}")
                return True
            else:
                print(f"❌ Failed to get tools: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Tool listing failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 MISO Ultimate MCP Tool Test Suite")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().isoformat()}")
    print()
    
    # Test 1: Tool listing
    tools_ok = await test_tool_listing()
    
    # Test 2: Benchmark tool execution
    benchmark_ok = await test_mcp_benchmark_tool()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"   Tool Listing: {'✅ PASS' if tools_ok else '❌ FAIL'}")
    print(f"   Benchmark Tool: {'✅ PASS' if benchmark_ok else '❌ FAIL'}")
    
    if tools_ok and benchmark_ok:
        print("\n🎉 ALL TESTS PASSED - MCP Tool is working with real API!")
    else:
        print("\n⚠️ Some tests failed - check server status")
    
    print(f"⏰ Completed at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    asyncio.run(main())
