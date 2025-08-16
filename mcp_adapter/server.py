#!/usr/bin/env python3
"""
🔧 MCP Server for MISO Ultimate
==============================

FastAPI-based MCP server that handles tool execution requests
and forwards them to real MISO Ultimate backend APIs.

Author: MISO Ultimate Team
Date: 30.07.2025
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator

from .tools.benchmarks import BenchmarkMatrixTool, BenchmarkQuantumTool, BenchmarkAllTool
from .discovery import MCPToolDiscovery
from .openapi_fusion import OpenAPIFusionEngine
from .websocket_bridge import WebSocketStreamingBridge


class MCPRequest(BaseModel):
    """MCP tool execution request"""
    tool: str = Field(description="Tool name to execute")
    params: Dict[str, Any] = Field(default={}, description="Tool parameters")
    
    @validator('params')
    def validate_params_size(cls, v):
        """Validate parameter payload size"""
        import json
        payload_size = len(json.dumps(v).encode('utf-8'))
        if payload_size > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError(f"Parameter payload too large: {payload_size} bytes (max: 10MB)")
        return v
    
    @validator('tool')
    def validate_tool_name(cls, v):
        """Validate tool name format"""
        if not v or not isinstance(v, str):
            raise ValueError("Tool name must be a non-empty string")
        if len(v) > 100:
            raise ValueError("Tool name too long (max: 100 characters)")
        return v.strip()


class MCPErrorResponse(BaseModel):
    """Standardized MCP error response"""
    error: str = Field(description="Error message")
    detail: str = Field(description="Detailed error description")
    error_code: str = Field(description="Error code for programmatic handling")
    timestamp: str = Field(description="ISO timestamp of error")
    request_id: str = Field(default="", description="Request ID for tracking")

class MCPResponse(BaseModel):
    """MCP tool execution response"""
    tool: str
    status: str
    timestamp: str
    result: Dict[str, Any]


class MCPToolRegistry:
    """Registry for MCP-compatible tools"""
    
    def __init__(self):
        self.tools: Dict[str, Any] = {}
    
    def register(self, tool_instance):
        """Register a tool instance"""
        self.tools[tool_instance.name] = tool_instance
    
    def get_tool(self, name: str):
        """Get tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with their schemas"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.schema() if hasattr(tool.parameters, 'schema') else {}
            }
            for tool in self.tools.values()
        ]


# Rate limiting storage
rate_limit_storage = defaultdict(lambda: deque())
MAX_REQUESTS_PER_MINUTE = 60
MAX_CONCURRENT_REQUESTS = 20
current_requests = 0

# Initialize FastAPI app
app = FastAPI(
    title="MISO Ultimate MCP Server",
    description="Model Context Protocol server for MISO Ultimate tools",
    version="1.0.0"
)

# Security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

# CORS middleware - Safari-compatible configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8082",  # Frontend dashboard
        "http://127.0.0.1:8081",  # Alternative frontend port
        "http://127.0.0.1:8080",  # Alternative frontend port
        "http://localhost:8082",  # Localhost variant
        "http://localhost:8081",  # Localhost variant
        "http://localhost:8080",  # Localhost variant
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting and concurrent request control"""
    global current_requests
    
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old requests (older than 1 minute)
    rate_limit_storage[client_ip] = deque([
        req_time for req_time in rate_limit_storage[client_ip]
        if current_time - req_time < 60
    ])
    
    # Check rate limit
    if len(rate_limit_storage[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
        return HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "detail": f"Maximum {MAX_REQUESTS_PER_MINUTE} requests per minute allowed",
                "error_code": "RATE_LIMIT_EXCEEDED",
                "timestamp": datetime.now().isoformat(),
                "retry_after": 60
            }
        ).json()
    
    # Check concurrent requests
    if current_requests >= MAX_CONCURRENT_REQUESTS:
        return HTTPException(
            status_code=503,
            detail={
                "error": "Server overloaded",
                "detail": f"Maximum {MAX_CONCURRENT_REQUESTS} concurrent requests allowed",
                "error_code": "SERVER_OVERLOAD",
                "timestamp": datetime.now().isoformat(),
                "retry_after": 5
            }
        ).json()
    
    # Record request and increment counter
    rate_limit_storage[client_ip].append(current_time)
    current_requests += 1
    
    try:
        response = await call_next(request)
        return response
    finally:
        current_requests -= 1

# Initialize tool registry, discovery system, OpenAPI fusion, and WebSocket bridge
registry = MCPToolRegistry()
discovery = MCPToolDiscovery()
fusion_engine = OpenAPIFusionEngine()
streaming_bridge = WebSocketStreamingBridge()

# Auto-discover tools from the tools package
print("🔍 Starting auto-discovery of MCP tools...")
discovered_tools = discovery.scan_package('mcp_adapter.tools')
print(f"🔍 Discovered {len(discovered_tools)} MCP tools")

# Auto-register discovered tools
registered_count = discovery.auto_register_tools(registry)
print(f"✅ Auto-registered {registered_count} tools")

# Manual fallback registration (if auto-discovery fails)
if registered_count == 0:
    print("⚠️ Auto-discovery failed, falling back to manual registration")
    benchmark_matrix_tool = BenchmarkMatrixTool()
    benchmark_quantum_tool = BenchmarkQuantumTool()
    benchmark_all_tool = BenchmarkAllTool()
    
    registry.register(benchmark_matrix_tool)
    registry.register(benchmark_quantum_tool)
    registry.register(benchmark_all_tool)
    print("✅ Manual registration completed")


@app.get("/")
async def root():
    """Root endpoint with server info"""
    return {
        "server": "MISO Ultimate MCP Server",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "tools_available": len(registry.tools)
    }


@app.get("/tools")
async def list_tools():
    """List all available MCP tools with basic info"""
    return {
        "tools": registry.list_tools(),
        "count": len(registry.tools),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/tools/discovery")
async def get_discovery_summary():
    """Get comprehensive tool discovery summary"""
    return discovery.get_all_tools_summary()


@app.get("/tools/{tool_name}/introspect")
async def introspect_tool(tool_name: str):
    """Get detailed introspection data for a specific tool"""
    introspection = discovery.get_tool_introspection(tool_name)
    if not introspection:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    return introspection


@app.get("/tools/{tool_name}/schema")
async def get_tool_schema(tool_name: str):
    """Get parameter schema for a specific tool"""
    tool = registry.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    schema = {}
    if hasattr(tool, 'parameters') and hasattr(tool.parameters, 'schema'):
        schema = tool.parameters.schema()
    
    return {
        "tool": tool_name,
        "schema": schema,
        "timestamp": datetime.now().isoformat()
    }


def create_error_response(error_msg: str, detail: str, error_code: str, status_code: int = 400) -> HTTPException:
    """Create standardized error response"""
    return HTTPException(
        status_code=status_code,
        detail={
            "error": error_msg,
            "detail": detail,
            "error_code": error_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest) -> MCPResponse:
    """
    Handle MCP tool execution request
    
    This is the main endpoint that processes MCP requests and forwards
    them to the appropriate tool, which then calls the real FastAPI backend.
    
    Edge cases handled:
    - Invalid tool names (404)
    - Empty/invalid parameters (422)
    - Payload size limits (413)
    - Rate limiting (429)
    - Server overload (503)
    """
    try:
        # Validate tool exists
        tool = registry.get_tool(request.tool)
        if not tool:
            raise create_error_response(
                error_msg="Tool not found",
                detail=f"Tool '{request.tool}' not found. Available tools: {list(registry.tools.keys())}",
                error_code="TOOL_NOT_FOUND",
                status_code=404
            )
        
        # Validate parameters if tool has parameter schema
        if hasattr(tool, 'parameters') and request.params:
            try:
                # Validate using Pydantic model if available
                if hasattr(tool.parameters, '__call__'):
                    validated_params = tool.parameters(**request.params)
                    request.params = validated_params.dict()
            except Exception as validation_error:
                raise create_error_response(
                    error_msg="Invalid parameters",
                    detail=f"Parameter validation failed: {str(validation_error)}",
                    error_code="INVALID_PARAMETERS",
                    status_code=422
                )
        
        # Check for empty matrix_sizes (specific edge case)
        if 'matrix_sizes' in request.params and not request.params['matrix_sizes']:
            raise create_error_response(
                error_msg="Invalid matrix_sizes",
                detail="matrix_sizes cannot be empty. Provide at least one matrix size.",
                error_code="EMPTY_MATRIX_SIZES",
                status_code=422
            )
        
        # Execute the tool with real API call
        result = await tool.run(request.params)
        
        # Return MCP-formatted response
        return MCPResponse(
            tool=request.tool,
            status="success",
            timestamp=datetime.now().isoformat(),
            result=result
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (already formatted)
        raise
    except ValueError as e:
        # Handle validation errors
        raise create_error_response(
            error_msg="Validation error",
            detail=str(e),
            error_code="VALIDATION_ERROR",
            status_code=422
        )
    except Exception as e:
        # Handle unexpected errors
        raise create_error_response(
            error_msg="Internal server error",
            detail=f"Unexpected error: {str(e)}",
            error_code="INTERNAL_ERROR",
            status_code=500
        )


@app.get("/health")
async def health_check():
    """Health check endpoint with discovery stats"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tools_registered": len(registry.tools),
        "tools_discovered": len(discovery.discovered_tools),
        "auto_discovery_enabled": True,
        "discovery_modules": list(discovery._get_module_statistics().keys())
    }


@app.post("/tools/rediscover")
async def rediscover_tools():
    """Trigger re-discovery of tools"""
    try:
        # Clear existing discoveries
        discovery.discovered_tools.clear()
        discovery.tool_instances.clear()
        
        # Re-discover tools
        discovered_tools = discovery.scan_package('mcp_adapter.tools')
        registered_count = discovery.auto_register_tools(registry)
        
        return {
            "status": "success",
            "discovered": len(discovered_tools),
            "registered": registered_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/openapi/fusion")
async def get_openapi_fusion_summary():
    """Get OpenAPI fusion summary and status"""
    return fusion_engine.get_fusion_summary()


@app.post("/openapi/sync")
async def sync_openapi_schema():
    """Sync OpenAPI schema from backend and generate templates"""
    try:
        # Fetch OpenAPI schema
        schema_success = fusion_engine.fetch_openapi_schema()
        if not schema_success:
            return {
                "status": "error",
                "error": "Failed to fetch OpenAPI schema",
                "timestamp": datetime.now().isoformat()
            }
        
        # Parse endpoints
        endpoints = fusion_engine.parse_endpoints()
        
        # Generate prompt templates
        templates = fusion_engine.generate_prompt_templates()
        
        # Export templates
        export_success = fusion_engine.export_templates()
        
        return {
            "status": "success",
            "schema_fetched": schema_success,
            "endpoints_parsed": len(endpoints),
            "templates_generated": len(templates),
            "templates_exported": export_success,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/openapi/templates")
async def get_prompt_templates():
    """Get all generated prompt templates"""
    return {
        "templates": {name: template.dict() for name, template in fusion_engine.prompt_templates.items()},
        "count": len(fusion_engine.prompt_templates),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/openapi/templates/{tool_name}")
async def get_tool_prompt_template(tool_name: str):
    """Get prompt template for specific tool"""
    template = fusion_engine.prompt_templates.get(tool_name)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template for tool '{tool_name}' not found")
    return template.model_dump()


@app.get("/streaming/status")
async def get_streaming_status():
    """Get WebSocket streaming bridge status"""
    return streaming_bridge.get_streaming_summary()


@app.get("/streaming/sessions")
async def get_active_streaming_sessions():
    """Get all active streaming sessions"""
    return {
        "active_sessions": [
            {
                "session_id": session.session_id,
                "tool_name": session.tool_name,
                "status": session.status,
                "created_at": session.created_at,
                "last_update": session.last_update,
                "message_count": len(session.messages)
            }
            for session in streaming_bridge.active_sessions.values()
        ],
        "total_sessions": len(streaming_bridge.active_sessions),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/streaming/cleanup/{session_id}")
async def cleanup_streaming_session(session_id: str):
    """Clean up a specific streaming session"""
    if session_id not in streaming_bridge.active_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    await streaming_bridge.cleanup_session(session_id)
    return {
        "status": "success",
        "message": f"Session {session_id} cleaned up",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/mcp/streaming")
async def handle_streaming_mcp_request(request: MCPRequest) -> Dict[str, Any]:
    """Handle MCP tool execution with streaming support"""
    try:
        # Get the requested tool
        tool = registry.get_tool(request.tool)
        if not tool:
            raise HTTPException(
                status_code=404, 
                detail=f"Tool '{request.tool}' not found. Available tools: {list(registry.tools.keys())}"
            )
        
        # Create a mock WebSocket session for streaming
        # In a real implementation, this would be connected to an actual WebSocket
        session = await streaming_bridge.create_streaming_session(request.tool, None)
        
        # Execute tool with streaming simulation
        await streaming_bridge.execute_streaming_tool(session, tool, request.params)
        
        # Return session info
        return {
            "tool": request.tool,
            "status": "streaming_started",
            "session_id": session.session_id,
            "timestamp": datetime.now().isoformat(),
            "streaming_url": f"ws://127.0.0.1:8002/stream/{session.session_id}",
            "message": "Tool execution started with streaming support"
        }
        
    except Exception as e:
        return {
            "tool": request.tool,
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "error_type": type(e).__name__
        }


# Mock Benchmark API Endpoints for MCP Tools
@app.post("/api/benchmarks/matrix/start")
async def start_matrix_benchmark(request: Dict[str, Any]):
    """Mock matrix benchmark endpoint"""
    import random
    import time
    
    config = request.get("config", {})
    
    # Simulate benchmark execution
    await asyncio.sleep(0.5)  # Simulate processing time
    
    # Generate realistic mock results
    matrix_sizes = config.get("matrix_sizes", [64, 128, 256])
    iterations = config.get("iterations", 100)
    
    results = {
        "benchmark_id": f"matrix_{int(time.time())}",
        "status": "completed",
        "config": config,
        "results": {
            "total_time": round(random.uniform(1.5, 5.0), 3),
            "avg_time_per_iteration": round(random.uniform(0.01, 0.05), 4),
            "throughput_ops_per_sec": round(random.uniform(800, 2000), 1),
            "memory_usage_mb": round(random.uniform(50, 200), 1),
            "efficiency_score": round(random.uniform(0.75, 0.95), 3),
            "matrix_results": [
                {
                    "size": size,
                    "time_ms": round(random.uniform(10, 100), 2),
                    "flops": round(random.uniform(1e9, 1e12), 0)
                } for size in matrix_sizes
            ]
        },
        "timestamp": datetime.now().isoformat(),
        "backend": "mock"
    }
    
    return results

@app.post("/api/benchmarks/quantum/start")
async def start_quantum_benchmark(request: Dict[str, Any]):
    """Mock quantum benchmark endpoint"""
    import random
    import time
    
    config = request.get("config", {})
    
    # Simulate benchmark execution
    await asyncio.sleep(0.3)
    
    qubits = config.get("qubits", 8)
    depth = config.get("depth", 10)
    shots = config.get("shots", 1000)
    
    results = {
        "benchmark_id": f"quantum_{int(time.time())}",
        "status": "completed",
        "config": config,
        "results": {
            "total_time": round(random.uniform(0.5, 3.0), 3),
            "circuit_depth": depth,
            "qubit_count": qubits,
            "shot_count": shots,
            "fidelity": round(random.uniform(0.85, 0.98), 4),
            "gate_errors": round(random.uniform(0.001, 0.01), 5),
            "execution_time_ms": round(random.uniform(100, 500), 2),
            "quantum_volume": 2 ** min(qubits, depth)
        },
        "timestamp": datetime.now().isoformat(),
        "backend": "mock"
    }
    
    return results

@app.post("/api/benchmarks/all/start")
async def start_comprehensive_benchmark(request: Dict[str, Any]):
    """Mock comprehensive benchmark endpoint"""
    import random
    import time
    
    config = request.get("config", {})
    
    # Simulate longer benchmark execution
    await asyncio.sleep(1.0)
    
    preset = config.get("preset", "standard")
    parallel = config.get("parallel", True)
    
    results = {
        "benchmark_id": f"comprehensive_{int(time.time())}",
        "status": "completed",
        "config": config,
        "results": {
            "total_time": round(random.uniform(5.0, 15.0), 3),
            "preset": preset,
            "parallel_execution": parallel,
            "overall_score": round(random.uniform(75, 95), 1),
            "component_scores": {
                "matrix_performance": round(random.uniform(80, 95), 1),
                "quantum_simulation": round(random.uniform(70, 90), 1),
                "memory_efficiency": round(random.uniform(85, 98), 1),
                "cpu_utilization": round(random.uniform(60, 85), 1),
                "stability": round(random.uniform(90, 99), 1)
            },
            "system_info": {
                "cpu_cores": 8,
                "memory_gb": 16,
                "platform": "darwin",
                "python_version": "3.13.3"
            }
        },
        "timestamp": datetime.now().isoformat(),
        "backend": "mock"
    }
    
    return results

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting MISO Ultimate MCP Server...")
    print("🔧 Tools available:", list(registry.tools.keys()))
    print("🔍 Tools discovered:", len(discovery.discovered_tools))
    print("📡 MCP endpoint: http://127.0.0.1:8001/mcp")
    print("📋 Tools list: http://127.0.0.1:8001/tools")
    print("🔍 Discovery summary: http://127.0.0.1:8001/tools/discovery")
    print("🔬 Tool introspection: http://127.0.0.1:8001/tools/{tool_name}/introspect")
    print("🔗 OpenAPI fusion: http://127.0.0.1:8001/openapi/fusion")
    print("📝 Prompt templates: http://127.0.0.1:8001/openapi/templates")
    print("🔄 Sync OpenAPI: POST http://127.0.0.1:8001/openapi/sync")
    print("🌊 Streaming status: http://127.0.0.1:8001/streaming/status")
    print("📡 Streaming MCP: POST http://127.0.0.1:8001/mcp/streaming")
    print("🔌 WebSocket streaming: ws://127.0.0.1:8002 (planned)")
    print("🎯 Mock Benchmark APIs: http://127.0.0.1:8001/api/benchmarks/")
    
    uvicorn.run(
        "mcp_adapter.server:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info"
    )
