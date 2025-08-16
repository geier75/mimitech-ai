#!/usr/bin/env python3
"""
🔧 MCP Benchmark Tools for MISO Ultimate
========================================

Production-ready MCP tools that call real FastAPI benchmark endpoints.
No dummy data - direct integration with existing backend APIs.

Author: MISO Ultimate Team
Date: 30.07.2025
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

import httpx
from pydantic import BaseModel, Field, validator


class BenchmarkMatrixParams(BaseModel):
    """Parameters for matrix benchmark tool"""
    
    preset: str = Field(
        default="standard",
        description="Benchmark preset: 'quick', 'standard', 'intensive', 'extreme'"
    )
    
    iterations: int = Field(
        default=100,
        description="Number of iterations for each test",
        ge=1,
        le=10000
    )
    
    matrix_sizes: List[int] = Field(
        default=[64, 128, 256, 512],
        description="Matrix sizes to test"
    )
    
    backend_preference: str = Field(
        default="auto",
        description="Preferred backend: 'auto', 'mlx', 'torch', 'numpy'"
    )
    
    timeout_seconds: int = Field(
        default=300,
        description="Maximum execution time in seconds",
        ge=10,
        le=3600
    )


class BenchmarkMatrixTool:
    """
    Production MCP tool for MISO Ultimate matrix benchmarks.
    
    Calls real FastAPI endpoint POST /api/benchmarks/matrix/start
    and returns actual benchmark results, not dummy data.
    """
    
    name = "benchmark_matrix"
    description = (
        "Execute real matrix benchmarks on MISO Ultimate system. "
        "Tests tensor operations, hardware acceleration (MLX/PyTorch), "
        "and performance metrics. Returns actual benchmark results."
    )
    parameters = BenchmarkMatrixParams
    
    def __init__(self, backend_url: str = "http://127.0.0.1:8001"):
        """
        Initialize the benchmark tool.
        
        Args:
            backend_url: Base URL of the FastAPI backend
        """
        self.backend_url = backend_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def run(self, params: dict) -> Dict[str, Any]:
        """
        Execute matrix benchmark via real FastAPI API call.
        
        Args:
            params: Parameters matching BenchmarkMatrixParams schema
            
        Returns:
            Dict containing real benchmark results from API
        """
        try:
            # Validate parameters using Pydantic schema
            validated_params = BenchmarkMatrixParams(**params)
            
            # Make real API call to FastAPI backend
            response = await self.client.post(
                f"{self.backend_url}/api/benchmarks/matrix/start",
                json={
                    "test_type": "matrix",
                    "config": validated_params.dict()
                },
                headers={"Content-Type": "application/json"}
            )
            
            # Handle HTTP errors
            if response.status_code != 200:
                return {
                    "tool": self.name,
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "backend_url": self.backend_url
                }
            
            # Parse real API response
            api_result = response.json()
            
            # Return real API response with MCP metadata
            return {
                "tool": self.name,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "parameters": validated_params.dict(),
                "api_response": api_result,
                "backend_url": self.backend_url,
                "test_id": api_result.get("test_id"),
                "message": api_result.get("message", "Benchmark started successfully")
            }
            
        except Exception as e:
            # Return error details for debugging
            return {
                "tool": self.name,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "error_type": type(e).__name__,
                "parameters": params,
                "backend_url": self.backend_url
            }
    
    async def close(self):
        """Clean up HTTP client"""
        await self.client.aclose()


class BenchmarkQuantumTool:
    """MCP tool for quantum benchmarks"""
    
    name = "benchmark_quantum"
    description = "Execute real quantum benchmarks on MISO Ultimate system"
    parameters = BenchmarkMatrixParams  # Reuse same params for now
    
    def __init__(self, backend_url: str = "http://127.0.0.1:8001"):
        self.backend_url = backend_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def run(self, params: dict) -> Dict[str, Any]:
        """Execute quantum benchmark via real API call"""
        try:
            validated_params = BenchmarkMatrixParams(**params)
            
            response = await self.client.post(
                f"{self.backend_url}/api/benchmarks/quantum/start",
                json={
                    "test_type": "quantum",
                    "config": validated_params.dict()
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                return {
                    "tool": self.name,
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "backend_url": self.backend_url
                }
            
            api_result = response.json()
            
            return {
                "tool": self.name,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "parameters": validated_params.dict(),
                "api_response": api_result,
                "backend_url": self.backend_url,
                "test_id": api_result.get("test_id"),
                "message": api_result.get("message", "Quantum benchmark started successfully")
            }
            
        except Exception as e:
            return {
                "tool": self.name,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "error_type": type(e).__name__,
                "parameters": params,
                "backend_url": self.backend_url
            }
    
    async def close(self):
        await self.client.aclose()


class BenchmarkAllTool:
    """MCP tool for running all benchmarks"""
    
    name = "benchmark_all"
    description = "Execute all benchmarks (matrix + quantum) on MISO Ultimate system"
    parameters = BenchmarkMatrixParams
    
    def __init__(self, backend_url: str = "http://127.0.0.1:8001"):
        self.backend_url = backend_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def run(self, params: dict) -> Dict[str, Any]:
        """Execute all benchmarks via real API call"""
        try:
            validated_params = BenchmarkMatrixParams(**params)
            
            response = await self.client.post(
                f"{self.backend_url}/api/benchmarks/all/start",
                json={
                    "test_type": "all",
                    "config": validated_params.dict()
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                return {
                    "tool": self.name,
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "backend_url": self.backend_url
                }
            
            api_result = response.json()
            
            return {
                "tool": self.name,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "parameters": validated_params.dict(),
                "api_response": api_result,
                "backend_url": self.backend_url,
                "matrix_test_id": api_result.get("matrix_test_id"),
                "quantum_test_id": api_result.get("quantum_test_id"),
                "message": api_result.get("message", "All benchmarks started successfully")
            }
            
        except Exception as e:
            return {
                "tool": self.name,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "error_type": type(e).__name__,
                "parameters": params,
                "backend_url": self.backend_url
            }
    
    async def close(self):
        await self.client.aclose()
