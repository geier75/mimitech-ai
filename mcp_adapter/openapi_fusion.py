#!/usr/bin/env python3
"""
🔗 OpenAPI Fusion System for MCP Tools
=====================================

Automatic OpenAPI schema integration that generates prompt templates
and enables seamless API-to-MCP tool conversion.

Author: MISO Ultimate Team
Date: 30.07.2025
"""

import json
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class OpenAPIEndpoint(BaseModel):
    """OpenAPI endpoint metadata"""
    path: str
    method: str
    summary: Optional[str] = None
    description: Optional[str] = None
    parameters: Dict[str, Any] = {}
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Any] = {}
    tags: List[str] = []


class PromptTemplate(BaseModel):
    """Generated prompt template for MCP tool"""
    tool_name: str
    description: str
    usage_example: str
    parameter_hints: List[str]
    expected_output: str
    curl_example: str
    mcp_payload: Dict[str, Any]


class OpenAPIFusionEngine:
    """
    OpenAPI Fusion Engine that converts FastAPI schemas to MCP tools
    and generates intelligent prompt templates.
    """
    
    def __init__(self, backend_url: str = "http://127.0.0.1:8000"):
        """
        Initialize OpenAPI Fusion Engine.
        
        Args:
            backend_url: Base URL of FastAPI backend
        """
        self.backend_url = backend_url.rstrip('/')
        self.openapi_schema: Optional[Dict[str, Any]] = None
        self.endpoints: List[OpenAPIEndpoint] = []
        self.prompt_templates: Dict[str, PromptTemplate] = {}
    
    def fetch_openapi_schema(self) -> bool:
        """
        Fetch OpenAPI schema from FastAPI backend.
        
        Returns:
            True if schema was successfully fetched
        """
        try:
            response = requests.get(f"{self.backend_url}/openapi.json", timeout=10)
            response.raise_for_status()
            
            self.openapi_schema = response.json()
            print(f"✅ OpenAPI schema fetched: {len(self.openapi_schema.get('paths', {}))} endpoints")
            return True
            
        except Exception as e:
            print(f"❌ Failed to fetch OpenAPI schema: {e}")
            return False
    
    def parse_endpoints(self) -> List[OpenAPIEndpoint]:
        """
        Parse OpenAPI schema and extract endpoint information.
        
        Returns:
            List of parsed endpoints
        """
        if not self.openapi_schema:
            return []
        
        endpoints = []
        paths = self.openapi_schema.get('paths', {})
        
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    endpoint = OpenAPIEndpoint(
                        path=path,
                        method=method.upper(),
                        summary=details.get('summary', ''),
                        description=details.get('description', ''),
                        parameters=self._extract_parameters(details),
                        request_body=details.get('requestBody', {}),
                        responses=details.get('responses', {}),
                        tags=details.get('tags', [])
                    )
                    endpoints.append(endpoint)
        
        self.endpoints = endpoints
        print(f"🔍 Parsed {len(endpoints)} API endpoints")
        return endpoints
    
    def _extract_parameters(self, endpoint_details: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from endpoint details"""
        parameters = {}
        
        # Path parameters
        for param in endpoint_details.get('parameters', []):
            param_name = param.get('name', '')
            param_info = {
                'type': param.get('schema', {}).get('type', 'string'),
                'description': param.get('description', ''),
                'required': param.get('required', False),
                'location': param.get('in', 'query')
            }
            parameters[param_name] = param_info
        
        # Request body parameters
        request_body = endpoint_details.get('requestBody', {})
        if request_body:
            content = request_body.get('content', {})
            json_content = content.get('application/json', {})
            schema = json_content.get('schema', {})
            
            if 'properties' in schema:
                for prop_name, prop_details in schema['properties'].items():
                    parameters[prop_name] = {
                        'type': prop_details.get('type', 'string'),
                        'description': prop_details.get('description', ''),
                        'required': prop_name in schema.get('required', []),
                        'location': 'body'
                    }
        
        return parameters
    
    def generate_mcp_tool_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate MCP tool mappings from OpenAPI endpoints.
        
        Returns:
            Dictionary mapping tool names to their configurations
        """
        mappings = {}
        
        for endpoint in self.endpoints:
            # Skip non-relevant endpoints
            if endpoint.path.startswith('/docs') or endpoint.path.startswith('/openapi'):
                continue
            
            # Generate tool name from path and method
            tool_name = self._generate_tool_name(endpoint.path, endpoint.method)
            
            # Create MCP tool configuration
            tool_config = {
                'name': tool_name,
                'description': endpoint.description or endpoint.summary or f"{endpoint.method} {endpoint.path}",
                'endpoint': {
                    'path': endpoint.path,
                    'method': endpoint.method
                },
                'parameters': endpoint.parameters,
                'tags': endpoint.tags,
                'generated_at': datetime.now().isoformat()
            }
            
            mappings[tool_name] = tool_config
        
        print(f"🔧 Generated {len(mappings)} MCP tool mappings")
        return mappings
    
    def _generate_tool_name(self, path: str, method: str) -> str:
        """Generate tool name from API path and method"""
        # Clean path and convert to tool name
        clean_path = path.strip('/').replace('/', '_').replace('{', '').replace('}', '')
        method_lower = method.lower()
        
        # Special cases for common patterns
        if 'api/benchmarks' in path:
            if 'start' in path:
                return f"benchmark_{clean_path.split('_')[-2]}_start"
            elif 'results' in path:
                return f"benchmark_get_results"
        
        if 'api/training' in path:
            return f"training_{clean_path.split('_')[-1]}"
        
        if 'api/system' in path:
            return f"system_{clean_path.split('_')[-1]}"
        
        # Default naming
        return f"{method_lower}_{clean_path}".replace('__', '_')
    
    def generate_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """
        Generate intelligent prompt templates for MCP tools.
        
        Returns:
            Dictionary of prompt templates by tool name
        """
        templates = {}
        
        for endpoint in self.endpoints:
            tool_name = self._generate_tool_name(endpoint.path, endpoint.method)
            
            # Generate usage example
            usage_example = self._generate_usage_example(endpoint)
            
            # Generate parameter hints
            parameter_hints = self._generate_parameter_hints(endpoint)
            
            # Generate expected output description
            expected_output = self._generate_expected_output(endpoint)
            
            # Generate curl example
            curl_example = self._generate_curl_example(endpoint)
            
            # Generate MCP payload example
            mcp_payload = self._generate_mcp_payload(endpoint, tool_name)
            
            template = PromptTemplate(
                tool_name=tool_name,
                description=endpoint.description or endpoint.summary or f"Execute {endpoint.method} {endpoint.path}",
                usage_example=usage_example,
                parameter_hints=parameter_hints,
                expected_output=expected_output,
                curl_example=curl_example,
                mcp_payload=mcp_payload
            )
            
            templates[tool_name] = template
        
        self.prompt_templates = templates
        print(f"📝 Generated {len(templates)} prompt templates")
        return templates
    
    def _generate_usage_example(self, endpoint: OpenAPIEndpoint) -> str:
        """Generate usage example for endpoint"""
        if 'benchmark' in endpoint.path:
            return f"Use this tool to start {endpoint.path.split('/')[-2]} benchmarks with custom parameters like preset and iterations."
        elif 'training' in endpoint.path:
            return "Use this tool to execute a training step with specified learning rate, batch size, and optimizer."
        elif 'system' in endpoint.path:
            return "Use this tool to get real-time system metrics including CPU, memory, and GPU usage."
        else:
            return f"Use this tool to {endpoint.method.lower()} data from {endpoint.path}"
    
    def _generate_parameter_hints(self, endpoint: OpenAPIEndpoint) -> List[str]:
        """Generate parameter hints for endpoint"""
        hints = []
        
        for param_name, param_info in endpoint.parameters.items():
            hint = f"{param_name} ({param_info['type']})"
            if param_info['required']:
                hint += " - REQUIRED"
            if param_info['description']:
                hint += f": {param_info['description']}"
            hints.append(hint)
        
        return hints
    
    def _generate_expected_output(self, endpoint: OpenAPIEndpoint) -> str:
        """Generate expected output description"""
        if 'start' in endpoint.path:
            return "Returns test_id, status, and message confirming benchmark started"
        elif 'results' in endpoint.path:
            return "Returns detailed benchmark results with performance metrics"
        elif 'status' in endpoint.path:
            return "Returns current system status with CPU, memory, and GPU metrics"
        else:
            return "Returns JSON response with operation result"
    
    def _generate_curl_example(self, endpoint: OpenAPIEndpoint) -> str:
        """Generate curl example for endpoint"""
        base_url = self.backend_url
        
        if endpoint.method == 'GET':
            return f"curl -X GET {base_url}{endpoint.path}"
        elif endpoint.method == 'POST':
            if endpoint.parameters:
                sample_data = {name: "example_value" for name in endpoint.parameters.keys()}
                return f"curl -X POST {base_url}{endpoint.path} -H 'Content-Type: application/json' -d '{json.dumps(sample_data)}'"
            else:
                return f"curl -X POST {base_url}{endpoint.path}"
        else:
            return f"curl -X {endpoint.method} {base_url}{endpoint.path}"
    
    def _generate_mcp_payload(self, endpoint: OpenAPIEndpoint, tool_name: str) -> Dict[str, Any]:
        """Generate MCP payload example"""
        sample_params = {}
        
        for param_name, param_info in endpoint.parameters.items():
            if param_info['type'] == 'string':
                sample_params[param_name] = "example_string"
            elif param_info['type'] == 'integer':
                sample_params[param_name] = 100
            elif param_info['type'] == 'boolean':
                sample_params[param_name] = True
            elif param_info['type'] == 'array':
                sample_params[param_name] = ["item1", "item2"]
            else:
                sample_params[param_name] = "example_value"
        
        return {
            "tool": tool_name,
            "params": sample_params
        }
    
    def export_templates(self, output_file: str = "mcp_prompt_templates.json") -> bool:
        """
        Export generated prompt templates to JSON file.
        
        Args:
            output_file: Output file path
            
        Returns:
            True if export was successful
        """
        try:
            export_data = {
                "generated_at": datetime.now().isoformat(),
                "backend_url": self.backend_url,
                "total_templates": len(self.prompt_templates),
                "templates": {
                    name: template.dict() 
                    for name, template in self.prompt_templates.items()
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"📄 Exported {len(self.prompt_templates)} templates to {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to export templates: {e}")
            return False
    
    def get_fusion_summary(self) -> Dict[str, Any]:
        """Get comprehensive fusion summary"""
        return {
            "backend_url": self.backend_url,
            "schema_loaded": self.openapi_schema is not None,
            "endpoints_parsed": len(self.endpoints),
            "templates_generated": len(self.prompt_templates),
            "fusion_timestamp": datetime.now().isoformat(),
            "endpoint_summary": [
                {
                    "path": ep.path,
                    "method": ep.method,
                    "tool_name": self._generate_tool_name(ep.path, ep.method),
                    "parameters": len(ep.parameters)
                }
                for ep in self.endpoints
            ]
        }
