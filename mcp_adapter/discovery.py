#!/usr/bin/env python3
"""
🔍 MCP Tool Auto-Discovery System
=================================

Automatic tool discovery and registration system that scans for MCP-compatible
tools and registers them dynamically with full introspection capabilities.

Author: MISO Ultimate Team
Date: 30.07.2025
"""

import inspect
import importlib
import pkgutil
from typing import Dict, Any, List, Type, Optional
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel


class ToolMetadata(BaseModel):
    """Metadata for discovered MCP tools"""
    name: str
    description: str
    module_path: str
    class_name: str
    parameters_schema: Dict[str, Any]
    methods: List[str]
    discovered_at: str
    version: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = []


class MCPToolDiscovery:
    """
    Automatic discovery system for MCP-compatible tools.
    
    Scans specified directories and modules for classes that implement
    the MCP tool interface and automatically registers them.
    """
    
    def __init__(self):
        self.discovered_tools: Dict[str, ToolMetadata] = {}
        self.tool_instances: Dict[str, Any] = {}
        
    def is_mcp_tool(self, cls: Type) -> bool:
        """
        Check if a class is MCP-compatible.
        
        Args:
            cls: Class to check
            
        Returns:
            True if class implements MCP tool interface
        """
        required_attributes = ['name', 'description', 'parameters']
        required_methods = ['run']
        
        # Check for required class attributes
        for attr in required_attributes:
            if not hasattr(cls, attr):
                return False
        
        # Check for required methods
        for method in required_methods:
            if not hasattr(cls, method) or not callable(getattr(cls, method)):
                return False
        
        # Additional validation
        try:
            # Check if name is string
            if not isinstance(cls.name, str):
                return False
            
            # Check if description is string
            if not isinstance(cls.description, str):
                return False
            
            # Check if parameters has schema method (Pydantic model)
            if hasattr(cls.parameters, 'schema'):
                return True
            elif hasattr(cls.parameters, '__annotations__'):
                return True
                
        except Exception:
            return False
        
        return False
    
    def extract_tool_metadata(self, cls: Type, module_path: str) -> ToolMetadata:
        """
        Extract comprehensive metadata from MCP tool class.
        
        Args:
            cls: Tool class
            module_path: Module path where tool was found
            
        Returns:
            ToolMetadata with full introspection data
        """
        # Get parameters schema
        parameters_schema = {}
        if hasattr(cls.parameters, 'schema'):
            parameters_schema = cls.parameters.schema()
        elif hasattr(cls.parameters, '__annotations__'):
            parameters_schema = {
                "type": "object",
                "properties": {
                    name: {"type": str(annotation).replace("<class '", "").replace("'>", "")}
                    for name, annotation in cls.parameters.__annotations__.items()
                }
            }
        
        # Get all methods
        methods = [
            method for method in dir(cls)
            if not method.startswith('_') and callable(getattr(cls, method))
        ]
        
        # Extract version and author from docstring if available
        version = None
        author = None
        tags = []
        
        if cls.__doc__:
            doc_lines = cls.__doc__.split('\n')
            for line in doc_lines:
                line = line.strip()
                if line.startswith('Version:'):
                    version = line.replace('Version:', '').strip()
                elif line.startswith('Author:'):
                    author = line.replace('Author:', '').strip()
                elif line.startswith('Tags:'):
                    tags = [tag.strip() for tag in line.replace('Tags:', '').split(',')]
        
        return ToolMetadata(
            name=cls.name,
            description=cls.description,
            module_path=module_path,
            class_name=cls.__name__,
            parameters_schema=parameters_schema,
            methods=methods,
            discovered_at=datetime.now().isoformat(),
            version=version,
            author=author,
            tags=tags
        )
    
    def scan_module(self, module_name: str) -> List[ToolMetadata]:
        """
        Scan a specific module for MCP tools.
        
        Args:
            module_name: Name of module to scan
            
        Returns:
            List of discovered tool metadata
        """
        discovered = []
        
        try:
            module = importlib.import_module(module_name)
            
            # Scan all classes in module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self.is_mcp_tool(obj):
                    metadata = self.extract_tool_metadata(obj, module_name)
                    discovered.append(metadata)
                    
                    # Store tool class for instantiation
                    self.discovered_tools[metadata.name] = metadata
                    
                    print(f"🔍 Discovered MCP tool: {metadata.name} in {module_name}")
                    
        except Exception as e:
            print(f"⚠️ Error scanning module {module_name}: {e}")
        
        return discovered
    
    def scan_package(self, package_name: str) -> List[ToolMetadata]:
        """
        Recursively scan a package for MCP tools.
        
        Args:
            package_name: Name of package to scan
            
        Returns:
            List of all discovered tool metadata
        """
        discovered = []
        
        try:
            package = importlib.import_module(package_name)
            
            # Scan main package module
            discovered.extend(self.scan_module(package_name))
            
            # Recursively scan submodules
            if hasattr(package, '__path__'):
                for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
                    full_module_name = f"{package_name}.{modname}"
                    discovered.extend(self.scan_module(full_module_name))
                    
                    # If it's a subpackage, scan recursively
                    if ispkg:
                        discovered.extend(self.scan_package(full_module_name))
                        
        except Exception as e:
            print(f"⚠️ Error scanning package {package_name}: {e}")
        
        return discovered
    
    def auto_register_tools(self, registry) -> int:
        """
        Automatically register all discovered tools with the registry.
        
        Args:
            registry: MCPToolRegistry instance
            
        Returns:
            Number of tools registered
        """
        registered_count = 0
        
        for tool_name, metadata in self.discovered_tools.items():
            try:
                # Import the module and get the class
                module = importlib.import_module(metadata.module_path)
                tool_class = getattr(module, metadata.class_name)
                
                # Instantiate the tool
                tool_instance = tool_class()
                
                # Register with registry
                registry.register(tool_instance)
                self.tool_instances[tool_name] = tool_instance
                
                registered_count += 1
                print(f"✅ Auto-registered tool: {tool_name}")
                
            except Exception as e:
                print(f"❌ Failed to register tool {tool_name}: {e}")
        
        return registered_count
    
    def get_tool_introspection(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed introspection data for a specific tool.
        
        Args:
            tool_name: Name of tool to introspect
            
        Returns:
            Comprehensive introspection data
        """
        if tool_name not in self.discovered_tools:
            return None
        
        metadata = self.discovered_tools[tool_name]
        tool_instance = self.tool_instances.get(tool_name)
        
        introspection = {
            "metadata": metadata.dict(),
            "runtime_info": {},
            "capabilities": []
        }
        
        if tool_instance:
            # Runtime information
            introspection["runtime_info"] = {
                "instance_type": type(tool_instance).__name__,
                "instance_id": id(tool_instance),
                "memory_usage": f"{id(tool_instance)} bytes",
                "is_async": inspect.iscoroutinefunction(tool_instance.run),
                "has_cleanup": hasattr(tool_instance, 'close'),
            }
            
            # Capabilities analysis
            capabilities = []
            if hasattr(tool_instance, 'backend_url'):
                capabilities.append("api_integration")
            if hasattr(tool_instance, 'client'):
                capabilities.append("http_client")
            if inspect.iscoroutinefunction(tool_instance.run):
                capabilities.append("async_execution")
            if hasattr(tool_instance, 'parameters'):
                capabilities.append("parameter_validation")
            
            introspection["capabilities"] = capabilities
        
        return introspection
    
    def get_all_tools_summary(self) -> Dict[str, Any]:
        """
        Get summary of all discovered tools.
        
        Returns:
            Summary with statistics and tool list
        """
        return {
            "total_discovered": len(self.discovered_tools),
            "total_registered": len(self.tool_instances),
            "discovery_timestamp": datetime.now().isoformat(),
            "tools": {
                name: {
                    "name": metadata.name,
                    "description": metadata.description,
                    "module": metadata.module_path,
                    "version": metadata.version,
                    "author": metadata.author,
                    "tags": metadata.tags,
                    "registered": name in self.tool_instances
                }
                for name, metadata in self.discovered_tools.items()
            },
            "statistics": {
                "by_module": self._get_module_statistics(),
                "by_author": self._get_author_statistics(),
                "by_tags": self._get_tag_statistics()
            }
        }
    
    def _get_module_statistics(self) -> Dict[str, int]:
        """Get statistics by module"""
        stats = {}
        for metadata in self.discovered_tools.values():
            module = metadata.module_path
            stats[module] = stats.get(module, 0) + 1
        return stats
    
    def _get_author_statistics(self) -> Dict[str, int]:
        """Get statistics by author"""
        stats = {}
        for metadata in self.discovered_tools.values():
            author = metadata.author or "Unknown"
            stats[author] = stats.get(author, 0) + 1
        return stats
    
    def _get_tag_statistics(self) -> Dict[str, int]:
        """Get statistics by tags"""
        stats = {}
        for metadata in self.discovered_tools.values():
            for tag in metadata.tags:
                stats[tag] = stats.get(tag, 0) + 1
        return stats
