#!/usr/bin/env python3
"""
🔧 MISO Ultimate MCP CLI Wrapper
===============================

Command-line interface for executing MCP tools with rich output formatting.
Perfect for team development, testing, and automation.

Usage:
    python mcp-cli.py benchmark_matrix --params '{"preset":"quick"}'
    python mcp-cli.py system_metrics
    python mcp-cli.py benchmark_all --params '{"iterations":50}'

Author: MISO Ultimate Team
Date: 30.07.2025
"""

import json
import sys
import time
from typing import Dict, Any, Optional

import click
import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.text import Text

# Initialize rich console
console = Console()

# MCP Server configuration
MCP_SERVER_URL = "http://127.0.0.1:8003"
TIMEOUT_SECONDS = 60

def format_response(response_data: Dict[str, Any]) -> None:
    """Format and display MCP response with rich formatting"""
    
    if response_data.get("status") == "success":
        # Success response
        console.print(Panel(
            f"✅ Tool executed successfully",
            title=f"🔧 {response_data.get('tool', 'Unknown')}",
            border_style="green"
        ))
        
        # Display result details
        result = response_data.get("result", {})
        if result:
            if "test_id" in result:
                console.print(f"🆔 Test ID: [bold cyan]{result['test_id']}[/bold cyan]")
            
            if "status" in result:
                console.print(f"📊 Status: [bold green]{result['status']}[/bold green]")
            
            if "message" in result:
                console.print(f"💬 Message: {result['message']}")
            
            # Display full result as JSON
            if len(result) > 3:  # More than just basic fields
                console.print("\n📋 Full Result:")
                syntax = Syntax(json.dumps(result, indent=2), "json", theme="monokai")
                console.print(syntax)
    
    elif "error" in response_data:
        # Error response
        error_detail = response_data.get("detail", {})
        if isinstance(error_detail, dict):
            console.print(Panel(
                f"❌ {error_detail.get('error', 'Unknown error')}\n"
                f"📝 {error_detail.get('detail', 'No details available')}\n"
                f"🔧 Error Code: {error_detail.get('error_code', 'UNKNOWN')}",
                title="🚨 Error",
                border_style="red"
            ))
        else:
            console.print(Panel(
                f"❌ {response_data.get('error', 'Unknown error')}\n"
                f"📝 {error_detail}",
                title="🚨 Error",
                border_style="red"
            ))
    
    else:
        # Generic response
        console.print(Panel(
            json.dumps(response_data, indent=2),
            title="📤 Response",
            border_style="blue"
        ))

def get_available_tools() -> Optional[Dict[str, Any]]:
    """Fetch available tools from MCP server"""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/tools", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            console.print(f"❌ Failed to fetch tools: HTTP {response.status_code}")
            return None
    except Exception as e:
        console.print(f"❌ Error connecting to MCP server: {e}")
        return None

def display_available_tools():
    """Display available tools in a formatted table"""
    tools_data = get_available_tools()
    if not tools_data:
        return
    
    table = Table(title="🔧 Available MCP Tools")
    table.add_column("Tool Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Parameters", style="yellow")
    
    for tool in tools_data.get("tools", []):
        params = tool.get("parameters", {}).get("properties", {})
        param_names = ", ".join(params.keys()) if params else "None"
        
        table.add_row(
            tool.get("name", "Unknown"),
            tool.get("description", "No description")[:60] + "..." if len(tool.get("description", "")) > 60 else tool.get("description", ""),
            param_names
        )
    
    console.print(table)
    console.print(f"\n📊 Total tools: {tools_data.get('count', 0)}")

@click.command()
@click.argument("tool_name", required=False)
@click.option("--params", default="{}", help="JSON string with parameters")
@click.option("--list-tools", is_flag=True, help="List all available tools")
@click.option("--server", default=MCP_SERVER_URL, help="MCP server URL")
@click.option("--timeout", default=TIMEOUT_SECONDS, help="Request timeout in seconds")
@click.option("--verbose", is_flag=True, help="Verbose output")
def main(tool_name: Optional[str], params: str, list_tools: bool, server: str, timeout: int, verbose: bool):
    """
    🔧 MISO Ultimate MCP CLI
    
    Execute MCP tools from the command line with rich formatting.
    
    Examples:
        mcp-cli.py benchmark_matrix --params '{"preset":"quick"}'
        mcp-cli.py system_metrics
        mcp-cli.py --list-tools
    """
    
    global MCP_SERVER_URL, TIMEOUT_SECONDS
    MCP_SERVER_URL = server
    TIMEOUT_SECONDS = timeout
    
    # Display header
    console.print(Panel(
        "🔥 MISO Ultimate MCP CLI\n"
        f"🌐 Server: {MCP_SERVER_URL}\n"
        f"⏱️  Timeout: {TIMEOUT_SECONDS}s",
        title="🚀 MCP Command Line Interface",
        border_style="blue"
    ))
    
    # List tools if requested
    if list_tools:
        display_available_tools()
        return
    
    # Validate tool name
    if not tool_name:
        console.print("❌ Tool name is required. Use --list-tools to see available tools.")
        sys.exit(1)
    
    # Parse parameters
    try:
        parsed_params = json.loads(params)
    except json.JSONDecodeError as e:
        console.print(f"❌ Invalid JSON in --params: {e}")
        sys.exit(1)
    
    # Prepare request payload
    payload = {
        "tool": tool_name,
        "params": parsed_params
    }
    
    if verbose:
        console.print(f"\n📤 Request payload:")
        syntax = Syntax(json.dumps(payload, indent=2), "json", theme="monokai")
        console.print(syntax)
    
    # Execute request with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Executing {tool_name}...", total=None)
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{MCP_SERVER_URL}/mcp",
                json=payload,
                timeout=timeout
            )
            execution_time = time.time() - start_time
            
            progress.update(task, completed=True)
            
            # Display execution time
            console.print(f"\n⏱️  Execution time: {execution_time:.2f}s")
            console.print(f"📡 HTTP Status: {response.status_code}")
            
            # Parse and display response
            if response.headers.get('content-type', '').startswith('application/json'):
                response_data = response.json()
                format_response(response_data)
            else:
                console.print(f"📄 Raw response: {response.text}")
                
        except requests.exceptions.Timeout:
            console.print(f"⏰ Request timed out after {timeout} seconds")
            sys.exit(1)
        except requests.exceptions.ConnectionError:
            console.print(f"🔌 Cannot connect to MCP server at {MCP_SERVER_URL}")
            console.print("💡 Make sure the MCP server is running")
            sys.exit(1)
        except Exception as e:
            console.print(f"💥 Unexpected error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
