#!/usr/bin/env python3
"""
🌊 WebSocket Streaming Bridge for MCP Tools
==========================================

Real-time streaming bridge that enables WebSocket-based tool execution
with live status updates and streaming responses.

Author: MISO Ultimate Team
Date: 30.07.2025
"""

import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum

import websockets
from pydantic import BaseModel


class StreamingStatus(str, Enum):
    """Streaming status enumeration"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    EXECUTING = "executing"
    STREAMING = "streaming"
    COMPLETED = "completed"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class StreamingMessage(BaseModel):
    """WebSocket streaming message"""
    message_id: str
    session_id: str
    tool_name: str
    status: StreamingStatus
    timestamp: str
    data: Dict[str, Any] = {}
    progress: Optional[float] = None
    error: Optional[str] = None


class StreamingSession(BaseModel):
    """WebSocket streaming session"""
    session_id: str
    tool_name: str
    websocket: Any = None  # WebSocket connection
    status: StreamingStatus = StreamingStatus.CONNECTING
    created_at: str
    last_update: str
    messages: List[StreamingMessage] = []
    
    class Config:
        arbitrary_types_allowed = True


class WebSocketStreamingBridge:
    """
    WebSocket Streaming Bridge for real-time MCP tool execution.
    
    Enables streaming tool responses, live progress updates,
    and real-time status broadcasting.
    """
    
    def __init__(self, backend_ws_url: str = "ws://127.0.0.1:8000/ws"):
        """
        Initialize WebSocket streaming bridge.
        
        Args:
            backend_ws_url: WebSocket URL of backend server
        """
        self.backend_ws_url = backend_ws_url
        self.active_sessions: Dict[str, StreamingSession] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.is_running = False
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """
        Register handler for specific message types.
        
        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        self.message_handlers[message_type] = handler
        print(f"📝 Registered handler for message type: {message_type}")
    
    async def create_streaming_session(
        self, 
        tool_name: str, 
        websocket
    ) -> StreamingSession:
        """
        Create new streaming session for tool execution.
        
        Args:
            tool_name: Name of tool to execute
            websocket: WebSocket connection
            
        Returns:
            Created streaming session
        """
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        session = StreamingSession(
            session_id=session_id,
            tool_name=tool_name,
            websocket=websocket,
            status=StreamingStatus.CONNECTING,
            created_at=timestamp,
            last_update=timestamp
        )
        
        self.active_sessions[session_id] = session
        print(f"🌊 Created streaming session: {session_id} for tool: {tool_name}")
        
        return session
    
    async def send_streaming_message(
        self, 
        session: StreamingSession,
        status: StreamingStatus,
        data: Dict[str, Any] = {},
        progress: Optional[float] = None,
        error: Optional[str] = None
    ):
        """
        Send streaming message to client.
        
        Args:
            session: Streaming session
            status: Current status
            data: Message data
            progress: Progress percentage (0-100)
            error: Error message if any
        """
        message = StreamingMessage(
            message_id=str(uuid.uuid4()),
            session_id=session.session_id,
            tool_name=session.tool_name,
            status=status,
            timestamp=datetime.now().isoformat(),
            data=data,
            progress=progress,
            error=error
        )
        
        # Update session
        session.status = status
        session.last_update = message.timestamp
        session.messages.append(message)
        
        # Send to client
        if session.websocket:
            try:
                await session.websocket.send(message.model_dump_json())
                print(f"📤 Sent streaming message: {status} to session {session.session_id}")
            except Exception as e:
                print(f"❌ Failed to send message to session {session.session_id}: {e}")
                session.status = StreamingStatus.ERROR
    
    async def execute_streaming_tool(
        self, 
        session: StreamingSession,
        tool_instance,
        params: Dict[str, Any]
    ):
        """
        Execute tool with streaming updates.
        
        Args:
            session: Streaming session
            tool_instance: Tool instance to execute
            params: Tool parameters
        """
        try:
            # Update status to executing
            await self.send_streaming_message(
                session, 
                StreamingStatus.EXECUTING,
                {"message": f"Starting execution of {session.tool_name}"}
            )
            
            # Execute tool
            if hasattr(tool_instance, 'run'):
                # Check if tool supports streaming
                if hasattr(tool_instance, 'run_streaming'):
                    # Use streaming version
                    async for progress_data in tool_instance.run_streaming(params):
                        await self.send_streaming_message(
                            session,
                            StreamingStatus.STREAMING,
                            progress_data,
                            progress_data.get('progress')
                        )
                else:
                    # Regular execution with simulated streaming
                    await self.send_streaming_message(
                        session,
                        StreamingStatus.STREAMING,
                        {"message": "Executing tool...", "progress": 25}
                    )
                    
                    result = await tool_instance.run(params)
                    
                    await self.send_streaming_message(
                        session,
                        StreamingStatus.STREAMING,
                        {"message": "Processing results...", "progress": 75}
                    )
                    
                    # Send final result
                    await self.send_streaming_message(
                        session,
                        StreamingStatus.COMPLETED,
                        {"result": result, "progress": 100}
                    )
            else:
                raise Exception(f"Tool {session.tool_name} does not have run method")
                
        except Exception as e:
            await self.send_streaming_message(
                session,
                StreamingStatus.ERROR,
                error=str(e)
            )
    
    async def connect_to_backend_stream(self, session: StreamingSession):
        """
        Connect to backend WebSocket for live updates.
        
        Args:
            session: Streaming session
        """
        try:
            async with websockets.connect(self.backend_ws_url) as backend_ws:
                print(f"🔗 Connected to backend WebSocket for session {session.session_id}")
                
                await self.send_streaming_message(
                    session,
                    StreamingStatus.CONNECTED,
                    {"message": "Connected to backend stream"}
                )
                
                # Listen for backend updates
                async for message in backend_ws:
                    try:
                        backend_data = json.loads(message)
                        
                        # Forward backend updates to client
                        await self.send_streaming_message(
                            session,
                            StreamingStatus.STREAMING,
                            {"backend_update": backend_data}
                        )
                        
                    except json.JSONDecodeError:
                        print(f"⚠️ Invalid JSON from backend: {message}")
                        
        except Exception as e:
            print(f"❌ Backend WebSocket connection failed: {e}")
            await self.send_streaming_message(
                session,
                StreamingStatus.ERROR,
                error=f"Backend connection failed: {str(e)}"
            )
    
    async def handle_websocket_connection(self, websocket, path: str):
        """
        Handle incoming WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        print(f"🌊 New WebSocket connection: {path}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_websocket_message(websocket, data)
                    
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "error": "Invalid JSON message",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            print("🔌 WebSocket connection closed")
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
    
    async def process_websocket_message(self, websocket, data: Dict[str, Any]):
        """
        Process incoming WebSocket message.
        
        Args:
            websocket: WebSocket connection
            data: Message data
        """
        message_type = data.get("type", "unknown")
        
        if message_type == "start_streaming_tool":
            # Start streaming tool execution
            tool_name = data.get("tool_name")
            params = data.get("params", {})
            
            if not tool_name:
                await websocket.send(json.dumps({
                    "error": "tool_name is required",
                    "timestamp": datetime.now().isoformat()
                }))
                return
            
            # Create streaming session
            session = await self.create_streaming_session(tool_name, websocket)
            
            # TODO: Get tool instance from registry and execute
            await self.send_streaming_message(
                session,
                StreamingStatus.EXECUTING,
                {"message": f"Would execute {tool_name} with params: {params}"}
            )
            
        elif message_type == "get_session_status":
            # Get session status
            session_id = data.get("session_id")
            session = self.active_sessions.get(session_id)
            
            if session:
                await websocket.send(json.dumps({
                    "type": "session_status",
                    "session": session.model_dump(exclude={"websocket"}),
                    "timestamp": datetime.now().isoformat()
                }))
            else:
                await websocket.send(json.dumps({
                    "error": f"Session {session_id} not found",
                    "timestamp": datetime.now().isoformat()
                }))
        
        elif message_type in self.message_handlers:
            # Custom message handler
            handler = self.message_handlers[message_type]
            await handler(websocket, data)
        
        else:
            await websocket.send(json.dumps({
                "error": f"Unknown message type: {message_type}",
                "timestamp": datetime.now().isoformat()
            }))
    
    def get_streaming_summary(self) -> Dict[str, Any]:
        """Get streaming bridge summary"""
        return {
            "active_sessions": len(self.active_sessions),
            "backend_ws_url": self.backend_ws_url,
            "message_handlers": list(self.message_handlers.keys()),
            "is_running": self.is_running,
            "sessions": [
                {
                    "session_id": session.session_id,
                    "tool_name": session.tool_name,
                    "status": session.status,
                    "created_at": session.created_at,
                    "message_count": len(session.messages)
                }
                for session in self.active_sessions.values()
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    async def cleanup_session(self, session_id: str):
        """
        Clean up streaming session.
        
        Args:
            session_id: Session ID to clean up
        """
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Send disconnection message
            await self.send_streaming_message(
                session,
                StreamingStatus.DISCONNECTED,
                {"message": "Session cleanup"}
            )
            
            # Remove session
            del self.active_sessions[session_id]
            print(f"🧹 Cleaned up session: {session_id}")
    
    async def start_streaming_server(self, host: str = "127.0.0.1", port: int = 8002):
        """
        Start WebSocket streaming server.
        
        Args:
            host: Server host
            port: Server port
        """
        self.is_running = True
        print(f"🌊 Starting WebSocket streaming server on {host}:{port}")
        
        async with websockets.serve(self.handle_websocket_connection, host, port):
            print(f"✅ WebSocket streaming server running on ws://{host}:{port}")
            await asyncio.Future()  # Run forever
