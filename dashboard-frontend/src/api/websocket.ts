// WebSocket helper for real-time streaming from MISO Ultimate MCP Server
export interface StreamMessage {
  type: 'status' | 'progress' | 'result' | 'error' | 'log';
  timestamp: string;
  data: any;
  session_id?: string;
  tool_name?: string;
}

export interface StreamingSession {
  sessionId: string;
  toolName: string;
  status: 'connecting' | 'connected' | 'running' | 'completed' | 'error' | 'disconnected';
  messages: StreamMessage[];
  onMessage?: (message: StreamMessage) => void;
  onStatusChange?: (status: string) => void;
  onError?: (error: Error) => void;
}

class WebSocketManager {
  private connections: Map<string, WebSocket> = new Map();
  private sessions: Map<string, StreamingSession> = new Map();

  // Connect to streaming session
  connectToStream(
    sessionId: string, 
    toolName: string,
    onMessage?: (message: StreamMessage) => void,
    onStatusChange?: (status: string) => void,
    onError?: (error: Error) => void
  ): StreamingSession {
    
    // Create session object
    const session: StreamingSession = {
      sessionId,
      toolName,
      status: 'connecting',
      messages: [],
      onMessage,
      onStatusChange,
      onError
    };

    this.sessions.set(sessionId, session);

    // WebSocket URL for MISO Ultimate MCP Server
    const wsUrl = `ws://127.0.0.1:8002/stream/${sessionId}`;
    
    try {
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log(`🔌 WebSocket connected for session: ${sessionId}`);
        session.status = 'connected';
        onStatusChange?.('connected');
      };

      ws.onmessage = (event) => {
        try {
          const message: StreamMessage = JSON.parse(event.data);
          session.messages.push(message);
          
          // Update session status based on message type
          if (message.type === 'status') {
            session.status = message.data.status || 'running';
            onStatusChange?.(session.status);
          }
          
          onMessage?.(message);
          console.log(`📨 Stream message:`, message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
          onError?.(new Error('Invalid message format'));
        }
      };

      ws.onerror = (error) => {
        console.error(`❌ WebSocket error for session ${sessionId}:`, error);
        session.status = 'error';
        onStatusChange?.('error');
        onError?.(new Error('WebSocket connection error'));
      };

      ws.onclose = (event) => {
        console.log(`🔌 WebSocket closed for session ${sessionId}:`, event.code, event.reason);
        session.status = 'disconnected';
        onStatusChange?.('disconnected');
        this.connections.delete(sessionId);
        this.sessions.delete(sessionId);
      };

      this.connections.set(sessionId, ws);
      
    } catch (error) {
      console.error(`Failed to create WebSocket connection:`, error);
      session.status = 'error';
      onError?.(error as Error);
    }

    return session;
  }

  // Disconnect from streaming session
  disconnect(sessionId: string): void {
    const ws = this.connections.get(sessionId);
    if (ws) {
      ws.close();
      this.connections.delete(sessionId);
    }
    this.sessions.delete(sessionId);
  }

  // Get session info
  getSession(sessionId: string): StreamingSession | undefined {
    return this.sessions.get(sessionId);
  }

  // Get all active sessions
  getActiveSessions(): StreamingSession[] {
    return Array.from(this.sessions.values());
  }

  // Send message to session (if supported)
  sendMessage(sessionId: string, message: any): boolean {
    const ws = this.connections.get(sessionId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
      return true;
    }
    return false;
  }

  // Cleanup all connections
  disconnectAll(): void {
    for (const [sessionId] of this.connections) {
      this.disconnect(sessionId);
    }
  }
}

// Global WebSocket manager instance
export const wsManager = new WebSocketManager();

// Helper functions for easy usage
export const streamingUtils = {
  // Start streaming for a benchmark
  startBenchmarkStream: (
    toolName: string,
    onMessage?: (message: StreamMessage) => void,
    onStatusChange?: (status: string) => void,
    onError?: (error: Error) => void
  ) => {
    const sessionId = `${toolName}_${Date.now()}`;
    return wsManager.connectToStream(sessionId, toolName, onMessage, onStatusChange, onError);
  },

  // Mock streaming for development (when WebSocket server is not available)
  mockStream: (
    toolName: string,
    onMessage?: (message: StreamMessage) => void,
    onStatusChange?: (status: string) => void
  ) => {
    const sessionId = `mock_${toolName}_${Date.now()}`;
    
    // Simulate streaming messages
    const mockMessages: StreamMessage[] = [
      { type: 'status', timestamp: new Date().toISOString(), data: { status: 'starting' } },
      { type: 'progress', timestamp: new Date().toISOString(), data: { progress: 0 } },
      { type: 'log', timestamp: new Date().toISOString(), data: { message: 'Initializing benchmark...' } },
      { type: 'progress', timestamp: new Date().toISOString(), data: { progress: 25 } },
      { type: 'log', timestamp: new Date().toISOString(), data: { message: 'Running matrix operations...' } },
      { type: 'progress', timestamp: new Date().toISOString(), data: { progress: 50 } },
      { type: 'progress', timestamp: new Date().toISOString(), data: { progress: 75 } },
      { type: 'log', timestamp: new Date().toISOString(), data: { message: 'Calculating results...' } },
      { type: 'progress', timestamp: new Date().toISOString(), data: { progress: 100 } },
      { type: 'result', timestamp: new Date().toISOString(), data: { 
        avg_time: 2.34, 
        throughput: 1250, 
        efficiency: 0.89,
        status: 'completed'
      }},
      { type: 'status', timestamp: new Date().toISOString(), data: { status: 'completed' } }
    ];

    let messageIndex = 0;
    const interval = setInterval(() => {
      if (messageIndex < mockMessages.length) {
        const message = mockMessages[messageIndex];
        onMessage?.(message);
        
        if (message.type === 'status') {
          onStatusChange?.(message.data.status);
        }
        
        messageIndex++;
      } else {
        clearInterval(interval);
      }
    }, 1000);

    return {
      sessionId,
      toolName,
      status: 'running' as const,
      messages: [],
      cleanup: () => clearInterval(interval)
    };
  }
};

export default wsManager;
