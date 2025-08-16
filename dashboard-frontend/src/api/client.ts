import axios from 'axios';

// Base API configuration - REAL BACKEND INTEGRATION
const API_BASE_URL = 'http://127.0.0.1:8001';
const MCP_BASE_URL = 'http://127.0.0.1:8001';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

// Type definitions for MCP Tools
export interface MCPTool {
  name: string;
  description: string;
  parameters: Record<string, any>;
}

export interface MCPRequest {
  tool: string;
  params: Record<string, any>;
}

export interface MCPResponse {
  tool: string;
  status: string;
  timestamp: string;
  result: Record<string, any>;
}

export interface BenchmarkResult {
  id: string;
  tool: string;
  status: 'running' | 'completed' | 'failed';
  startTime: string;
  endTime?: string;
  params: Record<string, any>;
  results?: Record<string, any>;
}

export interface SystemStatus {
  server: string;
  version: string;
  status: string;
  timestamp: string;
  tools_available: number;
}

// API Functions - REAL BACKEND CALLS
export const api = {
  // Get system status
  getSystemStatus: async (): Promise<SystemStatus> => {
    const response = await apiClient.get('/');
    return response.data;
  },

  // Get all available MCP tools
  getTools: async (): Promise<MCPTool[]> => {
    const response = await apiClient.get('/tools');
    return response.data;
  },

  // Get tool discovery summary
  getDiscovery: async () => {
    const response = await apiClient.get('/tools/discovery');
    return response.data;
  },

  // Get detailed tool introspection
  introspectTool: async (toolName: string) => {
    const response = await apiClient.get(`/tools/${toolName}/introspect`);
    return response.data;
  },

  // Execute MCP tool
  runBenchmark: async (tool: string, params: Record<string, any>): Promise<MCPResponse> => {
    const request: MCPRequest = { tool, params };
    const response = await apiClient.post('/mcp', request);
    return response.data;
  },

  // Execute MCP tool with streaming
  runBenchmarkStreaming: async (tool: string, params: Record<string, any>) => {
    const request: MCPRequest = { tool, params };
    const response = await apiClient.post('/mcp/streaming', request);
    return response.data;
  },

  // Get health check
  getHealth: async () => {
    const response = await apiClient.get('/health');
    return response.data;
  },

  // Get OpenAPI fusion status
  getOpenAPIFusion: async () => {
    const response = await apiClient.get('/openapi/fusion');
    return response.data;
  },

  // Get prompt templates
  getPromptTemplates: async () => {
    const response = await apiClient.get('/openapi/templates');
    return response.data;
  },

  // Sync OpenAPI schema
  syncOpenAPI: async () => {
    const response = await apiClient.post('/openapi/sync');
    return response.data;
  },

  // Get streaming status
  getStreamingStatus: async () => {
    const response = await apiClient.get('/streaming/status');
    return response.data;
  },

  // Get active streaming sessions
  getStreamingSessions: async () => {
    const response = await apiClient.get('/streaming/sessions');
    return response.data;
  },
};

// Mock data for development (fallback)
export const mockTools: MCPTool[] = [
  {
    name: 'benchmark_matrix',
    description: 'Matrix multiplication benchmark with configurable parameters',
    parameters: {
      preset: { type: 'string', enum: ['small', 'medium', 'large'], default: 'medium' },
      iterations: { type: 'number', default: 100 },
      matrix_sizes: { type: 'array', default: [64, 128, 256] },
      backend_preference: { type: 'string', enum: ['cpu', 'gpu', 'auto'], default: 'auto' },
      timeout_seconds: { type: 'number', default: 300 }
    }
  },
  {
    name: 'benchmark_quantum',
    description: 'Quantum computing simulation benchmark',
    parameters: {
      preset: { type: 'string', enum: ['small', 'medium', 'large'], default: 'medium' },
      qubits: { type: 'number', default: 8 },
      depth: { type: 'number', default: 10 },
      shots: { type: 'number', default: 1000 }
    }
  },
  {
    name: 'benchmark_all',
    description: 'Comprehensive benchmark suite',
    parameters: {
      preset: { type: 'string', enum: ['quick', 'standard', 'comprehensive'], default: 'standard' },
      parallel: { type: 'boolean', default: true },
      timeout_seconds: { type: 'number', default: 600 }
    }
  }
];

export const mockResults: BenchmarkResult[] = [
  {
    id: 'bench_001',
    tool: 'benchmark_matrix',
    status: 'completed',
    startTime: '2025-07-31T10:30:00Z',
    endTime: '2025-07-31T10:35:00Z',
    params: { preset: 'medium', iterations: 100 },
    results: { avg_time: 2.34, throughput: 1250, efficiency: 0.89 }
  },
  {
    id: 'bench_002',
    tool: 'benchmark_quantum',
    status: 'running',
    startTime: '2025-07-31T11:00:00Z',
    params: { preset: 'small', qubits: 8 },
  },
  {
    id: 'bench_003',
    tool: 'benchmark_all',
    status: 'failed',
    startTime: '2025-07-31T09:15:00Z',
    endTime: '2025-07-31T09:20:00Z',
    params: { preset: 'comprehensive' },
    results: { error: 'Timeout exceeded' }
  }
];

// Export individual functions for easier imports
export const getSystemStatus = api.getSystemStatus;
export const getMCPTools = api.getTools;
export const getDiscovery = api.getDiscovery;
export const introspectTool = api.introspectTool;
export const executeBenchmark = api.runBenchmark;
export const runBenchmarkStreaming = api.runBenchmarkStreaming;
export const getHealth = api.getHealth;
export const getOpenAPIFusion = api.getOpenAPIFusion;
export const getPromptTemplates = api.getPromptTemplates;
export const syncOpenAPI = api.syncOpenAPI;
export const getStreamingStatus = api.getStreamingStatus;
export const getStreamingSessions = api.getStreamingSessions;

export default api;
