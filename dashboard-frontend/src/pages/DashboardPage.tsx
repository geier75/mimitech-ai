import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { MCPTool, SystemStatus, MCPResponse } from '../api/client';
import { getSystemStatus, getMCPTools, getDiscovery } from '../api/client';
import { streamingUtils } from '../api/websocket';
import type { StreamMessage } from '../api/websocket';
import BenchmarkCard from '../components/BenchmarkCard';
import RunModal from '../components/RunModal';
import VoiceCommand from '../components/VoiceCommand';
import { Activity, Server, Zap, AlertCircle, CheckCircle, Clock } from 'lucide-react';

const DashboardPage: React.FC = () => {
  const [selectedTool, setSelectedTool] = useState<MCPTool | null>(null);
  const [isRunModalOpen, setIsRunModalOpen] = useState(false);
  const [streamingSessions, setStreamingSessions] = useState<Map<string, any>>(new Map());
  const queryClient = useQueryClient();

  // Query for system status
  const { data: systemStatus, isLoading: statusLoading, error: statusError } = useQuery<SystemStatus>({
    queryKey: ['systemStatus'],
    queryFn: api.getSystemStatus,
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Query for available tools
  const { data: tools, isLoading: toolsLoading, error: toolsError } = useQuery<MCPTool[]>({
    queryKey: ['tools'],
    queryFn: api.getTools,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Query for discovery summary
  const { data: discovery } = useQuery({
    queryKey: ['discovery'],
    queryFn: api.getDiscovery,
    refetchInterval: 60000, // Refresh every minute
  });

  // Mutation for running benchmarks
  const runBenchmarkMutation = useMutation({
    mutationFn: ({ tool, params }: { tool: string; params: Record<string, any> }) =>
      api.runBenchmark(tool, params),
    onSuccess: (data: MCPResponse) => {
      console.log('✅ Benchmark started successfully:', data);
      // Invalidate and refetch relevant queries
      queryClient.invalidateQueries({ queryKey: ['benchmarkResults'] });
      
      // Start streaming session for live updates
      if (data.status === 'success' && data.result?.session_id) {
        startStreamingSession(data.tool, data.result.session_id);
      }
    },
    onError: (error) => {
      console.error('❌ Benchmark failed:', error);
    },
  });

  // Start streaming session for live updates
  const startStreamingSession = (toolName: string, sessionId?: string) => {
    const session = streamingUtils.startBenchmarkStream(
      toolName,
      (message: StreamMessage) => {
        console.log(`📨 Stream message for ${toolName}:`, message);
        // Update UI based on streaming messages
        setStreamingSessions(prev => {
          const updated = new Map(prev);
          const existing = updated.get(toolName) || { messages: [] };
          existing.messages.push(message);
          existing.lastMessage = message;
          updated.set(toolName, existing);
          return updated;
        });
      },
      (status: string) => {
        console.log(`🔄 Status change for ${toolName}:`, status);
        setStreamingSessions(prev => {
          const updated = new Map(prev);
          const existing = updated.get(toolName) || {};
          existing.status = status;
          updated.set(toolName, existing);
          return updated;
        });
      },
      (error: Error) => {
        console.error(`❌ Streaming error for ${toolName}:`, error);
      }
    );

    setStreamingSessions(prev => {
      const updated = new Map(prev);
      updated.set(toolName, { session, status: 'connecting', messages: [] });
      return updated;
    });
  };

  // Handle benchmark start
  const handleStartBenchmark = (tool: MCPTool) => {
    setSelectedTool(tool);
    setIsRunModalOpen(true);
  };

  // Handle benchmark execution
  const handleRunBenchmark = (params: Record<string, any>) => {
    if (selectedTool) {
      runBenchmarkMutation.mutate({
        tool: selectedTool.name,
        params,
      });
      setIsRunModalOpen(false);
      setSelectedTool(null);
    }
  };

  // Get tool status from streaming sessions
  const getToolStatus = (toolName: string) => {
    const session = streamingSessions.get(toolName);
    if (session?.status) {
      return session.status;
    }
    return 'idle';
  };

  // Get last run info from streaming sessions
  const getLastRunInfo = (toolName: string) => {
    const session = streamingSessions.get(toolName);
    if (session?.lastMessage) {
      return {
        id: session.session?.sessionId || 'unknown',
        timestamp: session.lastMessage.timestamp,
        status: session.status || 'unknown'
      };
    }
    return null;
  };

  if (statusError || toolsError) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6">
        <div className="max-w-4xl mx-auto">
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
            <div className="flex items-center space-x-3">
              <AlertCircle className="h-6 w-6 text-red-600 dark:text-red-400" />
              <div>
                <h3 className="text-lg font-semibold text-red-800 dark:text-red-200">
                  Backend Connection Error
                </h3>
                <p className="text-red-600 dark:text-red-300 mt-1">
                  Unable to connect to MISO Ultimate MCP Server at http://127.0.0.1:8001
                </p>
                <p className="text-sm text-red-500 dark:text-red-400 mt-2">
                  Please ensure the backend server is running and accessible.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header Section */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                MISO Ultimate Dashboard
              </h1>
              <p className="text-gray-600 dark:text-gray-300 mt-1">
                Real-time benchmark execution and monitoring
              </p>
            </div>
            
            {/* System Status */}
            <div className="flex items-center space-x-4">
              {statusLoading ? (
                <div className="flex items-center space-x-2">
                  <Clock className="h-5 w-5 text-gray-400 animate-spin" />
                  <span className="text-sm text-gray-500">Connecting...</span>
                </div>
              ) : systemStatus ? (
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-5 w-5 text-green-500" />
                  <div className="text-sm">
                    <div className="font-medium text-gray-900 dark:text-white">
                      {systemStatus.server} v{systemStatus.version}
                    </div>
                    <div className="text-gray-500 dark:text-gray-400">
                      {systemStatus.tools_available} tools available
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center">
              <Server className="h-8 w-8 text-blue-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Server Status</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {systemStatus?.status || 'Unknown'}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center">
              <Zap className="h-8 w-8 text-green-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Available Tools</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {tools?.length || 0}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center">
              <Activity className="h-8 w-8 text-yellow-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Sessions</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {streamingSessions.size}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center">
              <CheckCircle className="h-8 w-8 text-purple-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Discovered</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {discovery?.tools_discovered || 0}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Voice Command Section */}
        <div className="mb-8">
          <VoiceCommand onBenchmarkExecuted={(result) => {
            console.log('Voice command executed benchmark:', result);
            // Voice command executed successfully - data will auto-refresh
          }} />
        </div>

        {/* Tools Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {toolsLoading ? (
            // Loading skeleton
            Array.from({ length: 3 }).map((_, index) => (
              <div key={index} className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 animate-pulse">
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-4"></div>
                <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
              </div>
            ))
          ) : tools && tools.length > 0 ? (
            tools.map((tool) => (
              <BenchmarkCard
                key={tool.name}
                name={tool.name}
                description={tool.description}
                status={getToolStatus(tool.name)}
                lastRun={getLastRunInfo(tool.name)}
                onStart={() => handleStartBenchmark(tool)}
                isLoading={runBenchmarkMutation.isPending && selectedTool?.name === tool.name}
              />
            ))
          ) : (
            <div className="col-span-full text-center py-12">
              <AlertCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                No Tools Available
              </h3>
              <p className="text-gray-500 dark:text-gray-400">
                No MCP tools were found. Please check the backend configuration.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Run Modal */}
      {isRunModalOpen && selectedTool && (
        <RunModal
          tool={selectedTool}
          isOpen={isRunModalOpen}
          onClose={() => {
            setIsRunModalOpen(false);
            setSelectedTool(null);
          }}
          onRun={handleRunBenchmark}
          isLoading={runBenchmarkMutation.isPending}
        />
      )}
    </div>
  );
};

export default DashboardPage;
