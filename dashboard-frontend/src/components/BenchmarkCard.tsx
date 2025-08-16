import React from 'react';
import { Play, Clock, CheckCircle, AlertCircle, Activity, Zap } from 'lucide-react';

interface LastRunInfo {
  id: string;
  timestamp: string;
  status: string;
}

interface BenchmarkCardProps {
  name: string;
  description: string;
  status: 'idle' | 'connecting' | 'connected' | 'running' | 'completed' | 'error' | 'disconnected';
  lastRun?: LastRunInfo | null;
  onStart: () => void;
  isLoading?: boolean;
}

const BenchmarkCard: React.FC<BenchmarkCardProps> = ({
  name,
  description,
  status,
  lastRun,
  onStart,
  isLoading = false
}) => {
  // Get status icon and color
  const getStatusIcon = () => {
    switch (status) {
      case 'running':
      case 'connecting':
        return <Activity className="h-4 w-4 text-yellow-500 animate-pulse" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'connected':
        return <Zap className="h-4 w-4 text-blue-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'running':
        return 'Running';
      case 'connecting':
        return 'Connecting';
      case 'connected':
        return 'Connected';
      case 'completed':
        return 'Completed';
      case 'error':
        return 'Error';
      case 'disconnected':
        return 'Disconnected';
      default:
        return 'Idle';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'running':
      case 'connecting':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200 dark:text-yellow-400 dark:bg-yellow-900/20 dark:border-yellow-800';
      case 'completed':
      case 'connected':
        return 'text-green-600 bg-green-50 border-green-200 dark:text-green-400 dark:bg-green-900/20 dark:border-green-800';
      case 'error':
        return 'text-red-600 bg-red-50 border-red-200 dark:text-red-400 dark:bg-red-900/20 dark:border-red-800';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200 dark:text-gray-400 dark:bg-gray-900/20 dark:border-gray-700';
    }
  };

  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch {
      return 'Unknown';
    }
  };

  // Get tool display name
  const getToolDisplayName = (toolName: string) => {
    return toolName
      .replace('benchmark_', '')
      .replace(/_/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase());
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow duration-200">
      {/* Header */}
      <div className="p-6 pb-4">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              {getToolDisplayName(name)}
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-300 leading-relaxed">
              {description}
            </p>
          </div>
          
          {/* Status Badge */}
          <div className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium border ${getStatusColor()}`}>
            {getStatusIcon()}
            <span className="ml-1.5">{getStatusText()}</span>
          </div>
        </div>
      </div>

      {/* Last Run Info */}
      {lastRun && (
        <div className="px-6 pb-4">
          <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600 dark:text-gray-400">Last Run:</span>
              <span className="font-medium text-gray-900 dark:text-white">
                {lastRun.id.substring(0, 8)}...
              </span>
            </div>
            <div className="flex items-center justify-between text-sm mt-1">
              <span className="text-gray-600 dark:text-gray-400">Time:</span>
              <span className="text-gray-900 dark:text-white">
                {formatTimestamp(lastRun.timestamp)}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="px-6 pb-6">
        <button
          onClick={onStart}
          disabled={isLoading || status === 'running' || status === 'connecting'}
          className={`w-full flex items-center justify-center px-4 py-2.5 rounded-lg font-medium text-sm transition-colors duration-200 ${
            isLoading || status === 'running' || status === 'connecting'
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed dark:bg-gray-700 dark:text-gray-500'
              : 'bg-blue-600 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800'
          }`}
        >
          {isLoading ? (
            <>
              <Activity className="h-4 w-4 mr-2 animate-spin" />
              Starting...
            </>
          ) : status === 'running' ? (
            <>
              <Activity className="h-4 w-4 mr-2 animate-pulse" />
              Running
            </>
          ) : status === 'connecting' ? (
            <>
              <Activity className="h-4 w-4 mr-2 animate-spin" />
              Connecting
            </>
          ) : (
            <>
              <Play className="h-4 w-4 mr-2" />
              Start Benchmark
            </>
          )}
        </button>
      </div>

      {/* Real-time Status Indicator */}
      {(status === 'running' || status === 'connecting') && (
        <div className="absolute top-2 right-2">
          <div className="w-3 h-3 bg-yellow-400 rounded-full animate-pulse"></div>
        </div>
      )}
    </div>
  );
};

export default BenchmarkCard;
