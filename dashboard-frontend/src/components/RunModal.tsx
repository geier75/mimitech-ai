import React, { useState, useEffect } from 'react';
import { X, Play, Settings, Info, AlertCircle } from 'lucide-react';
import { MCPTool } from '../api/client';

interface RunModalProps {
  tool: MCPTool;
  isOpen: boolean;
  onClose: () => void;
  onRun: (params: Record<string, any>) => void;
  isLoading?: boolean;
}

const RunModal: React.FC<RunModalProps> = ({
  tool,
  isOpen,
  onClose,
  onRun,
  isLoading = false
}) => {
  const [params, setParams] = useState<Record<string, any>>({});
  const [selectedPreset, setSelectedPreset] = useState<string>('');
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Initialize params with defaults when tool changes
  useEffect(() => {
    if (tool && tool.parameters) {
      const defaultParams: Record<string, any> = {};
      
      Object.entries(tool.parameters).forEach(([key, schema]: [string, any]) => {
        if (schema.default !== undefined) {
          defaultParams[key] = schema.default;
        } else if (schema.type === 'string' && schema.enum) {
          defaultParams[key] = schema.enum[0];
        } else if (schema.type === 'number') {
          defaultParams[key] = 1;
        } else if (schema.type === 'boolean') {
          defaultParams[key] = false;
        } else if (schema.type === 'array') {
          defaultParams[key] = schema.default || [];
        }
      });
      
      setParams(defaultParams);
      
      // Set default preset if available
      if (defaultParams.preset) {
        setSelectedPreset(defaultParams.preset);
      }
    }
  }, [tool]);

  // Preset configurations
  const getPresetConfigs = () => {
    const toolName = tool.name;
    
    switch (toolName) {
      case 'benchmark_matrix':
        return {
          small: {
            preset: 'small',
            iterations: 50,
            matrix_sizes: [32, 64],
            backend_preference: 'auto',
            timeout_seconds: 120
          },
          medium: {
            preset: 'medium',
            iterations: 100,
            matrix_sizes: [64, 128, 256],
            backend_preference: 'auto',
            timeout_seconds: 300
          },
          large: {
            preset: 'large',
            iterations: 200,
            matrix_sizes: [128, 256, 512],
            backend_preference: 'auto',
            timeout_seconds: 600
          }
        };
      
      case 'benchmark_quantum':
        return {
          small: {
            preset: 'small',
            qubits: 4,
            depth: 5,
            shots: 500
          },
          medium: {
            preset: 'medium',
            qubits: 8,
            depth: 10,
            shots: 1000
          },
          large: {
            preset: 'large',
            qubits: 12,
            depth: 20,
            shots: 2000
          }
        };
      
      case 'benchmark_all':
        return {
          quick: {
            preset: 'quick',
            parallel: true,
            timeout_seconds: 300
          },
          standard: {
            preset: 'standard',
            parallel: true,
            timeout_seconds: 600
          },
          comprehensive: {
            preset: 'comprehensive',
            parallel: false,
            timeout_seconds: 1200
          }
        };
      
      default:
        return {};
    }
  };

  // Handle preset change
  const handlePresetChange = (presetName: string) => {
    setSelectedPreset(presetName);
    const presets = getPresetConfigs();
    if (presets[presetName]) {
      setParams({ ...params, ...presets[presetName] });
    }
  };

  // Handle parameter change
  const handleParamChange = (key: string, value: any) => {
    setParams(prev => ({ ...prev, [key]: value }));
    
    // Clear error for this field
    if (errors[key]) {
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[key];
        return newErrors;
      });
    }
  };

  // Validate parameters
  const validateParams = (): boolean => {
    const newErrors: Record<string, string> = {};
    
    if (!tool.parameters) return true;
    
    Object.entries(tool.parameters).forEach(([key, schema]: [string, any]) => {
      const value = params[key];
      
      // Required field validation
      if (schema.required && (value === undefined || value === null || value === '')) {
        newErrors[key] = `${key} is required`;
        return;
      }
      
      // Type validation
      if (value !== undefined && value !== null) {
        if (schema.type === 'number' && (isNaN(value) || typeof value !== 'number')) {
          newErrors[key] = `${key} must be a number`;
        } else if (schema.type === 'string' && typeof value !== 'string') {
          newErrors[key] = `${key} must be a string`;
        } else if (schema.type === 'boolean' && typeof value !== 'boolean') {
          newErrors[key] = `${key} must be a boolean`;
        }
        
        // Enum validation
        if (schema.enum && !schema.enum.includes(value)) {
          newErrors[key] = `${key} must be one of: ${schema.enum.join(', ')}`;
        }
        
        // Range validation
        if (schema.type === 'number') {
          if (schema.minimum !== undefined && value < schema.minimum) {
            newErrors[key] = `${key} must be at least ${schema.minimum}`;
          }
          if (schema.maximum !== undefined && value > schema.maximum) {
            newErrors[key] = `${key} must be at most ${schema.maximum}`;
          }
        }
      }
    });
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (validateParams()) {
      onRun(params);
    }
  };

  // Render parameter input
  const renderParameterInput = (key: string, schema: any) => {
    const value = params[key];
    const error = errors[key];
    
    const baseInputClass = `w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors ${
      error 
        ? 'border-red-300 focus:border-red-500 dark:border-red-600' 
        : 'border-gray-300 focus:border-blue-500 dark:border-gray-600'
    } bg-white dark:bg-gray-700 text-gray-900 dark:text-white`;
    
    switch (schema.type) {
      case 'string':
        if (schema.enum) {
          return (
            <select
              value={value || ''}
              onChange={(e) => handleParamChange(key, e.target.value)}
              className={baseInputClass}
            >
              {schema.enum.map((option: string) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          );
        }
        return (
          <input
            type="text"
            value={value || ''}
            onChange={(e) => handleParamChange(key, e.target.value)}
            className={baseInputClass}
            placeholder={schema.description || `Enter ${key}`}
          />
        );
      
      case 'number':
        return (
          <input
            type="number"
            value={value || ''}
            onChange={(e) => handleParamChange(key, Number(e.target.value))}
            className={baseInputClass}
            min={schema.minimum}
            max={schema.maximum}
            step={schema.type === 'integer' ? 1 : 0.01}
            placeholder={schema.description || `Enter ${key}`}
          />
        );
      
      case 'boolean':
        return (
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={value || false}
              onChange={(e) => handleParamChange(key, e.target.checked)}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">
              {schema.description || key}
            </span>
          </label>
        );
      
      case 'array':
        return (
          <input
            type="text"
            value={Array.isArray(value) ? value.join(', ') : ''}
            onChange={(e) => {
              const arrayValue = e.target.value.split(',').map(v => v.trim()).filter(v => v);
              handleParamChange(key, arrayValue);
            }}
            className={baseInputClass}
            placeholder="Enter comma-separated values"
          />
        );
      
      default:
        return (
          <input
            type="text"
            value={value || ''}
            onChange={(e) => handleParamChange(key, e.target.value)}
            className={baseInputClass}
            placeholder={schema.description || `Enter ${key}`}
          />
        );
    }
  };

  if (!isOpen) return null;

  const presets = getPresetConfigs();
  const presetNames = Object.keys(presets);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-3">
            <Settings className="h-6 w-6 text-blue-500" />
            <div>
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Configure {tool.name.replace('benchmark_', '').replace(/_/g, ' ')}
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                {tool.description}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* Content */}
        <form onSubmit={handleSubmit} className="flex flex-col max-h-[calc(90vh-140px)]">
          <div className="flex-1 overflow-y-auto p-6">
            {/* Presets */}
            {presetNames.length > 0 && (
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Quick Presets
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {presetNames.map((presetName) => (
                    <button
                      key={presetName}
                      type="button"
                      onClick={() => handlePresetChange(presetName)}
                      className={`px-3 py-2 text-sm font-medium rounded-lg border transition-colors ${
                        selectedPreset === presetName
                          ? 'bg-blue-50 border-blue-200 text-blue-700 dark:bg-blue-900/20 dark:border-blue-700 dark:text-blue-300'
                          : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-600'
                      }`}
                    >
                      {presetName}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Parameters */}
            <div className="space-y-4">
              {tool.parameters && Object.entries(tool.parameters).map(([key, schema]: [string, any]) => (
                <div key={key}>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    {schema.required && <span className="text-red-500 ml-1">*</span>}
                  </label>
                  
                  {renderParameterInput(key, schema)}
                  
                  {errors[key] && (
                    <div className="flex items-center mt-1 text-sm text-red-600 dark:text-red-400">
                      <AlertCircle className="h-4 w-4 mr-1" />
                      {errors[key]}
                    </div>
                  )}
                  
                  {schema.description && !errors[key] && (
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                      {schema.description}
                    </p>
                  )}
                </div>
              ))}
            </div>

            {/* Info Box */}
            <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
              <div className="flex items-start space-x-3">
                <Info className="h-5 w-5 text-blue-500 mt-0.5" />
                <div className="text-sm text-blue-700 dark:text-blue-300">
                  <p className="font-medium mb-1">Benchmark Execution</p>
                  <p>
                    This will execute the benchmark with real backend processing. 
                    Results will be streamed live and saved to history.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-end space-x-3 p-6 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:bg-gray-700 dark:text-gray-300 dark:border-gray-600 dark:hover:bg-gray-600 dark:focus:ring-offset-gray-800 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className={`px-4 py-2 text-sm font-medium text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800 transition-colors ${
                isLoading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              {isLoading ? (
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Starting...
                </div>
              ) : (
                <div className="flex items-center">
                  <Play className="h-4 w-4 mr-2" />
                  Run Benchmark
                </div>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default RunModal;
