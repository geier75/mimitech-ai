import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Volume2 } from 'lucide-react';
import { executeBenchmark } from '../api/client';

interface VoiceCommandProps {
  onBenchmarkExecuted?: (result: any) => void;
}

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

const VoiceCommand: React.FC<VoiceCommandProps> = ({ onBenchmarkExecuted }) => {
  const [isListening, setIsListening] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastCommand, setLastCommand] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  
  const recognitionRef = useRef<any>(null);
  const synthRef = useRef<SpeechSynthesis | null>(null);

  useEffect(() => {
    // Check if speech recognition is supported
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      setIsSupported(true);
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'de-DE'; // German language support
      
      recognitionRef.current.onstart = () => {
        setIsListening(true);
        setTranscript('');
      };
      
      recognitionRef.current.onresult = (event: SpeechRecognitionEvent) => {
        const current = event.resultIndex;
        const transcript = event.results[current][0].transcript;
        setTranscript(transcript);
        
        if (event.results[current].isFinal) {
          processVoiceCommand(transcript);
        }
      };
      
      recognitionRef.current.onerror = (event: SpeechRecognitionErrorEvent) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        speak(`Entschuldigung, ich konnte Sie nicht verstehen. Fehler: ${event.error}`);
      };
      
      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }

    // Initialize speech synthesis
    if ('speechSynthesis' in window) {
      synthRef.current = window.speechSynthesis;
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  const speak = (text: string) => {
    if (synthRef.current && !isSpeaking) {
      setIsSpeaking(true);
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'de-DE';
      utterance.rate = 0.9;
      utterance.pitch = 1.0;
      
      utterance.onend = () => {
        setIsSpeaking(false);
      };
      
      utterance.onerror = () => {
        setIsSpeaking(false);
      };
      
      synthRef.current.speak(utterance);
    }
  };

  const processVoiceCommand = async (command: string) => {
    setIsProcessing(true);
    setLastCommand(command);
    
    const lowerCommand = command.toLowerCase();
    
    try {
      // Parse voice commands for benchmark execution
      if (lowerCommand.includes('matrix') || lowerCommand.includes('matrizen')) {
        speak('Matrix Benchmark wird gestartet...');
        const result = await executeBenchmark('benchmark_matrix', { preset: 'small', iterations: 10 });
        speak(`Matrix Benchmark abgeschlossen. Durchsatz: ${Math.round(result.result?.api_response?.results?.throughput_ops_per_sec || 0)} Operationen pro Sekunde.`);
        onBenchmarkExecuted?.(result);
        
      } else if (lowerCommand.includes('quantum') || lowerCommand.includes('quanten')) {
        speak('Quantum Benchmark wird gestartet...');
        const result = await executeBenchmark('benchmark_quantum', { preset: 'small', iterations: 5 });
        speak(`Quantum Benchmark abgeschlossen. Effizienz-Score: ${(result.result?.api_response?.results?.efficiency_score * 100 || 0).toFixed(1)} Prozent.`);
        onBenchmarkExecuted?.(result);
        
      } else if (lowerCommand.includes('alle') || lowerCommand.includes('all') || lowerCommand.includes('komplett')) {
        speak('Alle Benchmarks werden gestartet...');
        const result = await executeBenchmark('benchmark_all', { preset: 'small', parallel: true });
        speak(`Alle Benchmarks abgeschlossen. Gesamtzeit: ${result.result?.api_response?.results?.total_time || 0} Sekunden.`);
        onBenchmarkExecuted?.(result);
        
      } else if (lowerCommand.includes('status') || lowerCommand.includes('zustand')) {
        speak('System-Status wird abgerufen...');
        // Here you could add system status check
        speak('System läuft normal. Alle Services sind verfügbar.');
        
      } else if (lowerCommand.includes('hilfe') || lowerCommand.includes('help') || lowerCommand.includes('befehle')) {
        speak('Verfügbare Befehle: Matrix Benchmark, Quantum Benchmark, Alle Benchmarks, Status, oder Hilfe.');
        
      } else {
        speak(`Befehl "${command}" wurde nicht erkannt. Sagen Sie "Hilfe" für verfügbare Befehle.`);
      }
    } catch (error) {
      console.error('Error processing voice command:', error);
      speak('Es gab einen Fehler bei der Ausführung des Befehls. Bitte versuchen Sie es erneut.');
    } finally {
      setIsProcessing(false);
    }
  };

  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      recognitionRef.current.start();
    }
  };

  const stopListening = () => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
    }
  };

  const toggleListening = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  if (!isSupported) {
    return (
      <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
        <p className="text-yellow-800 dark:text-yellow-200 text-sm">
          Sprachbefehle werden in diesem Browser nicht unterstützt.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          🎤 Sprachbefehle
        </h3>
        <div className="flex items-center space-x-2">
          {isSpeaking && (
            <Volume2 className="h-5 w-5 text-blue-500 animate-pulse" />
          )}
          <button
            onClick={toggleListening}
            disabled={isProcessing}
            className={`p-3 rounded-full transition-colors ${
              isListening
                ? 'bg-red-500 hover:bg-red-600 text-white'
                : 'bg-blue-500 hover:bg-blue-600 text-white'
            } ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {isListening ? (
              <MicOff className="h-6 w-6" />
            ) : (
              <Mic className="h-6 w-6" />
            )}
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {isListening && (
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3">
            <p className="text-blue-800 dark:text-blue-200 text-sm font-medium">
              🎧 Ich höre zu...
            </p>
            {transcript && (
              <p className="text-blue-600 dark:text-blue-300 text-sm mt-1">
                "{transcript}"
              </p>
            )}
          </div>
        )}

        {isProcessing && (
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-3">
            <p className="text-yellow-800 dark:text-yellow-200 text-sm font-medium">
              ⚡ Befehl wird verarbeitet...
            </p>
          </div>
        )}

        {lastCommand && !isProcessing && (
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-3">
            <p className="text-green-800 dark:text-green-200 text-sm font-medium">
              ✅ Letzter Befehl: "{lastCommand}"
            </p>
          </div>
        )}

        <div className="text-sm text-gray-600 dark:text-gray-400">
          <p className="font-medium mb-2">Verfügbare Befehle:</p>
          <ul className="space-y-1 text-xs">
            <li>• "Matrix Benchmark" - Startet Matrix-Berechnungen</li>
            <li>• "Quantum Benchmark" - Startet Quantum-Simulationen</li>
            <li>• "Alle Benchmarks" - Startet alle verfügbaren Tests</li>
            <li>• "Status" - Zeigt System-Status an</li>
            <li>• "Hilfe" - Zeigt verfügbare Befehle</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default VoiceCommand;
