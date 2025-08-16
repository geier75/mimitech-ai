#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR Benchmark Backend - Vollständig funktionales Backend
Führt echte Benchmarks aus und liefert Live-Ergebnisse an das Dashboard

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import json
import time
import logging
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import psutil

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - VXOR.Backend - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Globale Variablen
BACKEND_PORT = 5152
RESULTS_DIR = Path(__file__).parent / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Benchmark-Status
benchmark_status = {
    "matrix": {"running": False, "progress": 0, "results": []},
    "quantum": {"running": False, "progress": 0, "results": []},
    "system": {"running": False, "progress": 0, "results": []}
}

class BenchmarkBackendHandler(BaseHTTPRequestHandler):
    """Handler für Benchmark-Backend API"""
    
    def do_GET(self):
        """GET-Anfragen behandeln"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # CORS-Header setzen
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
        if path == "/api/status":
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(benchmark_status).encode())
            
        elif path == "/api/system-info":
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            system_info = get_system_info()
            self.wfile.write(json.dumps(system_info).encode())
            
        elif path == "/api/results":
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            results = load_all_results()
            self.wfile.write(json.dumps(results).encode())
            
        else:
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"VXOR Benchmark Backend Active")
    
    def do_POST(self):
        """POST-Anfragen behandeln"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # CORS-Header setzen
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        if path == "/api/start-matrix":
            if not benchmark_status["matrix"]["running"]:
                threading.Thread(target=run_matrix_benchmarks, daemon=True).start()
                response = {"status": "started", "message": "Matrix Benchmarks gestartet"}
            else:
                response = {"status": "running", "message": "Matrix Benchmarks laufen bereits"}
            self.wfile.write(json.dumps(response).encode())
            
        elif path == "/api/start-quantum":
            if not benchmark_status["quantum"]["running"]:
                threading.Thread(target=run_quantum_benchmarks, daemon=True).start()
                response = {"status": "started", "message": "Quantum Benchmarks gestartet"}
            else:
                response = {"status": "running", "message": "Quantum Benchmarks laufen bereits"}
            self.wfile.write(json.dumps(response).encode())
            
        elif path == "/api/start-all":
            start_all_benchmarks()
            response = {"status": "started", "message": "Alle Benchmarks gestartet"}
            self.wfile.write(json.dumps(response).encode())
            
        else:
            response = {"status": "error", "message": "Unbekannter Endpoint"}
            self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        """OPTIONS-Anfragen für CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Logging überschreiben"""
        logger.info("%s - %s", self.address_string(), format % args)

def get_system_info():
    """System-Informationen sammeln"""
    try:
        # Hardware-Info
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        
        # MLX/PyTorch Detection
        mlx_available = False
        pytorch_available = False
        mps_available = False
        
        try:
            import mlx.core as mx
            mlx_available = True
        except ImportError:
            pass
            
        try:
            import torch
            pytorch_available = True
            mps_available = torch.backends.mps.is_available()
        except ImportError:
            pass
        
        return {
            "timestamp": datetime.now().isoformat(),
            "hardware": {
                "cpu_cores": cpu_count,
                "cpu_freq_mhz": cpu_freq.current if cpu_freq else 0,
                "memory_gb": round(memory.total / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_usage_percent": memory.percent
            },
            "acceleration": {
                "mlx_available": mlx_available,
                "pytorch_available": pytorch_available,
                "mps_available": mps_available
            },
            "platform": {
                "system": os.uname().sysname,
                "machine": os.uname().machine,
                "python_version": sys.version
            }
        }
    except Exception as e:
        logger.error(f"Fehler beim Sammeln der System-Info: {e}")
        return {"error": str(e)}

def run_matrix_benchmarks():
    """Matrix-Benchmarks ausführen"""
    logger.info("🚀 Matrix Benchmarks gestartet")
    benchmark_status["matrix"]["running"] = True
    benchmark_status["matrix"]["progress"] = 0
    benchmark_status["matrix"]["results"] = []
    
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    total_tests = len(sizes) * 2  # Matmul + SVD
    
    try:
        for i, size in enumerate(sizes):
            # Matrix Multiplication Test
            logger.info(f"Matrix Multiplication Test: {size}x{size}")
            result = run_matrix_multiplication_test(size)
            benchmark_status["matrix"]["results"].append(result)
            benchmark_status["matrix"]["progress"] = int((i * 2 + 1) / total_tests * 100)
            
            # SVD Test
            logger.info(f"SVD Test: {size}x{size}")
            result = run_svd_test(size)
            benchmark_status["matrix"]["results"].append(result)
            benchmark_status["matrix"]["progress"] = int((i * 2 + 2) / total_tests * 100)
            
            time.sleep(0.1)  # Kurze Pause für UI-Updates
        
        # Ergebnisse speichern
        save_results("matrix", benchmark_status["matrix"]["results"])
        logger.info("✅ Matrix Benchmarks abgeschlossen")
        
    except Exception as e:
        logger.error(f"Fehler bei Matrix Benchmarks: {e}")
    finally:
        benchmark_status["matrix"]["running"] = False

def run_quantum_benchmarks():
    """Quantum-Benchmarks ausführen"""
    logger.info("🚀 Quantum Benchmarks gestartet")
    benchmark_status["quantum"]["running"] = True
    benchmark_status["quantum"]["progress"] = 0
    benchmark_status["quantum"]["results"] = []
    
    qubits = [2, 4, 6, 8, 10, 12, 14, 16]
    
    try:
        for i, n_qubits in enumerate(qubits):
            logger.info(f"Quantum Simulation Test: {n_qubits} Qubits")
            result = run_quantum_simulation_test(n_qubits)
            benchmark_status["quantum"]["results"].append(result)
            benchmark_status["quantum"]["progress"] = int((i + 1) / len(qubits) * 100)
            time.sleep(0.1)
        
        # Ergebnisse speichern
        save_results("quantum", benchmark_status["quantum"]["results"])
        logger.info("✅ Quantum Benchmarks abgeschlossen")
        
    except Exception as e:
        logger.error(f"Fehler bei Quantum Benchmarks: {e}")
    finally:
        benchmark_status["quantum"]["running"] = False

def run_matrix_multiplication_test(size):
    """Matrix-Multiplikation Test"""
    start_time = time.time()
    
    # Test-Matrizen erstellen
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    # Benchmark
    times = []
    for _ in range(3):
        t_start = time.perf_counter()
        C = np.dot(A, B)
        t_end = time.perf_counter()
        times.append(t_end - t_start)
    
    avg_time = np.mean(times)
    flops = 2 * size**3  # Floating point operations
    gflops = flops / avg_time / 1e9
    
    return {
        "test": "matrix_multiplication",
        "size": size,
        "avg_time_s": avg_time,
        "gflops": gflops,
        "timestamp": datetime.now().isoformat()
    }

def run_svd_test(size):
    """SVD Test"""
    start_time = time.time()
    
    # Test-Matrix erstellen
    A = np.random.randn(size, size).astype(np.float32)
    
    # SVD Benchmark
    t_start = time.perf_counter()
    U, s, Vt = np.linalg.svd(A)
    t_end = time.perf_counter()
    
    return {
        "test": "svd",
        "size": size,
        "time_s": t_end - t_start,
        "timestamp": datetime.now().isoformat()
    }

def run_quantum_simulation_test(n_qubits):
    """Quantum Simulation Test"""
    start_time = time.time()
    
    # Quantum State Simulation (vereinfacht)
    state_size = 2**n_qubits
    
    # Zufälligen Quantenzustand erstellen
    t_start = time.perf_counter()
    state = np.random.randn(state_size) + 1j * np.random.randn(state_size)
    state = state / np.linalg.norm(state)
    
    # Quantum Gate Operation simulieren
    gate = np.random.randn(state_size, state_size) + 1j * np.random.randn(state_size, state_size)
    new_state = np.dot(gate, state)
    new_state = new_state / np.linalg.norm(new_state)
    
    t_end = time.perf_counter()
    
    return {
        "test": "quantum_simulation",
        "qubits": n_qubits,
        "state_size": state_size,
        "time_s": t_end - t_start,
        "timestamp": datetime.now().isoformat()
    }

def start_all_benchmarks():
    """Alle Benchmarks starten"""
    if not benchmark_status["matrix"]["running"]:
        threading.Thread(target=run_matrix_benchmarks, daemon=True).start()
    if not benchmark_status["quantum"]["running"]:
        threading.Thread(target=run_quantum_benchmarks, daemon=True).start()

def save_results(category, results):
    """Ergebnisse speichern"""
    filename = RESULTS_DIR / f"{category}_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Ergebnisse gespeichert: {filename}")

def load_all_results():
    """Alle gespeicherten Ergebnisse laden"""
    results = {"matrix": [], "quantum": [], "system": []}
    
    for json_file in RESULTS_DIR.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            if "matrix" in json_file.name:
                results["matrix"].extend(data if isinstance(data, list) else [data])
            elif "quantum" in json_file.name:
                results["quantum"].extend(data if isinstance(data, list) else [data])
                
        except Exception as e:
            logger.error(f"Fehler beim Laden von {json_file}: {e}")
    
    return results

def run_server():
    """Backend-Server starten"""
    try:
        server = HTTPServer(('127.0.0.1', BACKEND_PORT), BenchmarkBackendHandler)
        logger.info(f"🚀 VXOR Benchmark Backend gestartet auf Port {BACKEND_PORT}")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Fehler beim Starten des Backend-Servers: {e}")

if __name__ == "__main__":
    logger.info("🔥 VXOR Benchmark Backend wird gestartet...")
    run_server()
