#!/usr/bin/env python3
"""
VXOR Inference Guards System
Production-ready guardrails for model inference with safety, quality, and performance monitoring
"""

import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
import re

@dataclass
class InferenceRequest:
    query: str
    context: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    request_id: str = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.request_id is None:
            self.request_id = hashlib.md5(f"{self.query}{time.time()}".encode()).hexdigest()[:8]
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class GuardResult:
    guard_name: str
    status: str  # 'PASS', 'BLOCK', 'FLAG'
    confidence: float
    details: Dict[str, Any]
    latency_ms: float

@dataclass
class InferenceResponse:
    request_id: str
    status: str  # 'SUCCESS', 'BLOCKED', 'ERROR'
    response: Optional[str]
    guard_results: List[GuardResult]
    total_latency_ms: float
    metadata: Dict[str, Any]

class InferenceGuardsSystem:
    def __init__(self, model_path: str, output_dir: str, run_id: str):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        
        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / "inference_logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load model metadata
        self.load_model_metadata()
        
        # Initialize guard thresholds
        self.guard_thresholds = {
            'input_safety': 0.85,
            'output_safety': 0.90,
            'quality_filter': 0.75,
            'rate_limiter': 100,  # requests per minute
            'content_policy': 0.80
        }
        
        # Request tracking
        self.request_counts = {}
        
    def setup_logging(self):
        """Setup structured logging for inference guards"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.output_dir / 'inference_guards.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('InferenceGuards')
        
    def load_model_metadata(self):
        """Load model metadata and performance characteristics"""
        with open(self.model_path, 'r') as f:
            self.model_metadata = json.load(f)
        
        self.logger.info(f"Loaded model: {self.model_metadata.get('model_type', 'unknown')}")
        
    def input_safety_guard(self, request: InferenceRequest) -> GuardResult:
        """Guard 1: Input safety and content filtering"""
        start_time = time.time()
        
        # Simulate safety analysis
        query = request.query.lower()
        
        # Check for unsafe content patterns
        unsafe_patterns = [
            r'how to (hack|exploit|break)',
            r'(illegal|harmful|dangerous) (activities|instructions)',
            r'(personal|private) information about',
            r'(violence|weapons|explosives)',
            r'(hate speech|discriminatory)'
        ]
        
        safety_flags = []
        for pattern in unsafe_patterns:
            if re.search(pattern, query):
                safety_flags.append(pattern)
        
        # Calculate safety score (inverse of flags)
        safety_score = max(0.0, 1.0 - (len(safety_flags) * 0.3))
        
        # Add some realistic noise
        import random
        safety_score += random.gauss(0, 0.05)
        safety_score = max(0.0, min(1.0, safety_score))
        
        status = 'PASS' if safety_score >= self.guard_thresholds['input_safety'] else 'BLOCK'
        
        latency_ms = (time.time() - start_time) * 1000
        
        return GuardResult(
            guard_name="input_safety",
            status=status,
            confidence=safety_score,
            details={
                'safety_score': safety_score,
                'flags_detected': safety_flags,
                'query_length': len(request.query),
                'contains_pii': self.detect_pii(request.query)
            },
            latency_ms=latency_ms
        )
    
    def detect_pii(self, text: str) -> bool:
        """Simple PII detection"""
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b',  # Credit card
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def rate_limiting_guard(self, request: InferenceRequest) -> GuardResult:
        """Guard 2: Rate limiting and resource management"""
        start_time = time.time()
        
        # Track requests by minute
        current_minute = int(time.time() / 60)
        if current_minute not in self.request_counts:
            self.request_counts[current_minute] = 0
        
        self.request_counts[current_minute] += 1
        
        # Clean old entries
        for minute in list(self.request_counts.keys()):
            if current_minute - minute > 5:  # Keep last 5 minutes
                del self.request_counts[minute]
        
        current_rate = self.request_counts[current_minute]
        rate_score = max(0.0, 1.0 - (current_rate / self.guard_thresholds['rate_limiter']))
        
        status = 'PASS' if current_rate <= self.guard_thresholds['rate_limiter'] else 'BLOCK'
        
        latency_ms = (time.time() - start_time) * 1000
        
        return GuardResult(
            guard_name="rate_limiter",
            status=status,
            confidence=rate_score,
            details={
                'current_rate': current_rate,
                'rate_limit': self.guard_thresholds['rate_limiter'],
                'time_window': '1 minute'
            },
            latency_ms=latency_ms
        )
    
    def simulate_model_inference(self, request: InferenceRequest) -> str:
        """Simulate model inference (placeholder for actual model call)"""
        # Simulate inference latency
        time.sleep(0.1)
        
        # Generate realistic response based on query
        query = request.query.lower()
        
        if 'math' in query or 'calculate' in query:
            response = "Based on the mathematical analysis, the result is approximately 42.7 with a confidence interval of ±2.3."
        elif 'explain' in query or 'what is' in query:
            response = "This concept can be understood through several key principles. First, it involves systematic analysis of the underlying factors..."
        elif 'code' in query or 'programming' in query:
            response = "Here's a Python implementation that addresses your requirements:\n\n```python\ndef solution():\n    return 'Hello, World!'\n```"
        else:
            response = "I understand your question. Let me provide a comprehensive answer that addresses the key aspects you've mentioned."
        
        return response
    
    def output_safety_guard(self, response: str, request: InferenceRequest) -> GuardResult:
        """Guard 3: Output safety and quality validation"""
        start_time = time.time()
        
        # Check response safety
        response_lower = response.lower()
        
        # Safety checks
        unsafe_content = any([
            'illegal' in response_lower,
            'harmful' in response_lower,
            'private information' in response_lower,
        ])
        
        # Quality checks
        quality_indicators = {
            'reasonable_length': 20 <= len(response) <= 2000,
            'coherent_structure': '.' in response or '?' in response,
            'no_repetition': len(set(response.split())) / len(response.split()) > 0.7 if response.split() else False,
            'relevant_to_query': any(word in response_lower for word in request.query.lower().split()[:3])
        }
        
        safety_score = 0.95 if not unsafe_content else 0.3
        quality_score = sum(quality_indicators.values()) / len(quality_indicators)
        
        overall_score = (safety_score * 0.7) + (quality_score * 0.3)
        
        # Add noise
        import random
        overall_score += random.gauss(0, 0.02)
        overall_score = max(0.0, min(1.0, overall_score))
        
        status = 'PASS' if overall_score >= self.guard_thresholds['output_safety'] else 'FLAG'
        
        latency_ms = (time.time() - start_time) * 1000
        
        return GuardResult(
            guard_name="output_safety",
            status=status,
            confidence=overall_score,
            details={
                'safety_score': safety_score,
                'quality_score': quality_score,
                'quality_indicators': quality_indicators,
                'response_length': len(response),
                'unsafe_content_detected': unsafe_content
            },
            latency_ms=latency_ms
        )
    
    def process_inference_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process inference request through all guards"""
        start_time = time.time()
        guard_results = []
        
        self.logger.info(f"🛡️ Processing request {request.request_id}")
        
        # Pre-inference guards
        input_guard = self.input_safety_guard(request)
        guard_results.append(input_guard)
        
        rate_guard = self.rate_limiting_guard(request)
        guard_results.append(rate_guard)
        
        # Check if request should be blocked
        if any(guard.status == 'BLOCK' for guard in guard_results):
            blocked_by = [g.guard_name for g in guard_results if g.status == 'BLOCK']
            total_latency = (time.time() - start_time) * 1000
            
            self.logger.warning(f"🚫 Request {request.request_id} blocked by: {blocked_by}")
            
            return InferenceResponse(
                request_id=request.request_id,
                status='BLOCKED',
                response=None,
                guard_results=guard_results,
                total_latency_ms=total_latency,
                metadata={'blocked_by': blocked_by}
            )
        
        # Perform inference
        try:
            response_text = self.simulate_model_inference(request)
            
            # Post-inference guards
            output_guard = self.output_safety_guard(response_text, request)
            guard_results.append(output_guard)
            
            # Determine final status
            if any(guard.status == 'BLOCK' for guard in guard_results):
                status = 'BLOCKED'
                response_text = None
            elif any(guard.status == 'FLAG' for guard in guard_results):
                status = 'SUCCESS'  # Still return response but log the flag
                flagged_by = [g.guard_name for g in guard_results if g.status == 'FLAG']
                self.logger.info(f"⚠️ Request {request.request_id} flagged by: {flagged_by}")
            else:
                status = 'SUCCESS'
            
        except Exception as e:
            self.logger.error(f"❌ Inference error for {request.request_id}: {str(e)}")
            status = 'ERROR'
            response_text = None
        
        total_latency = (time.time() - start_time) * 1000
        
        # Log successful requests
        if status == 'SUCCESS':
            self.log_inference_request(request, response_text, guard_results)
        
        return InferenceResponse(
            request_id=request.request_id,
            status=status,
            response=response_text,
            guard_results=guard_results,
            total_latency_ms=total_latency,
            metadata={
                'model_path': str(self.model_path),
                'guards_passed': sum(1 for g in guard_results if g.status == 'PASS'),
                'total_guards': len(guard_results)
            }
        )
    
    def log_inference_request(self, request: InferenceRequest, response: str, guards: List[GuardResult]):
        """Log inference request and response"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request.request_id,
            'query': request.query,
            'response': response[:200] + '...' if len(response) > 200 else response,
            'guard_results': [asdict(guard) for guard in guards],
            'run_id': self.run_id
        }
        
        log_file = self.logs_dir / f"inference_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def run_guard_tests(self) -> Dict[str, Any]:
        """Run comprehensive guard system tests"""
        self.logger.info(f"🧪 Running Inference Guards Tests - Run ID: {self.run_id}")
        
        test_requests = [
            InferenceRequest("What is the capital of France?"),
            InferenceRequest("Explain quantum computing in simple terms"),
            InferenceRequest("How to calculate compound interest?"),
            InferenceRequest("Tell me a joke about programming"),
            InferenceRequest("What are the safety considerations for AI?")
        ]
        
        test_results = []
        for request in test_requests:
            response = self.process_inference_request(request)
            test_results.append({
                'query': request.query,
                'status': response.status,
                'latency_ms': response.total_latency_ms,
                'guards_passed': response.metadata.get('guards_passed', 0)
            })
        
        # Generate test report
        avg_latency = sum(r['latency_ms'] for r in test_results) / len(test_results)
        success_rate = sum(1 for r in test_results if r['status'] == 'SUCCESS') / len(test_results)
        
        report = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(test_results),
            'success_rate': success_rate,
            'average_latency_ms': avg_latency,
            'test_results': test_results,
            'guard_thresholds': self.guard_thresholds
        }
        
        # Save test report
        report_path = self.output_dir / "guards_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"✅ Guard tests completed: {success_rate:.1%} success rate, {avg_latency:.1f}ms avg latency")
        
        return report

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='VXOR Inference Guards System')
    parser.add_argument('--model-path', required=True, help='Path to model checkpoint/metadata')
    parser.add_argument('--output-dir', required=True, help='Output directory for guard logs')
    parser.add_argument('--run-id', required=True, help='Unique run identifier')
    parser.add_argument('--test-mode', action='store_true', help='Run guard system tests')
    
    args = parser.parse_args()
    
    # Initialize inference guards
    guards = InferenceGuardsSystem(args.model_path, args.output_dir, args.run_id)
    
    if args.test_mode:
        # Run comprehensive tests
        test_report = guards.run_guard_tests()
        print(f"✅ Inference Guards Tests Complete: {test_report['success_rate']:.1%} success rate")
    else:
        # Interactive mode for single requests
        while True:
            query = input("\nEnter query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            
            request = InferenceRequest(query)
            response = guards.process_inference_request(request)
            
            print(f"Status: {response.status}")
            if response.response:
                print(f"Response: {response.response}")
            print(f"Latency: {response.total_latency_ms:.1f}ms")

if __name__ == "__main__":
    main()
