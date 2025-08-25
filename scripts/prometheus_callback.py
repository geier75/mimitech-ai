#!/usr/bin/env python3
"""Prometheus Telemetry Callback for Hugging Face Trainer"""

import time
import logging
from typing import Optional
from prometheus_client import Gauge, Histogram, Counter, start_http_server
from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)

class PrometheusCallback(TrainerCallback):
    """Drop-in Prometheus callback for HF Trainer with comprehensive metrics"""
    
    def __init__(self, port: int = 9108, prefix: str = "mimikcompute"):
        """
        Initialize Prometheus metrics server and gauges
        
        Args:
            port: HTTP server port for Prometheus scraping
            prefix: Metric name prefix
        """
        try:
            start_http_server(port)
            logger.info(f"🔥 Prometheus server started on port {port}")
        except OSError as e:
            logger.warning(f"Prometheus server failed to start on port {port}: {e}")
            
        # Core training metrics
        self.g_loss = Gauge(f"{prefix}_train_loss", "Current training loss")
        self.g_lr = Gauge(f"{prefix}_train_lr", "Current learning rate")
        self.g_gn = Gauge(f"{prefix}_train_grad_norm", "Gradient norm")
        self.g_tps = Gauge(f"{prefix}_train_throughput_samples", "Training throughput (samples/sec)")
        self.g_epoch = Gauge(f"{prefix}_train_epoch", "Current training epoch")
        self.g_step = Gauge(f"{prefix}_train_step", "Global training step")
        
        # Performance histograms
        self.h_step = Histogram(
            f"{prefix}_train_step_seconds", 
            "Step execution time (seconds)",
            buckets=(0.5, 1, 2, 3, 5, 10, 20, float("inf"))
        )
        
        # Evaluation metrics
        self.g_eval_loss = Gauge(f"{prefix}_eval_loss", "Evaluation loss")
        self.g_eval_ppl = Gauge(f"{prefix}_eval_perplexity", "Evaluation perplexity")
        
        # System health counters
        self.c_nan_loss = Counter(f"{prefix}_nan_loss_total", "Total NaN/Inf loss occurrences")
        self.c_checkpoints = Counter(f"{prefix}_checkpoints_total", "Total checkpoints saved")
        
        # Stability metrics
        self.g_memory_usage = Gauge(f"{prefix}_memory_usage_mb", "Memory usage in MB")
        self.g_gpu_util = Gauge(f"{prefix}_gpu_utilization", "GPU utilization percentage")
        
        # Internal state for timing calculations
        self._last_time = None
        self._last_step = None
        self._step_times = []
        self._median_window = 100  # Steps for median calculation
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize training start metrics"""
        logger.info("🚀 Training started - Prometheus monitoring active")
        self.g_step.set(0)
        self._last_time = time.time()
        self._last_step = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Process training logs and update metrics"""
        if not logs:
            return
            
        current_time = time.time()
        current_step = state.global_step
        
        # Core metrics from logs
        if "loss" in logs:
            loss_val = float(logs["loss"])
            self.g_loss.set(loss_val)
            
            # Check for NaN/Inf
            if not (0 <= loss_val < float('inf')):
                self.c_nan_loss.inc()
                logger.warning(f"⚠️ Abnormal loss detected: {loss_val}")
                
        if "learning_rate" in logs:
            self.g_lr.set(float(logs["learning_rate"]))
            
        if "grad_norm" in logs:
            self.g_gn.set(float(logs["grad_norm"]))
            
        if "epoch" in logs:
            self.g_epoch.set(float(logs["epoch"]))
            
        # Update global step
        self.g_step.set(current_step)
        
        # Calculate step timing and throughput
        if self._last_time and self._last_step is not None:
            time_diff = current_time - self._last_time
            step_diff = current_step - self._last_step
            
            if step_diff > 0 and time_diff > 0:
                step_time = time_diff / step_diff
                self.h_step.observe(step_time)
                self._step_times.append(step_time)
                
                # Keep only recent step times for median calculation
                if len(self._step_times) > self._median_window:
                    self._step_times = self._step_times[-self._median_window:]
                
                # Calculate throughput if batch info available
                if "train_samples_per_second" in logs:
                    self.g_tps.set(float(logs["train_samples_per_second"]))
                elif hasattr(args, 'per_device_train_batch_size'):
                    # Estimate throughput
                    batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
                    throughput = batch_size / step_time
                    self.g_tps.set(throughput)
        
        # Evaluation metrics
        if "eval_loss" in logs:
            eval_loss = float(logs["eval_loss"])
            self.g_eval_loss.set(eval_loss)
            
            # Calculate perplexity if possible
            try:
                import math
                perplexity = math.exp(eval_loss)
                self.g_eval_ppl.set(perplexity)
            except (OverflowError, ValueError):
                pass
        
        # Memory usage (if torch available)
        try:
            import torch
            if torch.backends.mps.is_available():
                # For MPS, we can't get exact memory usage easily
                pass
            elif torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                self.g_memory_usage.set(memory_mb)
        except ImportError:
            pass
            
        # Update timing state
        self._last_time = current_time
        self._last_step = current_step
        
    def on_save(self, args, state, control, **kwargs):
        """Track checkpoint saves"""
        self.c_checkpoints.inc()
        logger.info(f"💾 Checkpoint saved at step {state.global_step}")
        
    def on_evaluate(self, args, state, control, **kwargs):
        """Track evaluation events"""
        logger.info(f"📊 Evaluation at step {state.global_step}")
        
    def on_train_end(self, args, state, control, **kwargs):
        """Training completion metrics"""
        logger.info("✅ Training completed - final metrics recorded")
        
    def get_performance_summary(self) -> dict:
        """Get current performance summary for alerting"""
        if not self._step_times:
            return {}
            
        import statistics
        recent_times = self._step_times[-20:]  # Last 20 steps
        
        return {
            "median_step_time": statistics.median(self._step_times),
            "recent_median": statistics.median(recent_times) if recent_times else 0,
            "p95_step_time": sorted(self._step_times)[int(0.95 * len(self._step_times))] if self._step_times else 0,
            "total_steps_recorded": len(self._step_times)
        }


# Convenience function for easy integration
def create_prometheus_callback(port: int = 9108, prefix: str = "mimikcompute") -> PrometheusCallback:
    """Create and return a configured Prometheus callback"""
    return PrometheusCallback(port=port, prefix=prefix)


if __name__ == "__main__":
    # Test the callback
    callback = create_prometheus_callback()
    print("Prometheus callback created successfully")
    print(f"Metrics available at: http://localhost:9108/metrics")
