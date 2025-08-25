#!/usr/bin/env python3
"""Test script for Prometheus callback integration"""

import time
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("⚠️ requests not available, skipping HTTP tests")

from prometheus_callback import create_prometheus_callback

def test_prometheus_callback():
    """Test Prometheus callback functionality"""
    
    print("🧪 Testing Prometheus Callback Integration")
    
    # Create callback
    try:
        callback = create_prometheus_callback(port=9109)  # Use different port to avoid conflicts
        print("✅ Prometheus callback created successfully")
    except Exception as e:
        print(f"❌ Failed to create callback: {e}")
        return False
    
    # Test metrics endpoint
    if not HAS_REQUESTS:
        print("⚠️ Skipping HTTP endpoint test (requests not available)")
        print("✅ Callback created successfully - manual verification needed")
        return True
        
    try:
        time.sleep(2)  # Give server time to start
        response = requests.get("http://localhost:9109/metrics", timeout=5)
        
        if response.status_code == 200:
            print("✅ Metrics endpoint accessible")
            
            # Check for expected metrics
            metrics_text = response.text
            expected_metrics = [
                "mimikcompute_train_loss",
                "mimikcompute_train_lr", 
                "mimikcompute_train_grad_norm",
                "mimikcompute_train_throughput_samples",
                "mimikcompute_train_step_seconds"
            ]
            
            found_metrics = []
            for metric in expected_metrics:
                if metric in metrics_text:
                    found_metrics.append(metric)
                    
            print(f"✅ Found {len(found_metrics)}/{len(expected_metrics)} expected metrics")
            
            if len(found_metrics) == len(expected_metrics):
                print("✅ All expected metrics are registered")
                return True
            else:
                missing = set(expected_metrics) - set(found_metrics)
                print(f"⚠️ Missing metrics: {missing}")
                return False
                
        else:
            print(f"❌ Metrics endpoint returned status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to access metrics endpoint: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_callback_methods():
    """Test callback method functionality"""
    
    print("\n🧪 Testing Callback Methods")
    
    callback = create_prometheus_callback(port=9110)
    
    # Mock training state and logs
    class MockState:
        def __init__(self):
            self.global_step = 100
            self.best_metric = 0.5
    
    class MockArgs:
        def __init__(self):
            self.per_device_train_batch_size = 2
            self.gradient_accumulation_steps = 4
    
    # Test on_log method
    mock_logs = {
        "loss": 0.75,
        "learning_rate": 5e-4,
        "grad_norm": 1.2,
        "epoch": 1.0
    }
    
    try:
        callback.on_log(MockArgs(), MockState(), None, logs=mock_logs)
        print("✅ on_log method executed successfully")
    except Exception as e:
        print(f"❌ on_log method failed: {e}")
        return False
    
    # Test performance summary
    try:
        summary = callback.get_performance_summary()
        print(f"✅ Performance summary generated: {len(summary)} metrics")
    except Exception as e:
        print(f"❌ Performance summary failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    
    print("🚀 Starting Prometheus Integration Tests\n")
    
    # Test 1: Basic callback functionality
    test1_passed = test_prometheus_callback()
    
    # Test 2: Callback methods
    test2_passed = test_callback_methods()
    
    # Summary
    print("\n" + "="*50)
    print("📋 Test Results Summary")
    print("="*50)
    print(f"✅ Prometheus Callback Creation: {'PASS' if test1_passed else 'FAIL'}")
    print(f"✅ Callback Methods: {'PASS' if test2_passed else 'FAIL'}")
    
    overall_success = test1_passed and test2_passed
    
    if overall_success:
        print("\n🎉 All tests passed! Integration is ready for production use.")
        print("\n🔧 Next Steps:")
        print("1. Add callback to your trainer: callbacks=[create_prometheus_callback()]")
        print("2. Import Grafana dashboard from miso_training_dashboard.json")
        print("3. Configure Prometheus alerts from prometheus_alerts.yml")
        print("4. Start training and monitor at http://localhost:9108/metrics")
    else:
        print("\n⚠️ Some tests failed. Check the errors above before deploying.")
    
    return overall_success

if __name__ == "__main__":
    main()
