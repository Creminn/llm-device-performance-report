#!/usr/bin/env python3
"""
Test script for enhanced Ollama performance testing system.
"""

import time
from app.metrics_collector import MetricsCollector, PerformanceTest

def test_metrics_collector():
    """Test the metrics collection system."""
    print("Testing Metrics Collector...")
    
    # Initialize collector
    collector = MetricsCollector(sampling_interval=0.5)
    
    # Start collection
    collector.start_collection()
    print("Metrics collection started...")
    
    # Let it run for a few seconds
    time.sleep(3)
    
    # Stop collection
    metrics = collector.stop_collection()
    
    print(f"Collected {len(metrics)} samples")
    if metrics:
        last_sample = metrics[-1]
        print("Latest sample:")
        print(f"  CPU: {last_sample['cpu']['overall_percent']:.1f}%")
        print(f"  Memory: {last_sample['memory']['virtual']['percentage']:.1f}%")
        print(f"  Memory Used: {last_sample['memory']['virtual']['used_mb']:.1f} MB")
        
        if 'gpu' in last_sample and last_sample['gpu']:
            for i, gpu in enumerate(last_sample['gpu']):
                print(f"  GPU {i}: {gpu['utilization']['compute_percent']}% compute, "
                      f"{gpu['memory']['utilization_percent']:.1f}% memory")
        else:
            print("  GPU: Not available")
    
    return True

def test_performance_test():
    """Test the performance test system (without actual Ollama calls)."""
    print("\nTesting Performance Test System...")
    
    perf_test = PerformanceTest(sampling_interval=1.0)
    print("Performance test system initialized successfully")
    
    return True

if __name__ == "__main__":
    print("Enhanced Ollama Performance Testing System - Test Suite")
    print("=" * 60)
    
    try:
        # Test metrics collector
        if test_metrics_collector():
            print("✅ Metrics Collector: PASSED")
        else:
            print("❌ Metrics Collector: FAILED")
            
        # Test performance test system
        if test_performance_test():
            print("✅ Performance Test System: PASSED")
        else:
            print("❌ Performance Test System: FAILED")
            
        print("\n🎉 All tests completed!")
        print("\nTo run the enhanced Streamlit app:")
        print("streamlit run app/enhanced_streamlit_app.py")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 