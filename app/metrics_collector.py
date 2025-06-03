#!/usr/bin/env python3
"""
Comprehensive metrics collection system for Ollama VM performance testing.
Collects system resources, GPU metrics, and inference performance data.
"""

import psutil
import time
import threading
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests

try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    print("Warning: nvidia-ml-py not available. GPU metrics will be disabled.")


class MetricsCollector:
    """Comprehensive metrics collector for performance testing."""
    
    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize the metrics collector.
        
        Args:
            sampling_interval: Interval in seconds between metric samples
        """
        self.sampling_interval = sampling_interval
        self.is_collecting = False
        self.metrics_data = []
        self.collection_thread = None
        self.start_time = None
        
        # Initialize NVIDIA monitoring if available
        self.gpu_available = False
        if NVIDIA_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_available = self.gpu_count > 0
                print(f"Initialized NVIDIA monitoring: {self.gpu_count} GPU(s) detected")
            except Exception as e:
                print(f"Failed to initialize NVIDIA monitoring: {e}")
                self.gpu_available = False
    
    def start_collection(self):
        """Start metrics collection in a background thread."""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.start_time = time.time()
        self.metrics_data = []
        self.collection_thread = threading.Thread(target=self._collect_metrics_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        print("Metrics collection started")
    
    def stop_collection(self) -> List[Dict[str, Any]]:
        """Stop metrics collection and return collected data."""
        if not self.is_collecting:
            return self.metrics_data
        
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
        
        print(f"Metrics collection stopped. Collected {len(self.metrics_data)} samples")
        return self.metrics_data
    
    def _collect_metrics_loop(self):
        """Main metrics collection loop running in background thread."""
        while self.is_collecting:
            try:
                timestamp = time.time()
                relative_time = timestamp - self.start_time
                
                metrics = {
                    'timestamp': timestamp,
                    'relative_time': relative_time,
                    'system': self._collect_system_metrics(),
                    'memory': self._collect_memory_metrics(),
                    'cpu': self._collect_cpu_metrics(),
                }
                
                if self.gpu_available:
                    metrics['gpu'] = self._collect_gpu_metrics()
                
                self.metrics_data.append(metrics)
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                time.sleep(self.sampling_interval)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect general system metrics."""
        return {
            'boot_time': psutil.boot_time(),
            'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            'process_count': len(psutil.pids()),
        }
    
    def _collect_memory_metrics(self) -> Dict[str, Any]:
        """Collect memory usage metrics."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'virtual': {
                'total_mb': memory.total / (1024 * 1024),
                'available_mb': memory.available / (1024 * 1024),
                'used_mb': memory.used / (1024 * 1024),
                'percentage': memory.percent,
                'free_mb': memory.free / (1024 * 1024),
                'cached_mb': getattr(memory, 'cached', 0) / (1024 * 1024),
                'buffers_mb': getattr(memory, 'buffers', 0) / (1024 * 1024)
            },
            'swap': {
                'total_mb': swap.total / (1024 * 1024),
                'used_mb': swap.used / (1024 * 1024),
                'free_mb': swap.free / (1024 * 1024),
                'percentage': swap.percent
            }
        }
    
    def _collect_cpu_metrics(self) -> Dict[str, Any]:
        """Collect CPU utilization metrics."""
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        cpu_freq = psutil.cpu_freq()
        cpu_stats = psutil.cpu_stats()
        
        return {
            'overall_percent': cpu_percent,
            'per_core_percent': cpu_per_core,
            'frequency': {
                'current_mhz': cpu_freq.current if cpu_freq else None,
                'min_mhz': cpu_freq.min if cpu_freq else None,
                'max_mhz': cpu_freq.max if cpu_freq else None
            },
            'stats': {
                'ctx_switches': cpu_stats.ctx_switches,
                'interrupts': cpu_stats.interrupts,
                'soft_interrupts': cpu_stats.soft_interrupts,
                'syscalls': cpu_stats.syscalls
            },
            'count': {
                'logical': psutil.cpu_count(logical=True),
                'physical': psutil.cpu_count(logical=False)
            }
        }
    
    def _collect_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Collect GPU metrics for all available GPUs."""
        if not self.gpu_available:
            return []
        
        gpu_metrics = []
        
        for i in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Basic GPU info
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                elif not isinstance(name, str):
                    name = str(name)
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = None
                
                # Power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power = None
                
                # Clock speeds
                try:
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except:
                    graphics_clock = None
                    memory_clock = None
                
                gpu_data = {
                    'gpu_id': i,
                    'name': name,
                    'memory': {
                        'total_mb': mem_info.total / (1024 * 1024),
                        'used_mb': mem_info.used / (1024 * 1024),
                        'free_mb': mem_info.free / (1024 * 1024),
                        'utilization_percent': (mem_info.used / mem_info.total) * 100
                    },
                    'utilization': {
                        'compute_percent': util.gpu,
                        'memory_percent': util.memory
                    },
                    'temperature_c': temp,
                    'power_watts': power,
                    'clocks': {
                        'graphics_mhz': graphics_clock,
                        'memory_mhz': memory_clock
                    }
                }
                
                gpu_metrics.append(gpu_data)
                
            except Exception as e:
                print(f"Error collecting metrics for GPU {i}: {e}")
        
        return gpu_metrics


class PerformanceTest:
    """Comprehensive performance test orchestrator."""
    
    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize performance test.
        
        Args:
            sampling_interval: Metrics sampling interval in seconds
        """
        self.metrics_collector = MetricsCollector(sampling_interval)
        self.test_results = []
    
    def run_inference_test(
        self,
        ip_address: str,
        port: int,
        model: str,
        prompt: str,
        device_name: str = None,
        iterations: int = 1,
        warm_up: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive inference performance test.
        
        Args:
            ip_address: Device IP address
            port: Ollama API port
            model: Model name
            prompt: Test prompt
            device_name: Optional device name for identification
            iterations: Number of test iterations
            warm_up: Whether to perform warm-up run
            
        Returns:
            Comprehensive test results including all metrics
        """
        test_id = str(uuid.uuid4())
        test_start_time = datetime.now()
        
        # Warm-up run if requested
        if warm_up:
            print("Performing warm-up run...")
            try:
                self._single_inference_call(ip_address, port, model, "Hello", device_name)
                time.sleep(2)  # Wait for model to fully load
            except Exception as e:
                print(f"Warm-up failed: {e}")
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        iteration_results = []
        
        try:
            for i in range(iterations):
                print(f"Running iteration {i+1}/{iterations}")
                
                # Record inference timing
                inference_start = time.time()
                
                try:
                    response = self._single_inference_call(ip_address, port, model, prompt, device_name)
                    inference_end = time.time()
                    
                    if response:
                        # Calculate performance metrics
                        total_duration_s = response.get('total_duration', 0) / 1e9  # Convert to seconds
                        load_duration_s = response.get('load_duration', 0) / 1e9
                        prompt_eval_duration_s = response.get('prompt_eval_duration', 0) / 1e9
                        eval_duration_s = response.get('eval_duration', 0) / 1e9
                        
                        prompt_eval_count = response.get('prompt_eval_count', 0)
                        eval_count = response.get('eval_count', 0)
                        
                        # Calculate rates
                        tokens_per_second = eval_count / eval_duration_s if eval_duration_s > 0 else 0
                        prompt_tokens_per_second = prompt_eval_count / prompt_eval_duration_s if prompt_eval_duration_s > 0 else 0
                        
                        # Time to first token (TTFT)
                        ttft_s = load_duration_s + prompt_eval_duration_s
                        
                        iteration_result = {
                            'iteration': i + 1,
                            'success': True,
                            'inference_time_s': inference_end - inference_start,
                            'response_length': len(response.get('response', '')),
                            'metrics': {
                                'total_duration_s': total_duration_s,
                                'load_duration_s': load_duration_s,
                                'prompt_eval_duration_s': prompt_eval_duration_s,
                                'eval_duration_s': eval_duration_s,
                                'prompt_eval_count': prompt_eval_count,
                                'eval_count': eval_count,
                                'tokens_per_second': tokens_per_second,
                                'prompt_tokens_per_second': prompt_tokens_per_second,
                                'ttft_s': ttft_s,
                                'ttft_ms': ttft_s * 1000
                            },
                            'response_sample': response.get('response', '')[:200] + '...' if len(response.get('response', '')) > 200 else response.get('response', '')
                        }
                    else:
                        iteration_result = {
                            'iteration': i + 1,
                            'success': False,
                            'error': 'No response received',
                            'inference_time_s': inference_end - inference_start
                        }
                
                except Exception as e:
                    inference_end = time.time()
                    iteration_result = {
                        'iteration': i + 1,
                        'success': False,
                        'error': str(e),
                        'inference_time_s': inference_end - inference_start
                    }
                
                iteration_results.append(iteration_result)
                
                # Small delay between iterations
                if i < iterations - 1:
                    time.sleep(1)
        
        finally:
            # Stop metrics collection
            system_metrics = self.metrics_collector.stop_collection()
        
        test_end_time = datetime.now()
        
        # Calculate aggregate statistics
        successful_iterations = [r for r in iteration_results if r.get('success', False)]
        
        if successful_iterations:
            # Performance statistics
            tokens_per_second_values = [r['metrics']['tokens_per_second'] for r in successful_iterations]
            ttft_values = [r['metrics']['ttft_ms'] for r in successful_iterations]
            total_duration_values = [r['metrics']['total_duration_s'] for r in successful_iterations]
            
            performance_stats = {
                'tokens_per_second': {
                    'mean': sum(tokens_per_second_values) / len(tokens_per_second_values),
                    'min': min(tokens_per_second_values),
                    'max': max(tokens_per_second_values),
                    'values': tokens_per_second_values
                },
                'ttft_ms': {
                    'mean': sum(ttft_values) / len(ttft_values),
                    'min': min(ttft_values),
                    'max': max(ttft_values),
                    'values': ttft_values
                },
                'total_duration_s': {
                    'mean': sum(total_duration_values) / len(total_duration_values),
                    'min': min(total_duration_values),
                    'max': max(total_duration_values),
                    'values': total_duration_values
                }
            }
        else:
            performance_stats = None
        
        # Compile comprehensive test results
        test_result = {
            'test_id': test_id,
            'timestamp': test_start_time.isoformat(),
            'test_duration_s': (test_end_time - test_start_time).total_seconds(),
            'device_info': {
                'ip_address': ip_address,
                'device_name': device_name or ip_address,
                'port': port
            },
            'test_config': {
                'model': model,
                'prompt': prompt,
                'prompt_length': len(prompt),
                'iterations': iterations,
                'warm_up': warm_up
            },
            'results': {
                'successful_iterations': len(successful_iterations),
                'failed_iterations': len(iteration_results) - len(successful_iterations),
                'success_rate': len(successful_iterations) / len(iteration_results) * 100,
                'performance_stats': performance_stats,
                'iteration_details': iteration_results
            },
            'system_metrics': {
                'sample_count': len(system_metrics),
                'sampling_interval_s': self.metrics_collector.sampling_interval,
                'samples': system_metrics
            }
        }
        
        self.test_results.append(test_result)
        return test_result
    
    def _single_inference_call(self, ip_address: str, port: int, model: str, prompt: str, device_name: str = None) -> Optional[Dict]:
        """Perform a single inference call to Ollama."""
        url = f"http://{ip_address}:{port}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=300)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"Error during inference: {e}")
            return None
    
    def get_test_results(self) -> List[Dict[str, Any]]:
        """Get all test results."""
        return self.test_results
    
    def clear_results(self):
        """Clear all test results."""
        self.test_results = []
        print("Test results cleared")
    
    def export_results(self, filename: str = None) -> str:
        """Export test results to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_test_results_{timestamp}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'test_count': len(self.test_results),
            'results': self.test_results
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Results exported to {filename}")
        return filename 