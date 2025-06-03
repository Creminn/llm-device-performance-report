# Enhanced Ollama VM Performance Testing System

A comprehensive performance testing platform for Ollama instances running across multiple VMs in a Proxmox environment. This system provides real-time metrics collection, system monitoring, GPU utilization tracking, and advanced analytics for optimizing AI model performance.

## 🚀 Features

### Core Performance Metrics
- **Tokens per second (tok/s)** - Generation and prompt processing speed
- **Time to First Token (TTFT)** - Response latency measurement  
- **RAM usage** - Current, peak, and available memory monitoring
- **VRAM usage** - GPU memory utilization and allocation tracking
- **CPU utilization** - Per-core breakdown and overall percentage
- **GPU utilization** - Compute percentage and memory bandwidth
- **Model loading time** - Cold start performance measurement
- **Total inference time** - Complete end-to-end request duration

### Advanced Capabilities
- **Real-time system monitoring** during inference
- **Comprehensive prompt categories** (short, medium, long, complex reasoning, creative, technical)
- **Configurable test scenarios** (quick test, standard test, comprehensive test, stress test)
- **Multi-device parallel testing**
- **Advanced analytics and recommendations**
- **Resource efficiency analysis**
- **Performance trend visualization**
- **Automated report generation**

## 📁 Project Structure

```
llm-device-performance-report/
├── app/
│   ├── streamlit_app.py           # Original Streamlit interface
│   ├── enhanced_streamlit_app.py  # Enhanced performance testing interface
│   ├── metrics_collector.py       # Comprehensive metrics collection system
│   ├── enhanced_prompt_list.json  # Categorized prompt templates
│   ├── ip_list.json               # Device configuration
│   ├── model_list.json            # Model management
│   └── prompt_list.json           # Basic prompt list
├── llm_device_performance_main.py # Core functionality
├── requirements.txt               # Dependencies
└── README.md                     # This file
```

## 🛠 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd llm-device-performance-report
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure your devices:**
Edit `app/ip_list.json` with your VM configurations:
```json
{
    "PC with RTX 4090": "192.168.0.40",
    "Raspberry Pi 5": "192.168.0.186", 
    "ASUS Laptop": "192.168.0.223",
    "Home Server": "192.168.1.123"
}
```

## 🚀 Usage

### Enhanced Performance Testing Interface

Run the enhanced Streamlit application:
```bash
streamlit run app/enhanced_streamlit_app.py
```

### Application Tabs

#### 1. 🖥️ Devices Tab
- **Device Selection**: Add VMs manually or from configuration file
- **Connection Testing**: Verify Ollama connectivity and model availability
- **Status Monitoring**: Real-time device status and model counts

#### 2. 🔧 Model Management Tab  
- **Model Discovery**: Automatic detection of available models
- **Bulk Operations**: Download/delete models across multiple devices
- **Size Monitoring**: Track model storage usage

#### 3. 🧪 Performance Testing Tab
- **Test Configuration**: 
  - Model selection across devices
  - Prompt configuration (single, test scenarios, custom categories)
  - Advanced settings (iterations, warm-up, timeouts)
- **Enhanced Prompts**:
  - **Short prompts** (< 50 tokens): Quick response testing
  - **Medium prompts** (50-200 tokens): Standard performance evaluation
  - **Long prompts** (200+ tokens): Comprehensive testing and stress testing  
  - **Complex reasoning**: Multi-step problem solving
  - **Creative prompts**: Open-ended creativity testing
  - **Technical prompts**: Domain-specific knowledge testing
- **Test Execution**: Real-time progress tracking with interim results

#### 4. 📊 Results Dashboard Tab
- **Performance Summary**: Overall statistics and success rates
- **Comparison Charts**: Tokens/sec, TTFT, and duration comparisons
- **Detailed Results**: Per-device metrics with system monitoring charts
- **Export Options**: JSON, CSV formats with customizable data inclusion

#### 5. 📈 Advanced Analytics Tab
- **Performance Analysis**: Correlation between prompt length and performance
- **Resource Efficiency**: CPU, memory, and GPU utilization analysis
- **Optimization Recommendations**: Automated suggestions for improvement
- **Efficiency Ranking**: Device performance comparison and scoring

## 📊 Test Configurations

### Pre-configured Test Scenarios

| Scenario | Description | Categories | Sample Size |
|----------|-------------|------------|-------------|
| **Quick Test** | Fast performance check | Short prompts | 3 |
| **Standard Test** | Balanced evaluation | Short + Medium | 5 |
| **Comprehensive Test** | Full capability assessment | Short + Medium + Long + Complex | 8 |
| **Stress Test** | Resource intensive testing | Long + Complex + Technical | 5 |
| **Creativity Test** | Creative response evaluation | Creative prompts | 3 |

### Custom Configuration
- Select specific prompt categories
- Define sample sizes per category  
- Configure iteration counts (1-100)
- Set custom sampling intervals (0.1-10s)
- Enable/disable specific metrics collection

## 🔍 Metrics Collection System

### Real-time Monitoring
- **Background threading** for concurrent metrics collection
- **Configurable sampling intervals** (0.5-5 seconds)
- **Error resilience** with graceful degradation
- **Synchronized timestamps** across devices

### System Metrics
```python
{
    "cpu": {
        "overall_percent": float,
        "per_core_percent": [float],
        "frequency": {"current_mhz": float, "min_mhz": float, "max_mhz": float},
        "count": {"logical": int, "physical": int}
    },
    "memory": {
        "virtual": {
            "total_mb": float, "used_mb": float, "available_mb": float,
            "percentage": float, "cached_mb": float, "buffers_mb": float
        },
        "swap": {"total_mb": float, "used_mb": float, "percentage": float}
    },
    "gpu": [{
        "gpu_id": int, "name": str,
        "memory": {"total_mb": float, "used_mb": float, "utilization_percent": float},
        "utilization": {"compute_percent": float, "memory_percent": float},
        "temperature_c": float, "power_watts": float,
        "clocks": {"graphics_mhz": float, "memory_mhz": float}
    }]
}
```

### Performance Metrics  
```python
{
    "tokens_per_second": {"mean": float, "min": float, "max": float, "values": [float]},
    "ttft_ms": {"mean": float, "min": float, "max": float, "values": [float]},
    "total_duration_s": {"mean": float, "min": float, "max": float, "values": [float]},
    "success_rate": float,
    "iteration_details": [...]
}
```

## 🎯 Advanced Features

### Error Handling
- **Connection timeouts** with configurable limits
- **Graceful degradation** when GPU monitoring unavailable
- **Partial test recovery** for interrupted sessions
- **Comprehensive error logging** and user feedback

### Export and Reporting
- **Multiple formats**: JSON, CSV, Excel
- **Configurable data inclusion**: Raw metrics, system info, aggregated stats
- **Timestamped exports** with comprehensive metadata
- **Download buttons** for immediate access

### Performance Optimization
- **Automated recommendations** based on resource utilization patterns
- **Efficiency scoring** algorithm for device ranking
- **Bottleneck detection** for CPU, memory, and GPU constraints
- **Configuration suggestions** for optimal performance

## 🔧 Configuration

### Dependencies
```txt
requests>=2.25.0
streamlit>=1.28.0
pandas>=1.3.0
plotly>=5.0.0
psutil>=5.8.0
nvidia-ml-py>=11.495.46
```

### GPU Monitoring Requirements
- **NVIDIA GPUs**: Requires `nvidia-ml-py` package
- **Non-NVIDIA GPUs**: System falls back to CPU/memory monitoring
- **Driver compatibility**: Ensure NVIDIA drivers are properly installed

### System Requirements
- **Python 3.7+**
- **Network access** to Ollama instances
- **Sufficient memory** for metrics storage during long tests
- **Administrative privileges** may be required for detailed system metrics

## 📈 Performance Analysis

### Key Performance Indicators (KPIs)
- **Throughput**: Tokens per second generation rate
- **Latency**: Time to first token (TTFT) 
- **Efficiency**: Performance per resource unit consumed
- **Reliability**: Success rate across test iterations
- **Scalability**: Performance consistency across prompt lengths

### Resource Efficiency Metrics
- **CPU Efficiency**: Tokens/sec per CPU percentage
- **Memory Efficiency**: Performance per GB memory used
- **GPU Efficiency**: Throughput per GPU utilization percentage
- **Power Efficiency**: Performance per watt consumed (when available)

## 🚨 Troubleshooting

### Common Issues

1. **GPU Metrics Not Available**
   - Install `nvidia-ml-py`: `pip install nvidia-ml-py`
   - Verify NVIDIA drivers are installed
   - Check CUDA compatibility

2. **Connection Timeouts**
   - Verify Ollama service is running on target devices
   - Check network connectivity and firewall settings
   - Increase timeout values in advanced settings

3. **Memory Issues During Long Tests**
   - Reduce sampling frequency
   - Limit concurrent device testing
   - Clear results between test sessions

4. **Incomplete Test Results**
   - Check device connectivity during tests
   - Verify model availability on all devices
   - Review error logs in the interface

### Performance Optimization Tips

1. **For Better Throughput**:
   - Use larger models with sufficient VRAM
   - Enable GPU acceleration
   - Optimize batch sizes

2. **For Lower Latency**:
   - Keep models warm with periodic requests
   - Use SSD storage for model files
   - Minimize network overhead

3. **For Resource Efficiency**:
   - Monitor memory usage patterns
   - Balance CPU cores vs model size
   - Consider quantized model variants

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama** - For providing the excellent LLM serving platform
- **Streamlit** - For the intuitive web application framework
- **Plotly** - For comprehensive data visualization capabilities
- **NVIDIA** - For GPU monitoring libraries and CUDA support 