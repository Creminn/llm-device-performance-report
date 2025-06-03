# 🚀 Quick Start Guide - Enhanced Ollama Performance Testing

Get your enhanced Ollama VM performance testing system up and running in minutes!

## 📋 Prerequisites

1. **Ollama instances** running on your VMs/devices
2. **Python 3.7+** installed
3. **Network access** to your Ollama instances

## ⚡ 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Your Devices
Edit `app/ip_list.json`:
```json
{
    "Main Server": "192.168.1.100",
    "GPU Workstation": "192.168.1.101", 
    "Test Device": "192.168.1.102"
}
```

### 3. Test Your Setup
```bash
python test_enhanced_app.py
```

### 4. Launch the Enhanced Interface
```bash
streamlit run app/enhanced_streamlit_app.py
```

## 🎯 First Performance Test

1. **Open the web interface** (usually http://localhost:8501)

2. **Go to Devices tab**:
   - Click "🔄 Load Devices"
   - Click "🎯 Use All Devices" 
   - Click "🔍 Test Connections"

3. **Go to Model Management tab**:
   - Click "🔄 Refresh Models"
   - Verify your models are available

4. **Go to Performance Testing tab**:
   - Select a model (e.g., "llama3.2:3b")
   - Choose "Test Configuration" → "Quick Test"
   - Click "🚀 Start Performance Test"

5. **View Results** in the Results Dashboard tab!

## 📊 Key Metrics You'll See

- **Tokens/sec**: How fast your models generate text
- **TTFT (Time to First Token)**: Response latency in milliseconds  
- **Resource Usage**: CPU, Memory, GPU utilization during inference
- **System Charts**: Real-time monitoring during tests

## 🎨 Test Configurations

| Configuration | Best For | Duration |
|---------------|----------|----------|
| **Quick Test** | Fast validation | ~1-2 minutes |
| **Standard Test** | Regular monitoring | ~3-5 minutes |
| **Comprehensive Test** | Full evaluation | ~10-15 minutes |
| **Stress Test** | Resource limits | ~5-10 minutes |

## 🔧 Troubleshooting

### Can't Connect to Devices?
- Verify Ollama is running: `curl http://YOUR-IP:11434/api/tags`
- Check firewall settings
- Confirm IP addresses in `ip_list.json`

### GPU Metrics Not Working?
- Install NVIDIA drivers
- Run: `pip install nvidia-ml-py`
- System will fallback to CPU/memory monitoring if GPU unavailable

### Performance Issues?
- Reduce sampling interval in advanced settings
- Limit concurrent device testing
- Use shorter prompts for initial testing

## 🚀 Pro Tips

1. **Warm-up Models**: Enable warm-up runs for consistent results
2. **Batch Testing**: Use test configurations for systematic evaluation  
3. **Export Data**: Download results in JSON/CSV for further analysis
4. **Monitor Resources**: Watch the system metrics during tests
5. **Compare Devices**: Use the Advanced Analytics tab for insights

## 📈 Next Steps

- Explore **Advanced Analytics** for optimization recommendations
- Set up **scheduled testing** for continuous monitoring
- Create **custom prompt categories** for your specific use cases
- Use **export features** to integrate with external monitoring tools

## 🆘 Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Review error messages in the Streamlit interface
- Run the test script: `python test_enhanced_app.py`

---

**Happy Testing! 🎉** Your enhanced Ollama performance testing system is ready to optimize your AI infrastructure! 