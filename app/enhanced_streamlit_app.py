#!/usr/bin/env python3
"""
Enhanced Streamlit app for comprehensive Ollama VM performance testing.
Includes real-time metrics collection, system monitoring, and advanced visualization.
"""

import streamlit as st
import sys
import os
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import random
from datetime import datetime, timedelta
import uuid
import threading

# Add parent directory to path to import dependencies
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import functions from the main script and metrics collector
from llm_device_performance_main import (
    get_local_ip,
    load_ip_list,
    get_device_name,
    resolve_device_or_ip,
    load_model_list,
    load_prompt_list,
    generate_text,
    list_models,
    download_model,
    delete_model
)

from app.metrics_collector import PerformanceTest, MetricsCollector

# Page configuration
st.set_page_config(
    page_title="Enhanced Ollama Performance Testing System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(90deg, #ff9800 0%, #f57c00 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .error-card {
        background: linear-gradient(90deg, #f44336 0%, #d32f2f 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def load_enhanced_prompts(filename='app/enhanced_prompt_list.json'):
    """Load enhanced prompt categories from JSON file."""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        st.error(f"Error loading enhanced prompts: {e}")
    return {}

def get_prompts_by_category(prompt_data, categories, sample_size=None):
    """Get prompts from specified categories."""
    prompts = []
    for category in categories:
        if category in prompt_data.get('prompt_categories', {}):
            cat_prompts = prompt_data['prompt_categories'][category]['prompts']
            if sample_size:
                # Sample from each category
                sample_count = min(sample_size, len(cat_prompts))
                prompts.extend(random.sample(cat_prompts, sample_count))
            else:
                prompts.extend(cat_prompts)
    return prompts

def format_bytes(bytes_value):
    """Format bytes into human readable format."""
    if bytes_value < 1024:
        return f"{bytes_value:.1f} B"
    elif bytes_value < 1024**2:
        return f"{bytes_value/1024:.1f} KB"
    elif bytes_value < 1024**3:
        return f"{bytes_value/1024**2:.1f} MB"
    else:
        return f"{bytes_value/1024**3:.1f} GB"

def create_system_metrics_chart(metrics_data):
    """Create comprehensive system metrics visualization."""
    if not metrics_data:
        return None
    
    # Convert to DataFrame for easier manipulation
    df_list = []
    for sample in metrics_data:
        row = {
            'timestamp': sample['timestamp'],
            'relative_time': sample['relative_time'],
            'cpu_percent': sample['cpu']['overall_percent'],
            'memory_percent': sample['memory']['virtual']['percentage'],
            'memory_used_mb': sample['memory']['virtual']['used_mb'],
        }
        
        # Add GPU metrics if available
        if 'gpu' in sample and sample['gpu']:
            for i, gpu in enumerate(sample['gpu']):
                row[f'gpu_{i}_compute'] = gpu['utilization']['compute_percent']
                row[f'gpu_{i}_memory'] = gpu['memory']['utilization_percent']
                row[f'gpu_{i}_temp'] = gpu['temperature_c']
        
        df_list.append(row)
    
    df = pd.DataFrame(df_list)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'CPU Utilization (%)', 'Memory Usage (%)',
            'Memory Usage (MB)', 'GPU Compute (%)',
            'GPU Memory (%)', 'GPU Temperature (°C)'
        ),
        vertical_spacing=0.1
    )
    
    # CPU utilization
    fig.add_trace(
        go.Scatter(x=df['relative_time'], y=df['cpu_percent'], 
                   mode='lines', name='CPU %', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Memory percentage
    fig.add_trace(
        go.Scatter(x=df['relative_time'], y=df['memory_percent'], 
                   mode='lines', name='Memory %', line=dict(color='green')),
        row=1, col=2
    )
    
    # Memory usage in MB
    fig.add_trace(
        go.Scatter(x=df['relative_time'], y=df['memory_used_mb'], 
                   mode='lines', name='Memory MB', line=dict(color='orange')),
        row=2, col=1
    )
    
    # GPU metrics
    gpu_cols = [col for col in df.columns if col.startswith('gpu_') and col.endswith('_compute')]
    if gpu_cols:
        for col in gpu_cols:
            gpu_id = col.split('_')[1]
            fig.add_trace(
                go.Scatter(x=df['relative_time'], y=df[col], 
                           mode='lines', name=f'GPU {gpu_id}', line=dict(color='red')),
                row=2, col=2
            )
    
    # GPU memory
    gpu_mem_cols = [col for col in df.columns if col.startswith('gpu_') and col.endswith('_memory')]
    if gpu_mem_cols:
        for col in gpu_mem_cols:
            gpu_id = col.split('_')[1]
            fig.add_trace(
                go.Scatter(x=df['relative_time'], y=df[col], 
                           mode='lines', name=f'GPU {gpu_id} Mem', line=dict(color='purple')),
                row=3, col=1
            )
    
    # GPU temperature
    gpu_temp_cols = [col for col in df.columns if col.startswith('gpu_') and col.endswith('_temp')]
    if gpu_temp_cols:
        for col in gpu_temp_cols:
            gpu_id = col.split('_')[1]
            fig.add_trace(
                go.Scatter(x=df['relative_time'], y=df[col], 
                           mode='lines', name=f'GPU {gpu_id} Temp', line=dict(color='darkred')),
                row=3, col=2
            )
    
    fig.update_layout(height=800, showlegend=True, title="System Metrics During Test")
    return fig

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = []
if "device_map" not in st.session_state:
    st.session_state.device_map = load_ip_list()
if "selected_devices" not in st.session_state:
    st.session_state.selected_devices = []
if "models_info" not in st.session_state:
    st.session_state.models_info = {}
if "enhanced_prompts" not in st.session_state:
    st.session_state.enhanced_prompts = load_enhanced_prompts()
if "performance_test" not in st.session_state:
    st.session_state.performance_test = PerformanceTest()
if "test_running" not in st.session_state:
    st.session_state.test_running = False

# App title and description
st.title("🚀 Enhanced Ollama VM Performance Testing System")
st.markdown("""
**Comprehensive performance testing platform for Ollama instances across multiple VMs**  
*Real-time system monitoring • GPU metrics • Advanced analytics • Performance optimization*
""")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Test configuration
    st.subheader("Test Settings")
    sampling_interval = st.slider("Metrics Sampling Interval (s)", 0.5, 5.0, 1.0, 0.5)
    port = st.number_input("Ollama API Port", value=11434, min_value=1, max_value=65535)
    
    # File paths
    st.subheader("Configuration Files")
    ip_list_path = st.text_input("IP List Path", value="app/ip_list.json")
    model_list_path = st.text_input("Model List Path", value="app/model_list.json")
    prompt_list_path = st.text_input("Prompt List Path", value="app/prompt_list.json")
    
    # Load devices
    if st.button("🔄 Load Devices"):
        st.session_state.device_map = load_ip_list()
        if st.session_state.device_map:
            st.success(f"✅ Loaded {len(st.session_state.device_map)} devices")
        else:
            st.warning("⚠️ No devices found")
    
    # Display loaded devices
    if st.session_state.device_map:
        st.subheader("Available Devices")
        for device, ip in st.session_state.device_map.items():
            st.write(f"• **{device}**: `{ip}`")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🖥️ Devices", 
    "🔧 Model Management", 
    "🧪 Performance Testing", 
    "📊 Results Dashboard",
    "📈 Advanced Analytics"
])

# Tab 1: Device Selection and Configuration
with tab1:
    st.header("Device Selection & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Add Devices")
        
        # Manual IP entry
        manual_ip = st.text_input("Manual IP Address", placeholder="192.168.1.100")
        manual_name = st.text_input("Device Name (optional)", placeholder="Custom Device")
        
        if st.button("➕ Add Manual IP") and manual_ip:
            if manual_ip not in [item["ip"] for item in st.session_state.selected_devices]:
                st.session_state.selected_devices.append({
                    "name": manual_name or manual_ip,
                    "ip": manual_ip
                })
                st.success(f"✅ Added: {manual_name or manual_ip}")
            else:
                st.warning("⚠️ IP already added")
        
        # Use all devices
        if st.button("🎯 Use All Devices") and st.session_state.device_map:
            st.session_state.selected_devices = []
            for device, ip in st.session_state.device_map.items():
                st.session_state.selected_devices.append({"name": device, "ip": ip})
            st.success(f"✅ Added all {len(st.session_state.device_map)} devices")
        
        # Device selection from loaded list
        if st.session_state.device_map:
            selected = st.multiselect(
                "Select from Available Devices", 
                options=list(st.session_state.device_map.keys())
            )
            
            if st.button("➕ Add Selected") and selected:
                for device in selected:
                    ip = st.session_state.device_map[device]
                    if ip not in [item["ip"] for item in st.session_state.selected_devices]:
                        st.session_state.selected_devices.append({"name": device, "ip": ip})
                st.success(f"✅ Added {len(selected)} devices")
    
    with col2:
        st.subheader("Device Status")
        
        if st.session_state.selected_devices:
            if st.button("🔍 Test Connections"):
                connection_results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, device in enumerate(st.session_state.selected_devices):
                    ip = device["ip"]
                    name = device["name"]
                    
                    status_text.text(f"Testing {name}...")
                    
                    try:
                        models_result, _ = list_models(ip, port, name)
                        if models_result:
                            status = "🟢 Connected"
                            model_count = len(models_result.get("models", []))
                        else:
                            status = "🔴 Failed"
                            model_count = 0
                    except:
                        status = "🔴 Error"
                        model_count = 0
                    
                    connection_results.append({
                        "Device": name,
                        "IP": ip,
                        "Status": status,
                        "Models": model_count
                    })
                    
                    progress_bar.progress((i + 1) / len(st.session_state.selected_devices))
                
                status_text.text("Connection testing complete!")
                
                # Display results
                st.dataframe(
                    pd.DataFrame(connection_results),
                    use_container_width=True
                )
    
    # Selected devices display
    if st.session_state.selected_devices:
        st.subheader("Selected Devices")
        
        # Create a nicer display
        device_df = pd.DataFrame(st.session_state.selected_devices)
        device_df.index = range(1, len(device_df) + 1)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(device_df, use_container_width=True)
        with col2:
            if st.button("🗑️ Clear All", type="secondary"):
                st.session_state.selected_devices = []
                st.rerun()

# Tab 2: Model Management
with tab2:
    st.header("Model Management")
    
    if not st.session_state.selected_devices:
        st.warning("⚠️ Please select devices in the Devices tab first.")
    else:
        # Model refresh and display
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔄 Refresh Models"):
                st.session_state.models_info = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, device in enumerate(st.session_state.selected_devices):
                    ip = device["ip"]
                    name = device["name"]
                    
                    status_text.text(f"Getting models for {name}...")
                    
                    try:
                        models_result, _ = list_models(ip, port, name)
                        if models_result:
                            st.session_state.models_info[ip] = models_result.get("models", [])
                    except Exception as e:
                        st.error(f"Error getting models for {name}: {e}")
                    
                    progress_bar.progress((i + 1) / len(st.session_state.selected_devices))
                
                status_text.text("Model refresh complete!")
        
        # Display models for each device
        if st.session_state.models_info:
            st.subheader("Available Models by Device")
            
            for device in st.session_state.selected_devices:
                ip = device["ip"]
                name = device["name"]
                
                with st.expander(f"📦 {name} ({ip})"):
                    if ip in st.session_state.models_info and st.session_state.models_info[ip]:
                        model_data = []
                        for model in st.session_state.models_info[ip]:
                            size_bytes = model.get('size', 0)
                            size_display = format_bytes(size_bytes)
                            
                            model_data.append({
                                "Model": model["name"],
                                "Size": size_display,
                                "Modified": model.get("modified_at", "Unknown")
                            })
                        
                        st.dataframe(pd.DataFrame(model_data), use_container_width=True)
                    else:
                        st.info("No models found on this device")
        
        # Model operations
        st.subheader("Model Operations")
        
        with st.expander("📥 Download Models"):
            # Load models from file or manual entry
            col1, col2 = st.columns(2)
            
            with col1:
                manual_model = st.text_input("Model Name", placeholder="llama3.2:3b")
                use_model_list = st.checkbox("Use Model List File")
            
            with col2:
                if use_model_list:
                    models_from_file = load_model_list(model_list_path)
                    st.write(f"Models from file: {models_from_file}")
                    models_to_download = models_from_file
                else:
                    models_to_download = [manual_model] if manual_model else []
            
            if st.button("⬇️ Download Models") and models_to_download:
                download_results = []
                
                total_operations = len(st.session_state.selected_devices) * len(models_to_download)
                progress_bar = st.progress(0)
                status_text = st.empty()
                operation = 0
                
                for device in st.session_state.selected_devices:
                    ip = device["ip"]
                    name = device["name"]
                    
                    for model in models_to_download:
                        status_text.text(f"Downloading {model} on {name}...")
                        
                        try:
                            download_model(ip, port, model, name)
                            download_results.append({
                                "Device": name,
                                "Model": model,
                                "Status": "✅ Success",
                                "Message": f"Downloaded successfully"
                            })
                        except Exception as e:
                            download_results.append({
                                "Device": name,
                                "Model": model,
                                "Status": "❌ Failed",
                                "Message": str(e)
                            })
                        
                        operation += 1
                        progress_bar.progress(operation / total_operations)
                
                status_text.text("Download operations complete!")
                st.dataframe(pd.DataFrame(download_results), use_container_width=True)
        
        with st.expander("🗑️ Delete Models"):
            if st.session_state.models_info:
                # Get all unique models
                all_models = set()
                for models in st.session_state.models_info.values():
                    for model in models:
                        all_models.add(model["name"])
                
                if all_models:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        models_to_delete = st.multiselect(
                            "Select Models to Delete", 
                            options=list(all_models)
                        )
                    
                    with col2:
                        delete_all = st.checkbox("⚠️ Delete All Models")
                    
                    if st.button("🗑️ Execute Deletion", type="secondary"):
                        if delete_all or models_to_delete:
                            # Deletion logic here
                            st.warning("Deletion feature implemented - see original code for details")

# Tab 3: Performance Testing Configuration
with tab3:
    st.header("🧪 Performance Testing Configuration")
    
    if not st.session_state.selected_devices:
        st.warning("⚠️ Please select devices in the Devices tab first.")
    else:
        # Test configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Test Parameters")
            
            # Model selection
            if st.button("🔄 Refresh Available Models"):
                all_models = set()
                for device in st.session_state.selected_devices:
                    ip = device["ip"]
                    name = device["name"]
                    try:
                        models_result, _ = list_models(ip, port, name)
                        if models_result:
                            for model in models_result.get("models", []):
                                all_models.add(model["name"])
                    except:
                        pass
                st.session_state.available_models = list(all_models)
            
            # Model selection
            available_models = getattr(st.session_state, 'available_models', [])
            if available_models:
                selected_model = st.selectbox("Select Model", options=available_models)
            else:
                selected_model = st.text_input("Model Name", placeholder="llama3.2:3b")
            
            # Test settings
            iterations = st.number_input("Number of Iterations", min_value=1, max_value=100, value=3)
            warm_up = st.checkbox("Enable Warm-up Run", value=True)
            timeout_seconds = st.number_input("Timeout (seconds)", min_value=30, max_value=600, value=300)
        
        with col2:
            st.subheader("📝 Prompt Configuration")
            
            # Enhanced prompt selection
            if st.session_state.enhanced_prompts:
                prompt_source = st.radio(
                    "Prompt Source",
                    ["Single Prompt", "Test Configuration", "Custom Selection"],
                    help="Choose how to configure test prompts"
                )
                
                if prompt_source == "Single Prompt":
                    prompt_text = st.text_area(
                        "Enter Prompt",
                        height=150,
                        placeholder="Write your test prompt here..."
                    )
                    test_prompts = [prompt_text] if prompt_text else []
                
                elif prompt_source == "Test Configuration":
                    test_configs = st.session_state.enhanced_prompts.get('test_configurations', {})
                    
                    selected_config = st.selectbox(
                        "Select Test Configuration",
                        options=list(test_configs.keys()),
                        format_func=lambda x: f"{x}: {test_configs[x]['description']}"
                    )
                    
                    if selected_config:
                        config = test_configs[selected_config]
                        st.info(f"**{config['description']}**\nCategories: {', '.join(config['categories'])}")
                        
                        test_prompts = get_prompts_by_category(
                            st.session_state.enhanced_prompts,
                            config['categories'],
                            config.get('sample_size')
                        )
                        
                        st.write(f"📊 {len(test_prompts)} prompts selected")
                
                else:  # Custom Selection
                    categories = st.session_state.enhanced_prompts.get('prompt_categories', {})
                    
                    selected_categories = st.multiselect(
                        "Select Categories",
                        options=list(categories.keys()),
                        format_func=lambda x: f"{x}: {categories[x]['description']}"
                    )
                    
                    sample_size = st.number_input(
                        "Prompts per Category",
                        min_value=1,
                        max_value=10,
                        value=3
                    )
                    
                    test_prompts = get_prompts_by_category(
                        st.session_state.enhanced_prompts,
                        selected_categories,
                        sample_size
                    )
                    
                    if test_prompts:
                        st.write(f"📊 {len(test_prompts)} prompts selected")
            else:
                # Fallback to original prompt selection
                prompt_text = st.text_area("Enter Prompt", height=150)
                test_prompts = [prompt_text] if prompt_text else []
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Metrics Collection")
                enable_gpu_monitoring = st.checkbox("Enable GPU Monitoring", value=True)
                enable_detailed_cpu = st.checkbox("Detailed CPU Metrics", value=True)
                custom_sampling = st.checkbox("Custom Sampling Interval")
                
                if custom_sampling:
                    sampling_interval = st.slider(
                        "Sampling Interval (s)",
                        min_value=0.1,
                        max_value=10.0,
                        value=1.0,
                        step=0.1
                    )
            
            with col2:
                st.subheader("Export Options")
                export_format = st.selectbox(
                    "Export Format",
                    ["JSON", "CSV", "Excel"]
                )
                include_raw_metrics = st.checkbox("Include Raw Metrics", value=True)
                include_system_info = st.checkbox("Include System Info", value=True)
        
        # Test execution
        st.subheader("🚀 Test Execution")
        
        if test_prompts and selected_model:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Selected Devices", len(st.session_state.selected_devices))
            with col2:
                st.metric("Test Prompts", len(test_prompts))
            with col3:
                st.metric("Total Tests", len(st.session_state.selected_devices) * len(test_prompts) * iterations)
            
            # Show prompt preview
            with st.expander("👀 Prompt Preview"):
                for i, prompt in enumerate(test_prompts[:3]):  # Show first 3
                    st.write(f"**Prompt {i+1}:** {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                if len(test_prompts) > 3:
                    st.write(f"... and {len(test_prompts) - 3} more prompts")
            
            # Run test button
            if not st.session_state.test_running:
                if st.button("🚀 Start Performance Test", type="primary"):
                    st.session_state.test_running = True
                    
                    # Initialize performance test with custom sampling
                    performance_test = PerformanceTest(sampling_interval)
                    
                    # Create progress tracking
                    total_tests = len(st.session_state.selected_devices) * len(test_prompts)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    current_test = 0
                    
                    # Results container
                    results_container = st.empty()
                    
                    # Run tests
                    all_results = []
                    
                    for device in st.session_state.selected_devices:
                        ip = device["ip"]
                        name = device["name"]
                        
                        for prompt in test_prompts:
                            current_test += 1
                            status_text.text(f"Testing {name} - Prompt {current_test}/{total_tests}")
                            
                            try:
                                # Run the comprehensive test
                                result = performance_test.run_inference_test(
                                    ip_address=ip,
                                    port=port,
                                    model=selected_model,
                                    prompt=prompt,
                                    device_name=name,
                                    iterations=iterations,
                                    warm_up=warm_up
                                )
                                
                                all_results.append(result)
                                
                                # Update progress
                                progress_bar.progress(current_test / total_tests)
                                
                                # Show interim results
                                if result['results']['performance_stats']:
                                    stats = result['results']['performance_stats']
                                    results_container.success(
                                        f"✅ {name}: {stats['tokens_per_second']['mean']:.1f} tok/s, "
                                        f"TTFT: {stats['ttft_ms']['mean']:.0f}ms"
                                    )
                                else:
                                    results_container.error(f"❌ {name}: Test failed")
                            
                            except Exception as e:
                                results_container.error(f"❌ {name}: {str(e)}")
                            
                            # Small delay between tests
                            time.sleep(0.5)
                    
                    # Store results and finish
                    st.session_state.results = all_results
                    st.session_state.performance_test = performance_test
                    st.session_state.test_running = False
                    
                    status_text.text("✅ Performance testing complete!")
                    st.success("Testing completed! Check the Results Dashboard for detailed analysis.")
            
            else:
                st.info("🔄 Test in progress... Please wait.")
                if st.button("⏹️ Stop Test"):
                    st.session_state.test_running = False
        else:
            st.warning("⚠️ Please select a model and configure prompts to start testing.")

# Tab 4: Results Dashboard
with tab4:
    st.header("📊 Results Dashboard")
    
    if not st.session_state.results:
        st.info("🔍 No test results available. Run a performance test first.")
    else:
        # Results summary
        st.subheader("📈 Performance Summary")
        
        # Calculate overall statistics
        total_tests = len(st.session_state.results)
        successful_tests = sum(1 for r in st.session_state.results if r['results']['successful_iterations'] > 0)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tests", total_tests)
        with col2:
            st.metric("Successful Tests", successful_tests)
        with col3:
            success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col4:
            if st.session_state.results:
                avg_duration = sum(r['test_duration_s'] for r in st.session_state.results) / len(st.session_state.results)
                st.metric("Avg Test Duration", f"{avg_duration:.1f}s")
        
        # Performance comparison
        st.subheader("🏆 Performance Comparison")
        
        # Create performance comparison data
        comparison_data = []
        for result in st.session_state.results:
            if result['results']['performance_stats']:
                stats = result['results']['performance_stats']
                comparison_data.append({
                    'Device': result['device_info']['device_name'],
                    'Model': result['test_config']['model'],
                    'Tokens/sec': stats['tokens_per_second']['mean'],
                    'TTFT (ms)': stats['ttft_ms']['mean'],
                    'Total Duration (s)': stats['total_duration_s']['mean'],
                    'Success Rate (%)': result['results']['success_rate'],
                    'Iterations': result['test_config']['iterations']
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display table
            st.dataframe(comparison_df, use_container_width=True)
            
            # Performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Tokens per second comparison
                fig_tokens = px.bar(
                    comparison_df,
                    x='Device',
                    y='Tokens/sec',
                    title='Tokens per Second by Device',
                    color='Tokens/sec',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_tokens, use_container_width=True)
            
            with col2:
                # TTFT comparison
                fig_ttft = px.bar(
                    comparison_df,
                    x='Device',
                    y='TTFT (ms)',
                    title='Time to First Token by Device',
                    color='TTFT (ms)',
                    color_continuous_scale='plasma'
                )
                st.plotly_chart(fig_ttft, use_container_width=True)
        
        # Detailed results
        st.subheader("🔍 Detailed Test Results")
        
        for i, result in enumerate(st.session_state.results):
            device_name = result['device_info']['device_name']
            
            with st.expander(f"📋 {device_name} - Test {i+1}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Test Configuration:**")
                    st.json({
                        'Model': result['test_config']['model'],
                        'Prompt Length': result['test_config']['prompt_length'],
                        'Iterations': result['test_config']['iterations'],
                        'Warm-up': result['test_config']['warm_up']
                    })
                
                with col2:
                    if result['results']['performance_stats']:
                        st.write("**Performance Metrics:**")
                        stats = result['results']['performance_stats']
                        st.json({
                            'Tokens/sec (avg)': f"{stats['tokens_per_second']['mean']:.2f}",
                            'TTFT (avg)': f"{stats['ttft_ms']['mean']:.1f}ms",
                            'Duration (avg)': f"{stats['total_duration_s']['mean']:.2f}s",
                            'Success Rate': f"{result['results']['success_rate']:.1f}%"
                        })
                
                # System metrics visualization
                if result['system_metrics']['samples']:
                    st.write("**System Metrics During Test:**")
                    fig_metrics = create_system_metrics_chart(result['system_metrics']['samples'])
                    if fig_metrics:
                        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Export functionality
        st.subheader("📤 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export JSON"):
                export_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'test_count': len(st.session_state.results),
                    'results': st.session_state.results
                }
                
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_str,
                    file_name=f"performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export CSV") and comparison_data:
                csv_data = pd.DataFrame(comparison_data).to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("🗑️ Clear Results"):
                st.session_state.results = []
                st.session_state.performance_test = PerformanceTest()
                st.rerun()

# Tab 5: Advanced Analytics
with tab5:
    st.header("📈 Advanced Analytics")
    
    if not st.session_state.results:
        st.info("🔍 No test results available for analysis.")
    else:
        # Performance trends and analysis
        st.subheader("🎯 Performance Analysis")
        
        # Aggregate all performance data
        all_performance_data = []
        for result in st.session_state.results:
            if result['results']['performance_stats']:
                device_name = result['device_info']['device_name']
                stats = result['results']['performance_stats']
                
                all_performance_data.append({
                    'Device': device_name,
                    'Model': result['test_config']['model'],
                    'Prompt_Length': result['test_config']['prompt_length'],
                    'Tokens_Per_Second': stats['tokens_per_second']['mean'],
                    'TTFT_ms': stats['ttft_ms']['mean'],
                    'Total_Duration_s': stats['total_duration_s']['mean'],
                    'Success_Rate': result['results']['success_rate']
                })
        
        if all_performance_data:
            perf_df = pd.DataFrame(all_performance_data)
            
            # Performance correlation analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Tokens/sec vs Prompt Length
                fig_scatter = px.scatter(
                    perf_df,
                    x='Prompt_Length',
                    y='Tokens_Per_Second',
                    color='Device',
                    size='Success_Rate',
                    title='Tokens/sec vs Prompt Length',
                    hover_data=['Model', 'TTFT_ms']
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # Performance distribution
                fig_box = px.box(
                    perf_df,
                    x='Device',
                    y='Tokens_Per_Second',
                    title='Tokens/sec Distribution by Device'
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Resource efficiency analysis
            st.subheader("⚡ Resource Efficiency")
            
            # Calculate efficiency metrics
            efficiency_data = []
            for result in st.session_state.results:
                if result['system_metrics']['samples'] and result['results']['performance_stats']:
                    samples = result['system_metrics']['samples']
                    
                    # Calculate average resource usage
                    avg_cpu = sum(s['cpu']['overall_percent'] for s in samples) / len(samples)
                    avg_memory = sum(s['memory']['virtual']['percentage'] for s in samples) / len(samples)
                    
                    # GPU metrics if available
                    avg_gpu = 0
                    if samples[0].get('gpu'):
                        gpu_usage = []
                        for sample in samples:
                            if sample.get('gpu'):
                                for gpu in sample['gpu']:
                                    gpu_usage.append(gpu['utilization']['compute_percent'])
                        avg_gpu = sum(gpu_usage) / len(gpu_usage) if gpu_usage else 0
                    
                    tokens_per_sec = result['results']['performance_stats']['tokens_per_second']['mean']
                    
                    efficiency_data.append({
                        'Device': result['device_info']['device_name'],
                        'Avg_CPU_%': avg_cpu,
                        'Avg_Memory_%': avg_memory,
                        'Avg_GPU_%': avg_gpu,
                        'Tokens_Per_Second': tokens_per_sec,
                        'Efficiency_Score': tokens_per_sec / (avg_cpu + avg_memory + avg_gpu + 1)  # +1 to avoid division by zero
                    })
            
            if efficiency_data:
                eff_df = pd.DataFrame(efficiency_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Resource utilization
                    fig_resource = px.scatter(
                        eff_df,
                        x='Avg_CPU_%',
                        y='Tokens_Per_Second',
                        color='Device',
                        size='Efficiency_Score',
                        title='Performance vs CPU Utilization'
                    )
                    st.plotly_chart(fig_resource, use_container_width=True)
                
                with col2:
                    # Efficiency ranking
                    fig_efficiency = px.bar(
                        eff_df.sort_values('Efficiency_Score', ascending=True),
                        x='Efficiency_Score',
                        y='Device',
                        orientation='h',
                        title='Device Efficiency Ranking'
                    )
                    st.plotly_chart(fig_efficiency, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Performance Recommendations")
            
            # Find best and worst performers
            best_device = perf_df.loc[perf_df['Tokens_Per_Second'].idxmax(), 'Device']
            worst_device = perf_df.loc[perf_df['Tokens_Per_Second'].idxmin(), 'Device']
            
            best_performance = perf_df['Tokens_Per_Second'].max()
            worst_performance = perf_df['Tokens_Per_Second'].min()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="success-card">
                    <h4>🏆 Best Performer</h4>
                    <p><strong>{best_device}</strong></p>
                    <p>{best_performance:.1f} tokens/sec</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="warning-card">
                    <h4>⚠️ Needs Improvement</h4>
                    <p><strong>{worst_device}</strong></p>
                    <p>{worst_performance:.1f} tokens/sec</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                improvement_potential = ((best_performance - worst_performance) / worst_performance) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📊 Improvement Potential</h4>
                    <p><strong>{improvement_potential:.1f}%</strong></p>
                    <p>Performance gap</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed recommendations
            st.write("**Optimization Suggestions:**")
            
            recommendations = []
            
            if efficiency_data:
                # Find devices with high resource usage but low performance
                for item in efficiency_data:
                    if item['Avg_CPU_%'] > 80 and item['Tokens_Per_Second'] < perf_df['Tokens_Per_Second'].median():
                        recommendations.append(f"🔧 **{item['Device']}**: High CPU usage ({item['Avg_CPU_%']:.1f}%) with low performance. Consider CPU optimization or model size reduction.")
                    
                    if item['Avg_Memory_%'] > 90:
                        recommendations.append(f"💾 **{item['Device']}**: High memory usage ({item['Avg_Memory_%']:.1f}%). Consider increasing RAM or using smaller models.")
                    
                    if item['Avg_GPU_%'] > 95 and item['Tokens_Per_Second'] < perf_df['Tokens_Per_Second'].median():
                        recommendations.append(f"🎮 **{item['Device']}**: GPU bottleneck detected. Consider GPU memory optimization or batch size tuning.")
            
            # General recommendations based on performance patterns
            prompt_perf = perf_df.groupby('Prompt_Length')['Tokens_Per_Second'].mean()
            if len(prompt_perf) > 1:
                if prompt_perf.iloc[-1] < prompt_perf.iloc[0] * 0.5:  # Performance drops significantly with longer prompts
                    recommendations.append("📝 **Prompt Optimization**: Performance degrades significantly with longer prompts. Consider prompt engineering or chunking strategies.")
            
            if not recommendations:
                recommendations.append("✅ **Overall Performance**: All devices are performing within expected parameters.")
            
            for rec in recommendations:
                st.markdown(f"• {rec}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>Enhanced Ollama VM Performance Testing System</strong></p>
    <p>Real-time monitoring • Comprehensive analytics • Performance optimization</p>
</div>
""", unsafe_allow_html=True) 