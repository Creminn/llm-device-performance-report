#!/usr/bin/env python3
"""
Streamlit app for LLM Device Performance Report using functionality from llm_device_performance_main.py
"""

import streamlit as st
import sys
import os
import time
import json
import pandas as pd
import plotly.express as px
from pathlib import Path

# Add parent directory to path to import llm_device_performance_main
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import functions from the main script
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

# Set page configuration
st.set_page_config(
    page_title="LLM Device Performance Report",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = []
if "device_map" not in st.session_state:
    st.session_state.device_map = load_ip_list()
if "selected_devices" not in st.session_state:
    st.session_state.selected_devices = []
if "models_info" not in st.session_state:
    st.session_state.models_info = {}

# App title
st.title("🤖 LLM Device Performance Report")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # File paths
    st.subheader("Configuration Files")
    ip_list_path = st.text_input("IP List Path", value="ip_list.json", key="ip_list_path")
    model_list_path = st.text_input("Model List Path", value="model_list.json", key="model_list_path")
    prompt_list_path = st.text_input("Prompt List Path", value="prompt_list.json", key="prompt_list_path")
    
    # Port
    port = st.number_input("Ollama API Port", value=11434, min_value=1, max_value=65535, key="port")
    
    # Load device map button
    if st.button("Load Devices", key="load_devices"):
        st.session_state.device_map = load_ip_list()
        if st.session_state.device_map:
            st.success(f"Loaded {len(st.session_state.device_map)} devices")
        else:
            st.warning("No devices found or error loading file")
    
    # Display the loaded devices
    if st.session_state.device_map:
        st.subheader("Available Devices")
        for device, ip in st.session_state.device_map.items():
            st.write(f"- {device}: {ip}")

# Main content area - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Device Selection", "Model Management", "Performance Testing", "Results"])

# Tab 1: Device Selection
with tab1:
    st.header("Select Devices")
    
    # Manual IP entry
    col1, col2 = st.columns(2)
    with col1:
        manual_ip = st.text_input("Manual IP Address", key="manual_ip")
        if st.button("Add Manual IP", key="add_manual_ip") and manual_ip:
            if manual_ip not in [item["ip"] for item in st.session_state.selected_devices]:
                st.session_state.selected_devices.append({
                    "name": None,
                    "ip": manual_ip
                })
                st.success(f"Added IP: {manual_ip}")
    
    # Use all devices from IP list
    with col2:
        if st.button("Use All Devices", key="use_all_devices") and st.session_state.device_map:
            st.session_state.selected_devices = []
            for device, ip in st.session_state.device_map.items():
                st.session_state.selected_devices.append({
                    "name": device,
                    "ip": ip
                })
            st.success(f"Added all {len(st.session_state.device_map)} devices")
    
    # Device selection from loaded list
    if st.session_state.device_map:
        st.subheader("Select from Loaded Devices")
        device_options = list(st.session_state.device_map.keys())
        selected = st.multiselect("Devices", options=device_options, key="device_multiselect")
        
        if st.button("Add Selected Devices", key="add_selected_devices") and selected:
            for device in selected:
                ip = st.session_state.device_map[device]
                if ip not in [item["ip"] for item in st.session_state.selected_devices]:
                    st.session_state.selected_devices.append({
                        "name": device,
                        "ip": ip
                    })
            st.success(f"Added {len(selected)} devices")
    
    # Display selected devices
    if st.session_state.selected_devices:
        st.subheader("Selected Devices")
        selected_df = pd.DataFrame(st.session_state.selected_devices)
        selected_df.index = range(1, len(selected_df) + 1)
        st.dataframe(selected_df)
        
        if st.button("Clear Selected Devices", key="clear_devices"):
            st.session_state.selected_devices = []
            st.success("Cleared device selection")
        
        # Test connection to selected devices
        if st.button("Test Connections", key="test_connections"):
            connection_results = []
            for device in st.session_state.selected_devices:
                ip = device["ip"]
                name = device["name"] or ip
                
                with st.spinner(f"Testing connection to {name}..."):
                    models_result, _ = list_models(ip, port, name)
                    status = "Connected" if models_result else "Failed"
                    model_count = len(models_result.get("models", [])) if models_result else 0
                    
                    connection_results.append({
                        "Device": name,
                        "IP": ip,
                        "Status": status,
                        "Models": model_count
                    })
            
            st.subheader("Connection Test Results")
            connection_df = pd.DataFrame(connection_results)
            st.dataframe(connection_df)

# Tab 2: Model Management
with tab2:
    st.header("Model Management")
    
    if not st.session_state.selected_devices:
        st.warning("Please select devices in the Device Selection tab first.")
    else:
        # Refresh available models
        if st.button("Refresh Available Models", key="refresh_models"):
            st.session_state.models_info = {}
            
            for device in st.session_state.selected_devices:
                ip = device["ip"]
                name = device["name"] or ip
                
                with st.spinner(f"Getting models for {name}..."):
                    models_result, _ = list_models(ip, port, name)
                    if models_result:
                        st.session_state.models_info[ip] = models_result.get("models", [])
        
        # Display available models for each device
        if st.session_state.models_info:
            st.subheader("Available Models")
            
            for device in st.session_state.selected_devices:
                ip = device["ip"]
                name = device["name"] or ip
                
                if ip in st.session_state.models_info:
                    st.write(f"**{name}** ({ip}):")
                    
                    if st.session_state.models_info[ip]:
                        model_data = []
                        for model in st.session_state.models_info[ip]:
                            size_kb = model.get('size', 'unknown')
                            if isinstance(size_kb, int):
                                size_gb = size_kb / (10 ** 9)  # Convert B to GB
                                size_display = f"{size_gb:.1f} GB"
                            else:
                                size_display = "unknown"
                            
                            model_data.append({
                                "Model": model["name"],
                                "Size": size_display
                            })
                        
                        st.dataframe(pd.DataFrame(model_data))
                    else:
                        st.write("No models found on this device.")
        
        # Model operations
        st.subheader("Model Operations")
        
        # Download models
        with st.expander("Download Models"):
            # Load models from model list file
            models_from_file = load_model_list(model_list_path)
            
            # Manual model input
            manual_model = st.text_input("Model Name", key="download_model_name")
            
            # Load from file checkbox
            use_model_list = st.checkbox("Use Model List File", key="use_model_list")
            
            if use_model_list and models_from_file:
                st.write(f"Models from file: {', '.join(models_from_file)}")
                models_to_download = models_from_file
            else:
                models_to_download = [manual_model] if manual_model else []
            
            # Execute download
            if st.button("Download Models", key="download_models") and models_to_download:
                download_results = []
                
                for device in st.session_state.selected_devices:
                    ip = device["ip"]
                    name = device["name"] or ip
                    
                    for model in models_to_download:
                        with st.spinner(f"Downloading {model} on {name}..."):
                            try:
                                download_model(ip, port, model, name)
                                download_results.append({
                                    "Device": name,
                                    "Model": model,
                                    "Status": "Success",
                                    "Message": f"Model '{model}' downloaded successfully on {name}."
                                })
                            except Exception as e:
                                download_results.append({
                                    "Device": name,
                                    "Model": model,
                                    "Status": "Failed",
                                    "Message": str(e)
                                })
                
                st.subheader("Download Results")
                st.dataframe(pd.DataFrame(download_results))
        
        # Delete models
        with st.expander("Delete Models"):
            if not st.session_state.models_info:
                st.warning("Please refresh available models first.")
            else:
                # Get all unique models across devices
                all_models = set()
                for models in st.session_state.models_info.values():
                    for model in models:
                        all_models.add(model["name"])
                
                if all_models:
                    models_to_delete = st.multiselect("Select Models to Delete", options=list(all_models), key="models_to_delete")
                    delete_all = st.checkbox("Delete All Models", key="delete_all_models")
                    
                    if st.button("Execute Deletion", key="execute_deletion"):
                        delete_results = []
                        
                        for device in st.session_state.selected_devices:
                            ip = device["ip"]
                            name = device["name"] or ip
                            
                            if delete_all:
                                # Delete all models on the device
                                if ip in st.session_state.models_info:
                                    for model in st.session_state.models_info[ip]:
                                        with st.spinner(f"Deleting {model['name']} from {name}..."):
                                            try:
                                                delete_model(ip, port, model["name"], name)
                                                delete_results.append({
                                                    "Device": name,
                                                    "Model": model["name"],
                                                    "Status": "Success",
                                                    "Message": f"Model '{model['name']}' deleted successfully from {name}."
                                                })
                                            except Exception as e:
                                                delete_results.append({
                                                    "Device": name,
                                                    "Model": model["name"],
                                                    "Status": "Failed",
                                                    "Message": str(e)
                                                })
                            elif models_to_delete:
                                # Delete selected models
                                for model_name in models_to_delete:
                                    with st.spinner(f"Deleting {model_name} from {name}..."):
                                        try:
                                            delete_model(ip, port, model_name, name)
                                            delete_results.append({
                                                "Device": name,
                                                "Model": model_name,
                                                "Status": "Success",
                                                "Message": f"Model '{model_name}' deleted successfully from {name}."
                                            })
                                        except Exception as e:
                                            delete_results.append({
                                                "Device": name,
                                                "Model": model_name,
                                                "Status": "Failed",
                                                "Message": str(e)
                                            })
                        
                        st.subheader("Deletion Results")
                        st.dataframe(pd.DataFrame(delete_results))
                else:
                    st.warning("No models available on selected devices.")

# Tab 3: Performance Testing
with tab3:
    st.header("Performance Testing")
    
    if not st.session_state.selected_devices:
        st.warning("Please select devices in the Device Selection tab first.")
    else:
        # Get models from all devices
        if "models_for_testing" not in st.session_state:
            st.session_state.models_for_testing = set()
            
        # Refresh button for models
        if st.button("Refresh Models for Testing", key="refresh_test_models"):
            st.session_state.models_for_testing = set()
            for device in st.session_state.selected_devices:
                ip = device["ip"]
                name = device["name"] or ip
                
                models_result, _ = list_models(ip, port, name)
                if models_result:
                    for model in models_result.get("models", []):
                        st.session_state.models_for_testing.add(model["name"])
        
        # Model selection
        if st.session_state.models_for_testing:
            selected_model = st.selectbox("Select Model", options=list(st.session_state.models_for_testing), key="model_selectbox")
        else:
            selected_model = st.text_input("Model Name", key="test_model_name")
        
        # Prompt selection
        st.subheader("Prompts")
        
        prompts_source = st.radio("Prompt Source", ["Single Prompt", "From Prompt List"], key="prompt_source")
        
        test_prompts = []
        
        if prompts_source == "Single Prompt":
            single_prompt = st.text_area("Enter Prompt", height=150, key="single_prompt")
            if single_prompt:
                test_prompts = [single_prompt]
        else:
            prompts_from_file = load_prompt_list(prompt_list_path)
            if prompts_from_file:
                st.write(f"Loaded {len(prompts_from_file)} prompts from file")
                selected_prompts = st.multiselect("Select Prompts", options=prompts_from_file, key="prompt_multiselect")
                
                if st.checkbox("Use All Prompts", key="use_all_prompts"):
                    test_prompts = prompts_from_file
                else:
                    test_prompts = selected_prompts
            else:
                st.warning("No prompts found in the prompt list file")
        
        # Stream option
        stream_response = st.checkbox("Stream Response", value=False, key="stream_response")
        
        # Run performance test
        if st.button("Run Performance Test", key="run_test") and selected_model and test_prompts:
            st.session_state.results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_steps = len(st.session_state.selected_devices) * len(test_prompts)
            step = 0
            
            for device in st.session_state.selected_devices:
                ip = device["ip"]
                name = device["name"] or ip
                
                device_results = {
                    "device": name or ip,
                    "ip": ip,
                    "results": []
                }
                
                # Check if model exists on this device
                models_result, _ = list_models(ip, port, name)
                if models_result:
                    available_models = [m['name'] for m in models_result.get('models', [])]
                    if selected_model not in available_models:
                        st.warning(f"Model '{selected_model}' not found on {name}")
                        continue
                
                for i, prompt in enumerate(test_prompts):
                    status_text.write(f"Processing prompt {i+1}/{len(test_prompts)} on {name}...")
                    
                    if stream_response:
                        response_container = st.empty()
                        response_container.write(f"Streaming response from {name}...")
                        
                        # Custom implementation for streaming in Streamlit
                        response_text = ""
                        try:
                            # In a real streaming scenario, this would use SSE or other streaming approach
                            # For demonstration, we're showing a non-streaming fallback
                            response = generate_text(ip, port, selected_model, prompt, False, name)
                            if response:
                                response_text = response.get('response', '')
                                response_container.write(response_text)
                                
                                # Record results even in streaming mode
                                device_results["results"].append({
                                    'prompt': prompt,
                                    'response': response_text,
                                    'total_duration': response.get('total_duration', 0)
                                })
                        except Exception as e:
                            response_container.error(f"Error: {str(e)}")
                    else:
                        response = generate_text(ip, port, selected_model, prompt, False, name)
                        if response:
                            device_results["results"].append({
                                'prompt': prompt,
                                'response': response.get('response', ''),
                                'total_duration': response.get('total_duration', 0)
                            })
                    
                    step += 1
                    progress_bar.progress(step / total_steps)
                
                if device_results["results"]:
                    st.session_state.results.append(device_results)
            
            status_text.write("Performance test completed!")
            progress_bar.progress(1.0)

# Tab 4: Results
with tab4:
    st.header("Performance Results")
    
    if not st.session_state.results:
        st.warning("No test results available. Run a performance test first.")
    else:
        # Summary metrics
        st.subheader("Summary")
        
        summary_data = []
        for device_result in st.session_state.results:
            device = device_result['device']
            avg_duration = sum(r['total_duration'] for r in device_result['results']) / len(device_result['results']) / 1000000000
            min_duration = min(r['total_duration'] for r in device_result['results']) / 1000000000
            max_duration = max(r['total_duration'] for r in device_result['results']) / 1000000000
            
            summary_data.append({
                "Device": device,
                "Prompts Processed": len(device_result['results']),
                "Avg Response Time (s)": round(avg_duration, 2),
                "Min Response Time (s)": round(min_duration, 2),
                "Max Response Time (s)": round(max_duration, 2)
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)
        
        # Plot results
        if len(summary_data) > 0:
            st.subheader("Performance Comparison")
            
            fig = px.bar(
                summary_df, 
                x="Device", 
                y="Avg Response Time (s)", 
                error_y=[d["Max Response Time (s)"] - d["Avg Response Time (s)"] for d in summary_data],
                title="Average Response Time by Device"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results
        st.subheader("Detailed Results")
        for device_idx, device_result in enumerate(st.session_state.results):
            with st.expander(f"Device: {device_result['device']}"):
                for prompt_idx, prompt_result in enumerate(device_result['results']):
                    st.write(f"**Prompt {prompt_idx + 1}**: {prompt_result['prompt'][:100]}...")
                    st.write(f"**Response Time**: {prompt_result['total_duration']/1000000000:.2f} seconds")
                    
                    # Show response in a text area
                    st.text_area(
                        f"Response {prompt_idx + 1}",
                        prompt_result['response'],
                        height=150,
                        key=f"response_{device_idx}_{prompt_idx}"
                    )
                    st.markdown("---")
        
        # Export results
        if st.button("Export Results", key="export_results"):
            # Prepare data for export
            export_data = {
                "summary": summary_data,
                "detailed_results": st.session_state.results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Create JSON string
            json_str = json.dumps(export_data, indent=2)
            
            # Provide download link
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"llm_performance_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_json"
            )

# Footer
st.markdown("---")
st.caption("LLM Device Performance Report Tool") 