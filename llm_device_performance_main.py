#!/usr/bin/env python3
"""
Script to send requests to Ollama running on a Device
"""

import requests
import json
import argparse
import socket
import os

def get_local_ip():
    """
    Get the local IP address of the machine
    """
    try:
        # Create a socket to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"  # Fallback to localhost

def load_ip_list():
    """
    Load device:IP map from ip_list.json file
    Returns a dictionary mapping device names to IPs
    """
    try:
        if os.path.exists('ip_list.json'):
            with open('ip_list.json', 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading IP list: {e}")
    return {}

def get_device_name(ip, device_map):
    """
    Get device name for an IP address
    Returns the device name if found, otherwise the IP address itself
    """
    for device, device_ip in device_map.items():
        if device_ip == ip:
            return device
    return None

def resolve_device_or_ip(device_or_ip, device_map):
    """
    Resolves a device name or IP to an IP address
    If device_or_ip is a device name in the map, returns its IP
    If device_or_ip is an IP address (not in the map), returns it directly
    """
    # Check if it's a device name in our map
    if device_or_ip in device_map:
        return device_map[device_or_ip]
    
    # Otherwise, treat it as an IP address
    return device_or_ip

def load_model_list(filename='model_list.json'):
    """
    Load model list from a JSON file
    Returns empty list if file doesn't exist or is invalid
    
    Parameters:
    - filename: Name of the JSON file containing the model list
    """
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                return data.get('models', [])
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading model list from {filename}: {e}")
    return []

def load_prompt_list(filename='prompt_list.json'):
    """
    Load prompt list from a JSON file
    Returns empty list if file doesn't exist or is invalid
    
    Parameters:
    - filename: Name of the JSON file containing the prompt list
    """
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                return data.get('prompts', [])
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading prompt list from {filename}: {e}")
    return []

def generate_text(ip_address, port, model, prompt, stream=False, device_name=None):
    """
    Send a text generation request to Ollama
    
    Parameters:
    - ip_address: Device IP address
    - port: Ollama API port
    - model: Name of the model to use (e.g., "tinyllama", "llama2")
    - prompt: Text prompt to send
    - stream: Whether to stream the response
    - device_name: Optional device name for display
    
    Returns:
    - Response from Ollama
    """
    url = f"http://{ip_address}:{port}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    device_display = f"{device_name} ({ip_address})" if device_name else ip_address
    
    try:
        if stream:
            # Handle streaming response
            print(f"Streaming response from {device_display}:")
            with requests.post(url, data=json.dumps(payload), headers=headers, stream=True) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            print(chunk.get('response', ''), end='', flush=True)
                            if chunk.get('done', False):
                                print()  # Add newline at the end
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON: {line}")
        else:
            # Handle regular response
            response = requests.post(url, data=json.dumps(payload), headers=headers)
            response.raise_for_status()
            
            result = response.json()
            return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama at {device_display}: {e}")
        return None

def list_models(ip_address, port, device_name=None):
    """
    List available models on the Ollama server
    
    Parameters:
    - ip_address: Device IP address
    - port: Ollama API port
    - device_name: Optional device name for display
    
    Returns:
    - List of available models
    """
    url = f"http://{ip_address}:{port}/api/tags"
    
    device_display = f"{device_name} ({ip_address})" if device_name else ip_address
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        models = response.json()
        return models, device_display
    
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama at {device_display}: {e}")
        return None, device_display

def try_devices(devices_or_ips, port, model, prompt, stream=False, device_map=None):
    """
    Try connecting to a list of devices/IPs until successful
    
    Parameters:
    - devices_or_ips: List of device names or IP addresses to try
    - port: Ollama API port
    - model: Name of the model to use
    - prompt: Text prompt to send
    - stream: Whether to stream the response
    - device_map: Dictionary mapping device names to IPs
    """
    if device_map is None:
        device_map = {}
    
    for device_or_ip in devices_or_ips:
        ip = resolve_device_or_ip(device_or_ip, device_map)
        device_name = get_device_name(ip, device_map)
        
        device_display = f"{device_name} ({ip})" if device_name else ip
        print(f"Trying to connect to {device_display}...")
        
        if prompt:
            response = generate_text(ip, port, model, prompt, stream, device_name)
            if response:
                return response
        else:
            models, _ = list_models(ip, port, device_name)
            if models:
                return models, device_display
    return None

def download_model(ip_address, port, model, device_name=None):
    """
    Download a new model to the Ollama server.

    Parameters:
    - ip_address: Device IP address
    - port: Ollama API port
    - model: Name of the model to download
    - device_name: Optional device name for display
    """
    url = f"http://{ip_address}:{port}/api/pull"
    
    payload = {
        "model": model
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    device_display = f"{device_name} ({ip_address})" if device_name else ip_address
    
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()
        print(f"Model '{model}' downloaded successfully on {device_display}.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model '{model}' on {device_display}: {e}")

def delete_model(ip_address, port, model, device_name=None):
    """
    Delete a model from the Ollama server.

    Parameters:
    - ip_address: Device IP address
    - port: Ollama API port
    - model: Name of the model to delete
    - device_name: Optional device name for display
    """
    url = f"http://{ip_address}:{port}/api/delete"
    
    payload = {
        "name": model
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    device_display = f"{device_name} ({ip_address})" if device_name else ip_address
    
    try:
        # First check if model exists
        models_result, _ = list_models(ip_address, port, device_name)
        if models_result:
            available_models = [m['name'] for m in models_result.get('models', [])]
            if model not in available_models:
                print(f"Warning: Model '{model}' not found on {device_display}, skipping deletion")
                return
        
        # If model exists, delete it
        response = requests.delete(url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()
        print(f"Model '{model}' deleted successfully from {device_display}.")
    except requests.exceptions.RequestException as e:
        print(f"Error deleting model '{model}' from {device_display}: {e}")

def process_prompt_list(devices_or_ips, port, model, prompts, stream=False, device_map=None):
    """
    Process a list of prompts and send them to specified devices/IPs
    
    Parameters:
    - devices_or_ips: List of device names or IP addresses
    - port: Ollama API port
    - model: Name of the model to use
    - prompts: List of prompts to process
    - stream: Whether to stream the response
    - device_map: Dictionary mapping device names to IPs
    """
    if device_map is None:
        device_map = {}
    
    results = []
    
    for device_or_ip in devices_or_ips:
        ip = resolve_device_or_ip(device_or_ip, device_map)
        device_name = get_device_name(ip, device_map)
        device_display = f"{device_name} ({ip})" if device_name else ip
        
        # Check if model exists on this device before proceeding
        models_result, _ = list_models(ip, port, device_name)
        if models_result:
            available_models = [m['name'] for m in models_result.get('models', [])]
            if model not in available_models:
                print(f"Error: Model '{model}' not found on {device_display}")
                print(f"Available models: {', '.join(available_models)}")
                continue
        else:
            print(f"Error: Could not retrieve model list from {device_display}")
            continue
        
        print(f"\nProcessing prompts on {device_display}:")
        device_results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\nPrompt {i}/{len(prompts)}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
            
            if stream:
                print(f"Streaming response from {device_display}:")
                generate_text(ip, port, model, prompt, True, device_name)
                # No results to collect when streaming
            else:
                response = generate_text(ip, port, model, prompt, False, device_name)
                if response:
                    print(f"Response summary: {response.get('response', 'No response')[:50]}...")
                    print(f"Processing time: {response.get('total_duration', 0)/1000000000:.2f} seconds")
                    device_results.append({
                        'prompt': prompt,
                        'response': response.get('response', ''),
                        'total_duration': response.get('total_duration', 0)
                    })
                else:
                    print(f"Failed to get response from {device_display}")
        
        if not stream and device_results:
            results.append({
                'device': device_name or ip,
                'ip': ip,
                'results': device_results
            })
    
    # Return overall results if not streaming
    if not stream:
        return results
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send requests to Ollama running on a Device")
    
    # Device/IP selection arguments
    device_group = parser.add_argument_group('Device Selection')
    device_group.add_argument("--device", help="Device name from ip_list.json")
    device_group.add_argument("--ip", help="Single Device IP address (overrides device and ip_list.json)")
    device_group.add_argument("--devices", nargs="+", help="List of device names from ip_list.json")
    device_group.add_argument("--ips", nargs="+", help="List of Device IP addresses (overrides devices and ip_list.json)")
    device_group.add_argument("--all-devices", action="store_true", help="Use all devices from ip_list.json")
    
    # Operation arguments
    parser.add_argument("--port", default=11434, type=int, help="Ollama API port")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--model", help="Model to use for generation")
    parser.add_argument("--prompt", help="Text prompt")
    parser.add_argument("--from-prompt-list", action="store_true", help="Use all prompts from prompt_list.json")
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    
    # Download arguments
    download_group = parser.add_argument_group('Model Download Options')
    download_group.add_argument("--download-model", help="Specify a single model to download")
    download_group.add_argument("--download-models", nargs="+", help="Specify multiple models to download")
    download_group.add_argument("--from-model-list", action="store_true", help="Download models from model_list.json")
    
    # Delete arguments
    delete_group = parser.add_argument_group('Model Delete Options')
    delete_group.add_argument("--delete-model", help="Specify a single model to delete")
    delete_group.add_argument("--delete-models", nargs="+", help="Specify multiple models to delete")
    delete_group.add_argument("--delete-all-models", action="store_true", help="Delete all models from the specified devices")
    
    args = parser.parse_args()
    
    # Load the device:ip mapping
    device_map = load_ip_list()
    
    # Determine which devices/IPs to use
    devices_or_ips = []
    
    if args.ip:
        # Single IP has highest priority
        devices_or_ips = [args.ip]
        # Check if this IP is in our device map and display device name if found
        device_name = get_device_name(args.ip, device_map)
        if device_name:
            print(f"IP {args.ip} belongs to device: {device_name}")
    elif args.device:
        # Single device name
        if args.device in device_map:
            devices_or_ips = [args.device]
        else:
            print(f"Device '{args.device}' not found in ip_list.json")
            exit(1)
    elif args.ips:
        # List of IPs
        devices_or_ips = args.ips
        # Check if any of these IPs are in our device map
        for ip in args.ips:
            device_name = get_device_name(ip, device_map)
            if device_name:
                print(f"IP {ip} belongs to device: {device_name}")
    elif args.devices:
        # List of device names
        devices_or_ips = []
        for device in args.devices:
            if device in device_map:
                devices_or_ips.append(device)
            else:
                print(f"Warning: Device '{device}' not found in ip_list.json, skipping")
    elif args.all_devices:
        # Use all devices from the file
        if device_map:
            devices_or_ips = list(device_map.keys())
            print(f"Using all devices from ip_list.json: {', '.join(devices_or_ips)}")
        else:
            print("No devices found in ip_list.json, falling back to localhost")
            devices_or_ips = ["127.0.0.1"]
    else:
        # No device/IP specified, fall back to localhost
        devices_or_ips = ["127.0.0.1"]
        print("No device or IP specified, using localhost (127.0.0.1)")
    
    # Determine which operation to perform
    is_download_operation = args.download_model or args.download_models or args.from_model_list
    is_delete_operation = args.delete_model or args.delete_models or args.delete_all_models
    
    if args.list_models:
        result = try_devices(devices_or_ips, args.port, None, None, device_map=device_map)
        if result:
            models, device_display = result
            print(f"Available models on {device_display}:")
            for model in models.get('models', []):
                size_kb = model.get('size', 'unknown')
                if isinstance(size_kb, int):  # Ensure size is an integer
                    size_gb = size_kb / (10 ** 9)  # Convert B to GB
                    print(f"- {model['name']} (Size: {size_gb:.1f} GB)")
                else:
                    print(f"- {model['name']} (Size: unknown)")
        else:
            print("No models found or error connecting to Ollama")
    
    elif args.from_prompt_list:
        if not args.model:
            print("Error: --model must be specified when using --from-prompt-list")
            exit(1)
            
        prompts = load_prompt_list('prompt_list.json')
        if not prompts:
            print("Error: No prompts found in prompt_list.json or error loading file")
            exit(1)
            
        print(f"Processing {len(prompts)} prompts from prompt_list.json using model {args.model}")
        results = process_prompt_list(devices_or_ips, args.port, args.model, prompts, args.stream, device_map)
        
        if results and not args.stream:
            print("\nSummary of results:")
            for device_result in results:
                device = device_result['device']
                avg_duration = sum(r['total_duration'] for r in device_result['results']) / len(device_result['results']) / 1000000000
                print(f"{device}: Processed {len(device_result['results'])} prompts, average response time: {avg_duration:.2f} seconds")
    
    elif args.prompt:
        if not args.model:
            print("Error: --model must be specified when using --prompt")
            exit(1)
            
        print(f"Sending prompt to {args.model}...")
        
        # Check if model exists on at least one device before proceeding
        model_exists_on_any_device = False
        for device_or_ip in devices_or_ips:
            ip = resolve_device_or_ip(device_or_ip, device_map)
            device_name = get_device_name(ip, device_map)
            
            # Check if model exists on this device
            models_result, _ = list_models(ip, args.port, device_name)
            if models_result:
                available_models = [m['name'] for m in models_result.get('models', [])]
                if args.model in available_models:
                    model_exists_on_any_device = True
                    break
        
        if not model_exists_on_any_device:
            print(f"Error: Model '{args.model}' not found on any specified device")
            exit(1)
            
        if args.stream:
            try_devices(devices_or_ips, args.port, args.model, args.prompt, True, device_map)
        else:
            response = try_devices(devices_or_ips, args.port, args.model, args.prompt, False, device_map)
            if response:
                print("Response:")
                print(response.get('response', 'No response'))
                print(f"Total duration: {response.get('total_duration', 0)/1000000000:.2f} seconds")
    
    elif is_download_operation:
        # Models to download
        models_to_download = []
        
        # Check each download source in priority order
        if args.download_models:
            # List of models has highest priority
            models_to_download = args.download_models
            print(f"Downloading models: {', '.join(models_to_download)}")
        
        elif args.from_model_list:
            # Models from model_list.json
            models_to_download = load_model_list('model_list.json')
            if models_to_download:
                print(f"Downloading models from model_list.json: {', '.join(models_to_download)}")
            else:
                print("No models found in model_list.json or error loading file.")
                exit(1)
        
        elif args.download_model:
            # Single model specified with --download-model
            models_to_download = [args.download_model]
            print(f"Downloading model: {args.download_model}")
        
        # Download all models to all specified devices
        for device_or_ip in devices_or_ips:
            ip = resolve_device_or_ip(device_or_ip, device_map)
            device_name = get_device_name(ip, device_map)
            
            for model in models_to_download:
                download_model(ip, args.port, model, device_name)
    
    elif is_delete_operation:
        # Models to delete
        models_to_delete = []
        
        # Check each delete source in priority order
        if args.delete_models:
            # List of models has highest priority
            models_to_delete = args.delete_models
            print(f"Deleting models: {', '.join(models_to_delete)}")
        
        elif args.delete_model:
            # Single model specified with --delete-model
            models_to_delete = [args.delete_model]
            print(f"Deleting model: {args.delete_model}")
        
        elif args.delete_all_models:
            # Delete all models on each device
            print("Deleting all models from specified devices")
            for device_or_ip in devices_or_ips:
                ip = resolve_device_or_ip(device_or_ip, device_map)
                device_name = get_device_name(ip, device_map)
                device_display = f"{device_name} ({ip})" if device_name else ip
                
                models_result, _ = list_models(ip, args.port, device_name)
                if models_result:
                    all_models = [m['name'] for m in models_result.get('models', [])]
                    if all_models:
                        print(f"Found {len(all_models)} models on {device_display}: {', '.join(all_models)}")
                        for model in all_models:
                            delete_model(ip, args.port, model, device_name)
                    else:
                        print(f"No models found on {device_display}")
                else:
                    print(f"Error: Could not retrieve model list from {device_display}")
        
        # If specific models to delete were provided
        if models_to_delete:
            for device_or_ip in devices_or_ips:
                ip = resolve_device_or_ip(device_or_ip, device_map)
                device_name = get_device_name(ip, device_map)
                
                for model in models_to_delete:
                    delete_model(ip, args.port, model, device_name)
    
    else:
        parser.print_help()