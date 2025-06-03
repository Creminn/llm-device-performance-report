# LLM Device Performance Report - Streamlit App

This is a Streamlit web interface for the LLM Device Performance Report tool. It provides a user-friendly way to interact with Ollama instances running on various devices for model management and performance testing.

## Features

- Connect to multiple devices running Ollama
- Manage models (list, download, delete)
- Run performance tests with various prompts
- Visualize and compare results with charts
- Export performance data as JSON

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Configuration

The app uses the same configuration files as the command-line tool:

- `ip_list.json` - Maps device names to IP addresses
- `model_list.json` - Lists models to download
- `prompt_list.json` - Contains prompts for performance testing

These files should be in the root directory of the project (parent directory of the app folder).

## Usage

1. **Device Selection**: Connect to Ollama instances running on various devices
2. **Model Management**: List, download, and delete models on connected devices
3. **Performance Testing**: Run prompts on selected models and measure performance
4. **Results**: View performance data with visualizations and export results

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Plotly
- Requests
- Ollama running on target devices 