#!/bin/bash
set -e

# Start DeepAAS
deepaas-run --listen-ip 0.0.0.0 --listen-port 5000 &

# Wait for DeepAAS to start
sleep 40

# Launch Gradio UI
python3 ai4-ai4life/UI/launch.py --api_url http://0.0.0.0:5000/ --ui_port 8000