#!/bin/bash
# Start DeepAAS in background
deepaas-run --listen-ip 0.0.0.0 --listen-port 5000 &

 
 
sleep 30
# Get the absolute script directory
SCRIPT_DIR=$(dirname "$0")
echo "Script directory is: $SCRIPT_DIR"

# Start Gradio UI, pointing to DeepAAS endpoint
python3  "${SCRIPT_DIR}/launch.py" \
    --api_url "http://0.0.0.0:5000/" \
    --ui_port 80
