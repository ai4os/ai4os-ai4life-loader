#!/bin/bash
# Start DeepAAS in background
deepaas-run --listen-ip 0.0.0.0 --listen-port 5000 &

# Wait briefly to ensure DeepAAS is running
sleep 5
SCRIPT_DIR=$(dirname "$0")
echo "Script directory is: $SCRIPT_DIR"
# Start Gradio UI, pointing to DeepAAS endpoint
python3 "${SCRIPT_DIR}/UI/launch.py" \
    --api_url "http://0.0.0.0:5000/" \
    --ui_port 8080