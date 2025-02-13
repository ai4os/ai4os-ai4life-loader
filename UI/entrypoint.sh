#!/bin/bash
# Start DeepAAS in background
deepaas-run --listen-ip 0.0.0.0 --listen-port 5000 &

# Wait until DeepAAS is ready
deepaas-run --listen-ip 0.0.0.0 --listen-port 5000 &
# Wait for DeepAAS to start
sleep 40
# Launch Gradio UI
 
# Get the absolute script directory
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "Script directory is: $SCRIPT_DIR"



# Start Gradio UI, pointing to DeepAAS endpoint
python3 "${SCRIPT_DIR}/UI/launch.py" \
    --api_url "http://0.0.0.0:5000/" \
    --ui_port 80
