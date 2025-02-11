#!/bin/bash
# Start DeepAAS in background
deepaas-run --listen-ip 0.0.0.0 --listen-port 5000 &

# Wait until DeepAAS is ready
until curl --output /dev/null --silent --head --fail http://0.0.0.0:5000; do
    echo "Waiting for DeepAAS to start..."
    sleep 2
done

# Get the absolute script directory
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "Script directory is: $SCRIPT_DIR"



# Start Gradio UI, pointing to DeepAAS endpoint
python3 "${SCRIPT_DIR}/UI/launch.py" \
    --api_url "http://0.0.0.0:5000/" \
    --ui_port 80
