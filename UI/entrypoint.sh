#!/bin/bash
# Start DeepAAS in background
deepaas-run --listen-ip 0.0.0.0 --listen-port 5000 &

 
# Wait for DeepAAS to start
until curl --output /dev/null --silent --head --fail http://0.0.0.0:5000; do
    echo "Waiting for DeepAAS to start..."
    sleep 2
done
# Get the absolute script directory
SCRIPT_DIR=$(readlink -f "$(dirname "$0")")
# Remove the trailing /UI if present
SCRIPT_DIR=${SCRIPT_DIR%/UI}
echo "Script directory is: $SCRIPT_DIR"



# Start Gradio UI, pointing to DeepAAS endpoint
python3 "/srv/ai4-ai4life/UI/launch.py" \
    --api_url "http://0.0.0.0:5000/" \
    --ui_port 80
