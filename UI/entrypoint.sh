#!/bin/bash
# Start DeepAAS in background
deepaas-run --listen-ip 0.0.0.0 --listen-port 5000 &

# Wait briefly to ensure DeepAAS is running
sleep 60
# Start Gradio UI, pointing to DeepAAS endpoint
python3 ai4-ai4life//UI/launch.py --api_url http://0.0.0.0:5000/ --ui_port 8000
