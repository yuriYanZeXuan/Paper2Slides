#!/bin/bash
# Starts the Gemini Proxy service
# Port is hardcoded to 51958 in the python script

export PYTHONPATH=$PYTHONPATH:.
echo "Starting Gemini Proxy..."
python3 Paper2Slides/gemini_proxy.py

