#!/bin/bash
# Build script for Render
set -e

echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel

# Install compatible versions for model compatibility
pip install tensorflow-cpu==2.13.0
pip install keras==2.13.0
pip install fastapi==0.110.0
pip install uvicorn[standard]==0.27.0
pip install pillow==10.2.0
pip install python-multipart==0.0.6
pip install numpy==1.24.3
pip install pydantic==2.5.3

echo "✓ Build complete"
