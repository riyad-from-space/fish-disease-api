#!/bin/bash

# Fish Disease Detection - Run Script

# Activate virtual environment
source venv/bin/activate

# Run the application
echo "🐟 Starting Fish Disease Detection API..."
echo "🌐 Server will be available at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py
