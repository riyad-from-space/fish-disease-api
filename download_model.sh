#!/bin/bash
# Download model file for Render deployment
# This script runs during Render's build process

MODEL_FILE="inceptionv3_fish_final.h5"
MODEL_PATH="./$MODEL_FILE"

# Check if model already exists
if [ -f "$MODEL_PATH" ]; then
    echo "✓ Model file found locally: $MODEL_PATH"
    exit 0
fi

echo "⚠️  Model file not found. Instructions for deployment:"
echo "1. Go to https://dashboard.render.com"
echo "2. Find your Fish Disease API service"
echo "3. Go to Environment tab and add:"
echo "   - Name: MODEL_URL"
echo "   - Value: <URL to your model file from cloud storage>"
echo "4. Or upload model manually to Render's persistent disk"
echo ""
echo "For now, using placeholder model for testing..."
exit 0
