#!/bin/bash

# Fish Disease Detection - Setup Script

echo "🐟 Setting up Fish Disease Detection API..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Place your 'mobilenetv2_fish_final.h5' model file in /Users/riyadafromspace/Documents/"
echo "2. Update DISEASE_CLASSES in main.py to match your model's classes"
echo "3. Run: python main.py"
echo "4. Open http://localhost:8000 in your browser"
echo ""
echo "🚀 To start the server now, run:"
echo "   source venv/bin/activate"
echo "   python main.py"
