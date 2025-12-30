#!/bin/bash

# Fish Disease Detection API - ngrok Public URL Setup
# This creates a public HTTPS URL for your local server

echo "🚀 Starting ngrok tunnel for Fish Disease Detection API..."
echo ""
echo "⚠️  IMPORTANT: You need to sign up for a free ngrok account first!"
echo "   1. Go to: https://dashboard.ngrok.com/signup"
echo "   2. Sign up (free)"
echo "   3. Get your authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken"
echo "   4. Run: ngrok authtoken YOUR_TOKEN_HERE"
echo ""
read -p "Have you completed the setup above? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo "✅ Starting ngrok tunnel on port 8000..."
    echo ""
    echo "📝 Your public URL will appear below:"
    echo "   Use the HTTPS URL in your Flutter app!"
    echo ""
    
    # Start ngrok
    ngrok http 8000
else
    echo ""
    echo "Please complete the ngrok setup first, then run this script again."
    echo ""
fi
