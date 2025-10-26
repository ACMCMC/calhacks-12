#!/bin/bash

# PrivAds Web Ad Service Startup Script

echo "🚀 Starting PrivAds Web Ad Service..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Please run this script from the web_ad_service directory"
    exit 1
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Check for required environment variables
if [ -f ".env" ]; then
    echo "📄 Found .env file, loading environment variables..."
    export $(cat .env | grep -v '^#' | xargs)
elif [ -z "$GOOGLE_AI_API_KEY" ]; then
    echo "⚠️  GOOGLE_AI_API_KEY not set. Gemini customization will use fallback mode."
    echo "   To enable Gemini AI:"
    echo "   1. Create a .env file with: GOOGLE_AI_API_KEY='your_key_here'"
    echo "   2. Or set: export GOOGLE_AI_API_KEY='your_key_here'"
fi

# Start the service
echo "🌐 Starting web ad service on http://localhost:8002"
echo "   Press Ctrl+C to stop the service"
echo ""

python3 main.py

