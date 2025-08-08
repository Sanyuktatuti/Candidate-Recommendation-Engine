#!/bin/bash

echo "🚀 Starting Candidate Recommendation Engine locally..."
echo "======================================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please create one first:"
    echo "   python -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Check if requirements are installed
echo "🔍 Checking dependencies..."
python -c "import streamlit, openai, pandas, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. The app will work but without your premium APIs."
    echo "   Copy .env.example to .env and add your API keys for best experience."
fi

# Start the app
echo "🎯 Starting Streamlit app..."
echo "📱 App will open in your browser at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the app"
echo ""

streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
