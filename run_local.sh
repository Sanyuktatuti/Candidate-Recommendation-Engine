#!/bin/bash

echo "🚀 Starting Candidate Recommendation Engine locally..."
echo "======================================================="

# Function to kill processes on specific ports
kill_port_processes() {
    local port=$1
    echo "🔍 Checking for processes on port $port..."
    
    # Find and kill processes using the port
    local pids=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pids" ]; then
        echo "🛑 Found processes on port $port. Stopping them..."
        for pid in $pids; do
            echo "   Killing PID: $pid"
            kill -9 $pid 2>/dev/null || true
        done
        sleep 2
        echo "✅ Port $port is now available"
    else
        echo "✅ Port $port is available"
    fi
}

# Function to find available port
find_available_port() {
    local start_port=$1
    local port=$start_port
    
    while [ $port -lt $((start_port + 100)) ]; do
        if ! lsof -i:$port >/dev/null 2>&1; then
            echo $port
            return
        fi
        port=$((port + 1))
    done
    
    # If no port found in range, return the start port
    echo $start_port
}

# Set default port
DEFAULT_PORT=8501
PORT=${1:-$DEFAULT_PORT}

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment. Please check Python installation."
        exit 1
    fi
    echo "✅ Virtual environment created successfully"
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip first
echo "🔧 Ensuring pip is up to date..."
pip install --upgrade pip >/dev/null 2>&1

# Check if requirements are installed
echo "🔍 Checking dependencies..."
python -c "import streamlit, openai, pandas, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies. Please check requirements.txt"
        exit 1
    fi
    echo "✅ Dependencies installed successfully"
else
    echo "✅ All dependencies are installed"
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. The app will work but without your premium APIs."
    echo "   Copy .env.example to .env and add your API keys for best experience."
    echo ""
fi

# Handle port conflicts
kill_port_processes $PORT

# Double-check port availability and find alternative if needed
if lsof -i:$PORT >/dev/null 2>&1; then
    echo "⚠️  Port $PORT is still busy. Finding alternative..."
    PORT=$(find_available_port $PORT)
    echo "📡 Using port $PORT instead"
fi

# Check if streamlit_app.py exists
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ streamlit_app.py not found in current directory"
    echo "   Please ensure you're in the correct project directory"
    exit 1
fi

# Create cleanup function for graceful shutdown
cleanup() {
    echo ""
    echo "🛑 Shutting down gracefully..."
    kill_port_processes $PORT
    echo "✅ Cleanup complete. Goodbye!"
    exit 0
}

# Set trap for graceful shutdown
trap cleanup SIGINT SIGTERM

# Start the app
echo "🎯 Starting Streamlit app..."
echo "📱 App will open in your browser at: http://localhost:$PORT"
echo "🛑 Press Ctrl+C to stop the app"
echo "🔄 Starting in 3 seconds..."
echo ""

# Give a moment for any cleanup to complete
sleep 3

# Start Streamlit with error handling
streamlit run streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port $PORT \
    --server.headless true \
    --server.runOnSave true \
    --browser.gatherUsageStats false

# If we reach here, streamlit exited
echo ""
echo "📱 Streamlit app has stopped"
cleanup
