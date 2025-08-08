#!/bin/bash

echo "🛑 Stopping Candidate Recommendation Engine..."
echo "=============================================="

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
            kill -15 $pid 2>/dev/null || kill -9 $pid 2>/dev/null || true
        done
        sleep 2
        
        # Double check if any processes are still running
        local remaining_pids=$(lsof -ti:$port 2>/dev/null)
        if [ ! -z "$remaining_pids" ]; then
            echo "🔨 Force killing remaining processes..."
            for pid in $remaining_pids; do
                kill -9 $pid 2>/dev/null || true
            done
        fi
        
        echo "✅ Port $port is now available"
    else
        echo "✅ No processes found on port $port"
    fi
}

# Also kill any streamlit processes
echo "🔍 Stopping all Streamlit processes..."
pkill -f streamlit 2>/dev/null || true

# Kill common Streamlit ports
for port in 8501 8502 8503 8504 8505; do
    kill_port_processes $port
done

echo "✅ All processes stopped successfully!"
echo "💡 You can now run ./run_local.sh to start fresh"
