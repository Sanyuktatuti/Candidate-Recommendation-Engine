#!/bin/bash

# Candidate Recommendation Engine - Stop Script
# This script will stop all running servers

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_PORT=8000
STREAMLIT_PORT=8501

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to kill processes on a specific port
kill_port() {
    local port=$1
    local service_name=$2
    
    print_info "Stopping $service_name on port $port..."
    
    # Find and kill processes using the port
    local pids=$(lsof -ti tcp:$port 2>/dev/null || true)
    
    if [ -n "$pids" ]; then
        echo "$pids" | xargs kill -TERM 2>/dev/null || true
        sleep 2
        
        # Force kill if still running
        pids=$(lsof -ti tcp:$port 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill -9 2>/dev/null || true
            sleep 1
        fi
        
        print_status "$service_name stopped"
    else
        print_info "No $service_name processes found on port $port"
    fi
}

# Function to cleanup PID files and logs
cleanup_files() {
    print_info "Cleaning up process files..."
    
    # Remove PID files
    if [ -f "api.pid" ]; then
        rm -f api.pid
        print_info "Removed api.pid"
    fi
    
    if [ -f "streamlit.pid" ]; then
        rm -f streamlit.pid
        print_info "Removed streamlit.pid"
    fi
    
    print_status "Cleanup completed"
}

# Main execution
echo "🛑 Stopping Candidate Recommendation Engine..."
echo "==============================================="

# Stop services by port
kill_port $API_PORT "FastAPI"
kill_port $STREAMLIT_PORT "Streamlit"

# Also kill by process name patterns
print_info "Stopping any remaining processes..."
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "streamlit run streamlit_app.py" 2>/dev/null || true

# Cleanup files
cleanup_files

echo ""
print_status "All services stopped successfully!"
print_info "You can restart with: ./run.sh"
