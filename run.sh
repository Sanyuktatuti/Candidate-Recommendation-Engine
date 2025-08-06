#!/bin/bash

# Candidate Recommendation Engine - Launch Script
# This script will start both the FastAPI backend and Streamlit frontend

set -e

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
API_PORT=8000
STREAMLIT_PORT=8501
VENV_PATH=".venv"

# Function to print colored output
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

print_header() {
    echo -e "${PURPLE}🎯 $1${NC}"
}

# Function to kill processes on a specific port
kill_port() {
    local port=$1
    local process_name=$2
    
    print_info "Checking for existing processes on port $port..."
    
    # Find processes using the port
    local pids=$(lsof -ti tcp:$port 2>/dev/null || true)
    
    if [ -n "$pids" ]; then
        print_warning "Found existing $process_name processes on port $port. Killing them..."
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2
        print_status "Cleaned up port $port"
    else
        print_info "Port $port is available"
    fi
}

# Function to check if virtual environment exists
check_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        print_error "Virtual environment not found at $VENV_PATH"
        print_info "Please create a virtual environment first:"
        print_info "python -m venv .venv"
        print_info "source .venv/bin/activate"
        print_info "pip install -r requirements.txt"
        exit 1
    fi
}

# Function to check if .env file exists
check_env_file() {
    if [ ! -f ".env" ]; then
        print_error ".env file not found!"
        print_info "Creating a template .env file..."
        echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
        echo "DEBUG=True" >> .env
        print_warning "Please edit .env and add your OpenAI API key before running again!"
        exit 1
    fi
    
    # Check if API key is set (now optional since we support free mode)
    if grep -q "your_openai_api_key_here" .env; then
        print_warning "⚠️  Please set your OpenAI API key in the .env file!"
        print_info "ℹ️  Edit .env and replace 'your_openai_api_key_here' with your actual API key"
        print_info "💡 Or you can continue with Free Mode (TF-IDF + keyword matching)"
        print_info "🚀 The app will work in Free Mode, but OpenAI provides better quality"
        echo ""
    fi
}

# Function to activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
    print_status "Virtual environment activated"
}

# Function to start API server in background
start_api() {
    print_info "Starting FastAPI server on port $API_PORT..."
    
    nohup uvicorn app.main:app --host 0.0.0.0 --port $API_PORT > api.log 2>&1 &
    local api_pid=$!
    echo $api_pid > api.pid
    
    # Wait a bit for the server to start
    sleep 3
    
    # Check if the process is still running
    if kill -0 $api_pid 2>/dev/null; then
        print_status "FastAPI server started successfully (PID: $api_pid)"
    else
        print_error "Failed to start FastAPI server"
        print_info "Check api.log for error details:"
        tail -10 api.log
        exit 1
    fi
}

# Function to start Streamlit in background
start_streamlit() {
    print_info "Starting Streamlit server on port $STREAMLIT_PORT..."
    
    nohup streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $STREAMLIT_PORT > streamlit.log 2>&1 &
    local streamlit_pid=$!
    echo $streamlit_pid > streamlit.pid
    
    # Wait a bit for the server to start
    sleep 3
    
    # Check if the process is still running
    if kill -0 $streamlit_pid 2>/dev/null; then
        print_status "Streamlit server started successfully (PID: $streamlit_pid)"
    else
        print_error "Failed to start Streamlit server"
        print_info "Check streamlit.log for error details:"
        tail -10 streamlit.log
        exit 1
    fi
}

# Function to check server health
check_health() {
    print_info "Checking server health..."
    
    # Check API health
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f http://localhost:$API_PORT/health > /dev/null 2>&1; then
            print_status "API server is healthy"
            break
        else
            if [ $attempt -eq $max_attempts ]; then
                print_warning "API server health check failed after $max_attempts attempts"
                print_info "API may still be starting up..."
            else
                print_info "Waiting for API server... (attempt $attempt/$max_attempts)"
                sleep 2
            fi
        fi
        ((attempt++))
    done
}

# Function to display access information
show_access_info() {
    echo ""
    print_header "🚀 Candidate Recommendation Engine is Running!"
    echo ""
    print_status "Access Points:"
    echo "  📱 Streamlit App:      http://localhost:$STREAMLIT_PORT"
    echo "  📚 API Documentation:  http://localhost:$API_PORT/docs"
    echo "  🏥 Health Check:       http://localhost:$API_PORT/health"
    echo ""
    print_info "Log Files:"
    echo "  📄 API Logs:           api.log"
    echo "  📄 Streamlit Logs:     streamlit.log"
    echo ""
    print_info "Process Management:"
    echo "  🛑 Stop All:           ./stop.sh"
    echo "  📊 Check Status:       ./status.sh"
    echo ""
    print_warning "Keep this terminal open or run in background with 'nohup ./run.sh &'"
}

# Function to wait for user interrupt
wait_for_interrupt() {
    print_info "Press Ctrl+C to stop all servers..."
    
    # Trap Ctrl+C and cleanup
    trap 'print_info "Shutting down..."; cleanup; exit 0' INT
    
    # Wait indefinitely
    while true; do
        sleep 1
    done
}

# Function to cleanup processes
cleanup() {
    print_info "Cleaning up processes..."
    
    if [ -f "api.pid" ]; then
        local api_pid=$(cat api.pid)
        if kill -0 $api_pid 2>/dev/null; then
            kill $api_pid
            print_status "Stopped API server"
        fi
        rm -f api.pid
    fi
    
    if [ -f "streamlit.pid" ]; then
        local streamlit_pid=$(cat streamlit.pid)
        if kill -0 $streamlit_pid 2>/dev/null; then
            kill $streamlit_pid
            print_status "Stopped Streamlit server"
        fi
        rm -f streamlit.pid
    fi
    
    # Also kill any remaining processes on our ports
    kill_port $API_PORT "API"
    kill_port $STREAMLIT_PORT "Streamlit"
}

# Main execution
main() {
    print_header "Candidate Recommendation Engine - Launch Script"
    echo "================================================================"
    
    # Pre-flight checks
    check_venv
    check_env_file
    
    # Clean up any existing processes
    cleanup
    kill_port $API_PORT "API"
    kill_port $STREAMLIT_PORT "Streamlit"
    
    # Activate virtual environment
    activate_venv
    
    # Start servers
    start_api
    start_streamlit
    
    # Check health
    check_health
    
    # Show access information
    show_access_info
    
    # Wait for user to stop
    wait_for_interrupt
}

# Run main function
main "$@"
