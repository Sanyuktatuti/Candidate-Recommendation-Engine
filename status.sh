#!/bin/bash

# Candidate Recommendation Engine - Status Script
# This script checks the status of all services

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

print_header() {
    echo -e "${PURPLE}📊 $1${NC}"
}

# Function to check if a port is in use
check_port() {
    local port=$1
    local service_name=$2
    
    local pids=$(lsof -ti tcp:$port 2>/dev/null || true)
    
    if [ -n "$pids" ]; then
        print_status "$service_name is running on port $port"
        echo "   PIDs: $pids"
        
        # Check if service is responding
        case $service_name in
            "FastAPI")
                if curl -s -f http://localhost:$port/health > /dev/null 2>&1; then
                    print_status "API health check: PASSED"
                else
                    print_warning "API health check: FAILED"
                fi
                ;;
            "Streamlit")
                if curl -s -f http://localhost:$port > /dev/null 2>&1; then
                    print_status "Streamlit health check: PASSED"
                else
                    print_warning "Streamlit health check: FAILED"
                fi
                ;;
        esac
    else
        print_error "$service_name is NOT running on port $port"
    fi
    
    echo ""
}

# Function to check process files
check_process_files() {
    print_info "Process Files:"
    
    if [ -f "api.pid" ]; then
        local api_pid=$(cat api.pid)
        if kill -0 $api_pid 2>/dev/null; then
            print_status "api.pid exists and process is running (PID: $api_pid)"
        else
            print_warning "api.pid exists but process is not running (stale PID: $api_pid)"
        fi
    else
        print_info "api.pid not found"
    fi
    
    if [ -f "streamlit.pid" ]; then
        local streamlit_pid=$(cat streamlit.pid)
        if kill -0 $streamlit_pid 2>/dev/null; then
            print_status "streamlit.pid exists and process is running (PID: $streamlit_pid)"
        else
            print_warning "streamlit.pid exists but process is not running (stale PID: $streamlit_pid)"
        fi
    else
        print_info "streamlit.pid not found"
    fi
    
    echo ""
}

# Function to check log files
check_logs() {
    print_info "Log Files:"
    
    if [ -f "api.log" ]; then
        local log_size=$(wc -l < api.log)
        print_info "api.log exists ($log_size lines)"
        if [ $log_size -gt 0 ]; then
            print_info "Last 3 lines of api.log:"
            tail -3 api.log | sed 's/^/    /'
        fi
    else
        print_info "api.log not found"
    fi
    
    echo ""
    
    if [ -f "streamlit.log" ]; then
        local log_size=$(wc -l < streamlit.log)
        print_info "streamlit.log exists ($log_size lines)"
        if [ $log_size -gt 0 ]; then
            print_info "Last 3 lines of streamlit.log:"
            tail -3 streamlit.log | sed 's/^/    /'
        fi
    else
        print_info "streamlit.log not found"
    fi
    
    echo ""
}

# Function to show access URLs
show_access_urls() {
    print_info "Access URLs (if services are running):"
    echo "  📱 Streamlit App:      http://localhost:$STREAMLIT_PORT"
    echo "  📚 API Documentation:  http://localhost:$API_PORT/docs"
    echo "  🏥 Health Check:       http://localhost:$API_PORT/health"
    echo ""
}

# Function to show available commands
show_commands() {
    print_info "Available Commands:"
    echo "  🚀 Start Services:     ./run.sh"
    echo "  🛑 Stop Services:      ./stop.sh"
    echo "  📊 Check Status:       ./status.sh"
    echo ""
}

# Main execution
print_header "Candidate Recommendation Engine - Status Check"
echo "================================================================"

# Check services
check_port $API_PORT "FastAPI"
check_port $STREAMLIT_PORT "Streamlit"

# Check process files
check_process_files

# Check logs
check_logs

# Show access URLs
show_access_urls

# Show available commands
show_commands

# Overall status summary
api_running=$(lsof -ti tcp:$API_PORT 2>/dev/null || true)
streamlit_running=$(lsof -ti tcp:$STREAMLIT_PORT 2>/dev/null || true)

if [ -n "$api_running" ] && [ -n "$streamlit_running" ]; then
    print_status "Overall Status: ALL SERVICES RUNNING"
elif [ -n "$api_running" ] || [ -n "$streamlit_running" ]; then
    print_warning "Overall Status: PARTIAL SERVICES RUNNING"
else
    print_error "Overall Status: NO SERVICES RUNNING"
    print_info "Run './run.sh' to start all services"
fi
