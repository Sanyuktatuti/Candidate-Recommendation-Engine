#!/bin/bash

# Deployment script for Candidate Recommendation Engine

set -e

echo "🎯 Candidate Recommendation Engine - Deployment Script"
echo "======================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if .env file exists
if [ ! -f .env ]; then
    print_error ".env file not found!"
    print_info "Please create a .env file with your OpenAI API key:"
    echo "OPENAI_API_KEY=your_api_key_here"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_info "Docker and Docker Compose are available"

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p data logs

# Build and start services
print_info "Building Docker images..."
docker-compose build

print_info "Starting services..."
docker-compose up -d

# Wait for services to be healthy
print_info "Waiting for services to start..."
sleep 10

# Check service health
print_info "Checking service health..."

# Check API health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    print_status "API service is healthy"
else
    print_warning "API service might still be starting up..."
fi

# Check if Streamlit is accessible
if curl -f http://localhost:8501 > /dev/null 2>&1; then
    print_status "Streamlit service is accessible"
else
    print_warning "Streamlit service might still be starting up..."
fi

echo ""
print_status "Deployment completed!"
echo ""
print_info "Access points:"
echo "📍 Streamlit App: http://localhost:8501"
echo "📍 API Documentation: http://localhost:8000/docs"
echo "📍 API Health Check: http://localhost:8000/health"
echo ""
print_info "Useful commands:"
echo "• View logs: docker-compose logs -f"
echo "• Stop services: docker-compose down"
echo "• Restart services: docker-compose restart"
echo "• View status: docker-compose ps"
echo ""
print_warning "Make sure your OpenAI API key is correctly set in the .env file!"

# Optional: Show logs for a few seconds
print_info "Showing recent logs (last 20 lines):"
docker-compose logs --tail=20
