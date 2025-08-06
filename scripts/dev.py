#!/usr/bin/env python3
"""
Development server script for running API and Streamlit simultaneously.
"""
import subprocess
import sys
import time
import signal
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_env_file():
    """Check if .env file exists with required variables."""
    env_file = project_root / ".env"
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return False
    
    # Check if OPENAI_API_KEY is set
    with open(env_file, 'r') as f:
        content = f.read()
        if "OPENAI_API_KEY" not in content or "your_openai_api_key_here" in content:
            print("❌ Please set your OpenAI API key in the .env file")
            return False
    
    return True


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, cwd=project_root)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def start_api_server():
    """Start the FastAPI server."""
    print("🚀 Starting API server...")
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "app.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ], cwd=project_root)


def start_streamlit_server():
    """Start the Streamlit server."""
    print("🖥️ Starting Streamlit server...")
    return subprocess.Popen([
        sys.executable, "-m", "streamlit", 
        "run", "streamlit_app.py",
        "--server.address", "0.0.0.0",
        "--server.port", "8501"
    ], cwd=project_root)


def main():
    """Main development server."""
    print("🎯 Candidate Recommendation Engine - Development Server")
    print("=" * 60)
    
    # Check environment
    if not check_env_file():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    processes = []
    
    try:
        # Start API server
        api_process = start_api_server()
        processes.append(api_process)
        
        # Wait a bit for API to start
        print("⏳ Waiting for API server to start...")
        time.sleep(5)
        
        # Start Streamlit server
        streamlit_process = start_streamlit_server()
        processes.append(streamlit_process)
        
        print("\n✅ Both servers are starting up!")
        print("📍 API Documentation: http://localhost:8000/docs")
        print("📍 Streamlit App: http://localhost:8501")
        print("\nPress Ctrl+C to stop both servers")
        
        # Wait for processes
        while True:
            time.sleep(1)
            
            # Check if any process died
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    print(f"❌ Process {i} exited with code {process.returncode}")
                    break
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
    
    finally:
        # Cleanup processes
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        print("✅ All servers stopped")


if __name__ == "__main__":
    main()
