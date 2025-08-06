#!/usr/bin/env python3
"""
Quick setup validation script for the Candidate Recommendation Engine.
"""
import sys
import importlib.util
from pathlib import Path


def check_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except ImportError:
        return False


def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return Path(file_path).exists()


def main():
    """Run setup validation checks."""
    print("🎯 Candidate Recommendation Engine - Setup Validation")
    print("=" * 60)
    
    checks = [
        ("Python Version", sys.version_info >= (3, 11), "Python 3.11+ required"),
        ("FastAPI", check_import("fastapi"), "pip install fastapi"),
        ("Streamlit", check_import("streamlit"), "pip install streamlit"),
        ("OpenAI", check_import("openai"), "pip install openai"),
        ("FAISS", check_import("faiss"), "pip install faiss-cpu"),
        ("PyPDF2", check_import("PyPDF2"), "pip install PyPDF2"),
        ("Python-docx", check_import("docx"), "pip install python-docx"),
        ("Config File", check_file_exists("config.py"), "config.py not found"),
        ("Main App", check_file_exists("app/main.py"), "app/main.py not found"),
        ("Streamlit App", check_file_exists("streamlit_app.py"), "streamlit_app.py not found"),
        ("Requirements", check_file_exists("requirements.txt"), "requirements.txt not found"),
        ("Docker File", check_file_exists("Dockerfile"), "Dockerfile not found"),
    ]
    
    all_passed = True
    
    for check_name, passed, error_msg in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:<20} {status}")
        
        if not passed:
            print(f"    → {error_msg}")
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 All checks passed! Your setup looks good.")
        print("\nNext steps:")
        print("1. Create .env file with your OpenAI API key")
        print("2. Run: python scripts/dev.py")
        print("3. Or use Docker: ./scripts/deploy.sh")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\nTo install all dependencies:")
        print("pip install -r requirements.txt")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
