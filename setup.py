#!/usr/bin/env python3
"""
Complete setup script for Neo4j Investment Platform
"""
import os
import sys
from pathlib import Path

def setup_project():
    """Set up the complete project"""
    print("🚀 Setting up Neo4j Investment Platform...")
    
    # Create virtual environment
    print("📦 Creating virtual environment...")
    os.system("python -m venv venv")
    
    # Install backend dependencies
    print("📚 Installing backend dependencies...")
    pip_cmd = "venv/bin/pip" if sys.platform != "win32" else "venv\\Scripts\\pip"
    os.system(f"{pip_cmd} install -r backend/requirements.txt")
    
    # Install frontend dependencies
    print("🎨 Installing frontend dependencies...")
    os.chdir("frontend")
    os.system("npm install")
    os.chdir("..")
    
    print("✅ Setup complete!")
    print("\n📋 Next steps:")
    print("1. Copy .env.example to .env and fill in your credentials")
    print("2. Run 'python scripts/init_database.py' to initialize Neo4j")
    print("3. Start the backend: cd backend && uvicorn main:app --reload")
    print("4. Start the frontend: cd frontend && npm run dev")

if __name__ == "__main__":
    setup_project()