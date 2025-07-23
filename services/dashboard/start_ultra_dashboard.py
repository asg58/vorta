#!/usr/bin/env python3
"""
🚀 VORTA ULTRA DASHBOARD STARTUP SCRIPT
Start the ultra high-grade dashboard server with all dependencies
"""

import os
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn", 
        "pydantic",
        "psutil",
        "torch"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("💡 Installing missing packages...")
        
        try:
            # Install missing packages
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "fastapi[all]", "uvicorn[standard]", "pydantic", "psutil"
            ])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    return True

def start_dashboard():
    """Start the VORTA Ultra Dashboard Server"""
    print("\n🚀 Starting VORTA Ultra High-Grade Dashboard...")
    
    # Change to dashboard directory
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)
    
    # Start the server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "ultra_dashboard_server_v2:app",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--reload",
            "--log-level", "info"
        ])
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard server stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        
        # Fallback - try to run directly
        print("🔄 Trying fallback method...")
        try:
            subprocess.run([sys.executable, "ultra_dashboard_server_v2.py"])
        except Exception as fallback_error:
            print(f"❌ Fallback failed: {fallback_error}")

def main():
    """Main startup function"""
    print("=" * 60)
    print("🚀 VORTA ULTRA HIGH-GRADE DASHBOARD STARTUP")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("ultra_dashboard_server_v2.py").exists():
        print("❌ Dashboard server file not found!")
        print("💡 Please run this script from the services/dashboard directory")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed!")
        return 1
    
    # Start dashboard
    start_dashboard()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
