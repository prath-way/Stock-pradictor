#!/usr/bin/env python3
"""
Startup script for the Stock Price Predictor
This script checks dependencies and starts the Flask application
"""

import sys
import subprocess
import importlib.util

def check_dependency(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_dependency(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def check_dependencies():
    """Check and install required dependencies"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask',
        'yfinance', 
        'tensorflow',
        'pandas',
        'numpy',
        'scikit-learn',
        'plotly',
        'matplotlib',
        'seaborn',
        'ta'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if not check_dependency(package):
            missing_packages.append(package)
            print(f"❌ Missing: {package}")
        else:
            print(f"✅ Found: {package}")
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_dependency(package):
                print(f"✅ Successfully installed {package}")
            else:
                print(f"❌ Failed to install {package}")
                return False
    
    print("✅ All dependencies are ready!")
    return True

def main():
    """Main startup function"""
    print("🚀 Stock Price Predictor - Startup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Failed to install required dependencies")
        print("💡 Try running: pip install -r requirements.txt")
        return
    
    print("\n🎯 Starting Flask application...")
    print("🌐 The web interface will be available at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        print("💡 Make sure all files are in the correct location")

if __name__ == "__main__":
    main() 