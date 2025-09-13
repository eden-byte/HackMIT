#!/usr/bin/env python3
"""
Setup script for Video Emotion Analysis Backend
Run this to set up your development environment
"""

import os
import sys
import subprocess

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories"""
    dirs = [
        'uploads',
        'extracted/frames', 
        'extracted/audio',
        'results',
        'templates',
        'static'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python version {sys.version} is compatible")
    return True

def main():
    print("🚀 Setting up Video Emotion Analysis Backend")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create directories
    print("\n📁 Creating project directories...")
    create_directories()
    
    # Install system dependencies (FFmpeg)
    print("\n🔧 Checking system dependencies...")
    if sys.platform.startswith('darwin'):  # macOS
        print("📱 Detected macOS")
        if not run_command("which ffmpeg", "Checking FFmpeg"):
            print("Installing FFmpeg via Homebrew...")
            run_command("brew install ffmpeg", "Installing FFmpeg")
    elif sys.platform.startswith('linux'):  # Linux
        print("🐧 Detected Linux")
        if not run_command("which ffmpeg", "Checking FFmpeg"):
            print("Please install FFmpeg manually:")
            print("Ubuntu/Debian: sudo apt install ffmpeg")
            print("CentOS/RHEL: sudo yum install ffmpeg")
    elif sys.platform.startswith('win'):  # Windows
        print("🪟 Detected Windows")
        print("Please install FFmpeg manually from: https://ffmpeg.org/download.html")
    
    # Create virtual environment
    print("\n🐍 Setting up Python virtual environment...")
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return
    
    # Activate virtual environment and install dependencies
    print("\n📦 Installing Python dependencies...")
    if sys.platform.startswith('win'):
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return
    
    # Create environment file
    print("\n🔑 Creating environment configuration...")
    env_content = """# Environment Configuration for Video Emotion Analysis
# Copy this to .env and fill in your actual API key

ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: Adjust these settings
FLASK_ENV=development
FLASK_DEBUG=True
MAX_CONTENT_LENGTH=1073741824  # 1GB max file size
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_content)
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
    
    print("✅ Created .env.example and .env files")
    
    # Create a simple test script
    test_script = '''#!/usr/bin/env python3
"""
Simple test script to verify setup
"""
import cv2
import librosa
import whisper
import anthropic
from flask import Flask

def test_imports():
    print("Testing imports...")
    try:
        print("✅ OpenCV version:", cv2.__version__)
        print("✅ Librosa version:", librosa.__version__)
        print("✅ Whisper available")
        print("✅ Anthropic available")
        print("✅ Flask available")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_ffmpeg():
    print("Testing FFmpeg...")
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
            return True
        else:
            print("❌ FFmpeg not working properly")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg not found in PATH")
        return False

if __name__ == "__main__":
    print("🧪 Running setup verification tests...")
    print("=" * 40)
    
    all_good = True
    all_good &= test_imports()
    all_good &= test_ffmpeg()
    
    if all_good:
        print("\\n🎉 All tests passed! Your setup is ready.")
        print("\\nNext steps:")
        print("1. Add your Anthropic API key to .env file")
        print("2. Run: python app.py")
        print("3. Open http://localhost:5000 in your browser")
    else:
        print("\\n❌ Some tests failed. Please check the errors above.")
'''
    
    with open('test_setup.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created test_setup.py")
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Add your Anthropic API key to the .env file:")
    print("   ANTHROPIC_API_KEY=your-actual-api-key-here")
    print("\n2. Test your setup:")
    if sys.platform.startswith('win'):
        print("   venv\\Scripts\\python test_setup.py")
    else:
        print("   venv/bin/python test_setup.py")
    print("\n3. Start the server:")
    if sys.platform.startswith('win'):
        print("   venv\\Scripts\\python app.py")
    else:
        print("   venv/bin/python app.py")
    print("\n4. Open http://localhost:5000 in your browser")
    print("\n🚀 Happy hacking!")
if __name__ == "__main__":
    main()