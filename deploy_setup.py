#!/usr/bin/env python3
"""
Deployment setup script for AI Image Colorizer
Prepares the app for Streamlit Cloud deployment
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def check_file_size(file_path):
    """Check file size in MB"""
    if not os.path.exists(file_path):
        return 0
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def setup_git_lfs():
    """Setup Git LFS for large files"""
    print("🔧 Setting up Git LFS...")
    
    try:
        # Check if Git LFS is installed
        subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
        print("✅ Git LFS is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git LFS not found. Please install it first:")
        print("   - Windows: Download from https://git-lfs.github.io/")
        print("   - Mac: brew install git-lfs")
        print("   - Linux: sudo apt install git-lfs")
        return False
    
    # Initialize Git LFS
    try:
        subprocess.run(["git", "lfs", "install"], check=True)
        print("✅ Git LFS initialized")
    except subprocess.CalledProcessError:
        print("⚠️ Git LFS already initialized")
    
    # Track large files
    large_files = ["*.keras", "*.h5", "*.tflite"]
    for pattern in large_files:
        try:
            subprocess.run(["git", "lfs", "track", pattern], check=True)
            print(f"✅ Tracking {pattern} with Git LFS")
        except subprocess.CalledProcessError:
            print(f"⚠️ {pattern} already tracked")
    
    return True

def create_deployment_files():
    """Create necessary deployment files"""
    print("📝 Creating deployment files...")
    
    # Create .streamlit directory
    streamlit_dir = Path(".streamlit")
    streamlit_dir.mkdir(exist_ok=True)
    
    # Create config.toml
    config_content = """
[server]
maxUploadSize = 200
enableCORS = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
"""
    
    with open(streamlit_dir / "config.toml", "w") as f:
        f.write(config_content.strip())
    print("✅ Created .streamlit/config.toml")
    
    # Update .gitignore
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Streamlit
.streamlit/secrets.toml

# Model files (if not using Git LFS)
# *.keras
# *.h5
# *.tflite

# Data
*.zip
*.tar.gz
dataset/
data/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    print("✅ Updated .gitignore")

def choose_deployment_strategy():
    """Help user choose the best deployment strategy"""
    print("\n🚀 Choosing deployment strategy...")
    
    # Check model file sizes
    original_size = check_file_size("landscape_colorization_gan_generator.keras")
    compressed_size = check_file_size("compressed_model.tflite")
    
    print(f"📊 Model sizes:")
    print(f"   Original model: {original_size:.1f} MB")
    print(f"   Compressed model: {compressed_size:.1f} MB")
    print(f"   GitHub limit: 100.0 MB")
    
    if compressed_size > 0 and compressed_size < 100:
        print("\n✅ RECOMMENDED: Use compressed model")
        print("   - File size under GitHub limit")
        print("   - No additional setup needed")
        print("   - Faster deployment")
        
        # Copy compressed app as main app
        if os.path.exists("app_compressed.py"):
            shutil.copy("app_compressed.py", "app_deploy.py")
            print("✅ Created app_deploy.py with compressed model")
        
        return "compressed"
    
    elif original_size < 100:
        print("\n✅ Use original model (under 100MB)")
        return "original"
    
    else:
        print("\n⚠️ Original model exceeds GitHub limit")
        print("📋 Available options:")
        print("   1. Git LFS (recommended)")
        print("   2. External hosting (GitHub Releases, Google Drive)")
        print("   3. Model compression")
        
        return "lfs_or_external"

def create_readme():
    """Create deployment README"""
    readme_content = """# 🎨 AI Image Colorizer

Transform grayscale images into vibrant colorful images using AI!

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL_HERE)

## ✨ Features

- 📁 **File Upload**: Drag & drop or browse files
- 🔗 **URL Input**: Load images from web URLs
- 📋 **Clipboard Support**: Paste images directly
- 🌈 **AI Colorization**: Advanced GAN-based colorization
- 📥 **Download Results**: Save colorized images
- 📱 **Responsive Design**: Works on all devices

## 🛠️ Technology

- **Model**: Generative Adversarial Network (GAN)
- **Framework**: TensorFlow/Keras
- **Interface**: Streamlit
- **Processing**: LAB colorspace conversion
- **Deployment**: Streamlit Cloud

## 🎯 Best Results

- Use clear, high-contrast grayscale images
- Landscape and nature images work particularly well
- Avoid very dark or very bright images

## 🔧 Local Development

```bash
# Clone repository
git clone YOUR_REPO_URL
cd YOUR_REPO_NAME

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content.strip())
    print("✅ Created README.md")

def main():
    """Main deployment setup function"""
    print("🎨 AI Image Colorizer - Deployment Setup")
    print("=" * 50)
    
    # Check current directory
    if not os.path.exists("app.py"):
        print("❌ app.py not found. Please run this script in the project directory.")
        return
    
    # Choose deployment strategy
    strategy = choose_deployment_strategy()
    
    # Create deployment files
    create_deployment_files()
    
    # Setup based on strategy
    if strategy == "compressed":
        print("\n🎯 Setting up for compressed model deployment...")
        print("✅ Ready to deploy! Use app_deploy.py as your main file.")
        
    elif strategy == "lfs_or_external":
        print("\n🎯 Setting up Git LFS...")
        if setup_git_lfs():
            print("✅ Git LFS setup complete!")
            print("📋 Next steps:")
            print("   1. git add .")
            print("   2. git commit -m 'Add model files with Git LFS'")
            print("   3. git push origin main")
            print("   4. Deploy on Streamlit Cloud")
        else:
            print("❌ Git LFS setup failed. Consider using external hosting.")
    
    # Create README
    create_readme()
    
    print("\n🎉 Deployment setup complete!")
    print("\n📋 Final checklist:")
    print("   ✅ Model files ready")
    print("   ✅ Configuration files created")
    print("   ✅ Git setup (if using Git LFS)")
    print("   ✅ README created")
    print("\n🚀 Ready to deploy to Streamlit Cloud!")

if __name__ == "__main__":
    main()