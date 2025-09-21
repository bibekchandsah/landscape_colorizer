#!/usr/bin/env python3
"""
Simple script to run the Streamlit colorization app
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    # Check if required model files exist
    required_files = [
        "landscape_colorization_gan_generator.keras",
        "landscape_colorization_checkpoint_generator.weights.h5",
        "landscape_colorization_checkpoint_discriminator.weights.h5"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure you have trained the model and the files are in the current directory.")
        return
    
    print("✅ All model files found!")
    print("🚀 Starting Streamlit app...")
    
    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")

if __name__ == "__main__":
    main()