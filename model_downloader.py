"""
Model downloader for Streamlit Cloud deployment
Downloads large model files from external storage
"""

import os
import requests
import streamlit as st
from pathlib import Path
import hashlib

# Model configuration
MODEL_CONFIG = {
    "landscape_colorization_gan_generator.keras": {
        "url": "YOUR_GOOGLE_DRIVE_LINK_HERE",  # Replace with actual link
        "size": 111.70,  # MB
        "md5": "your_model_md5_hash_here"  # Optional: for integrity check
    }
}

def download_file_from_google_drive(file_id, destination):
    """Download file from Google Drive"""
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def get_confirm_token(response):
    """Get confirmation token for large files"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Save downloaded content to file"""
    CHUNK_SIZE = 32768
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def download_from_url(url, destination, progress_bar=None):
    """Download file from direct URL with progress"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if progress_bar and total_size > 0:
                    progress_bar.progress(downloaded / total_size)

def verify_file_integrity(file_path, expected_md5):
    """Verify downloaded file integrity"""
    if not expected_md5:
        return True
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest() == expected_md5

@st.cache_data
def ensure_model_downloaded(model_name):
    """Ensure model is downloaded and available"""
    model_path = Path(model_name)
    
    if model_path.exists():
        st.success(f"✅ Model {model_name} already exists")
        return str(model_path)
    
    if model_name not in MODEL_CONFIG:
        st.error(f"❌ Model {model_name} not configured for download")
        return None
    
    config = MODEL_CONFIG[model_name]
    
    st.info(f"📥 Downloading {model_name} ({config['size']:.1f} MB)...")
    progress_bar = st.progress(0)
    
    try:
        # Download the model
        download_from_url(config['url'], model_path, progress_bar)
        
        # Verify integrity if MD5 provided
        if 'md5' in config and config['md5']:
            if verify_file_integrity(model_path, config['md5']):
                st.success(f"✅ {model_name} downloaded and verified successfully!")
            else:
                st.error(f"❌ {model_name} integrity check failed!")
                model_path.unlink()  # Delete corrupted file
                return None
        else:
            st.success(f"✅ {model_name} downloaded successfully!")
        
        return str(model_path)
        
    except Exception as e:
        st.error(f"❌ Failed to download {model_name}: {e}")
        if model_path.exists():
            model_path.unlink()  # Clean up partial download
        return None

def get_model_path(model_name):
    """Get path to model, downloading if necessary"""
    return ensure_model_downloaded(model_name)

# Alternative: Hugging Face Hub integration
def download_from_huggingface(repo_id, filename, cache_dir="."):
    """Download model from Hugging Face Hub"""
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    except ImportError:
        st.error("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        return None
    except Exception as e:
        st.error(f"❌ Failed to download from Hugging Face: {e}")
        return None