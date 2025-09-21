import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage import color as skcolor
import io
import requests
import base64
from streamlit_paste_button import paste_image_button

# Import TensorFlow only when needed to avoid startup issues
tf = None

# Configure page
st.set_page_config(
    page_title="AI Image Colorizer",
    page_icon="🎨",
    layout="wide"
)

# Model configuration for external hosting
MODEL_URLS = {
    "landscape_colorization_gan_generator.keras": {
        "url": "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0/landscape_colorization_gan_generator.keras",
        "size_mb": 111.7,
        "backup_url": "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_FILE_ID"
    }
}

def download_model_if_needed(model_name):
    """Download model from external source if not present locally"""
    import os
    from pathlib import Path
    
    model_path = Path(model_name)
    
    # If model exists locally, use it
    if model_path.exists():
        return str(model_path)
    
    # Check if we have download configuration
    if model_name not in MODEL_URLS:
        st.error(f"❌ Model {model_name} not found locally and no download URL configured")
        return None
    
    config = MODEL_URLS[model_name]
    
    st.info(f"📥 Model not found locally. Downloading {model_name} ({config['size_mb']:.1f} MB)...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Try primary URL first
        url = config['url']
        status_text.text(f"Downloading from primary source...")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Download completed!")
        st.success(f"✅ Model {model_name} downloaded successfully!")
        
        return str(model_path)
        
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        
        # Try backup URL if available
        if 'backup_url' in config:
            st.info("🔄 Trying backup download source...")
            try:
                backup_response = requests.get(config['backup_url'], stream=True, timeout=30)
                backup_response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in backup_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                st.success(f"✅ Model downloaded from backup source!")
                return str(model_path)
                
            except Exception as backup_e:
                st.error(f"❌ Backup download also failed: {backup_e}")
        
        # Clean up partial download
        if model_path.exists():
            model_path.unlink()
        
        return None

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_colorization_model():
    """Load the trained GAN generator model"""
    global tf
    try:
        # Import TensorFlow here to avoid startup issues
        if tf is None:
            import tensorflow as tf_module
            tf = tf_module
        
        # Download model if needed
        model_path = download_model_if_needed("landscape_colorization_gan_generator.keras")
        if model_path is None:
            return None
        
        # Load the model
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess uploaded image for the model"""
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize image
    image_resized = cv2.resize(image, target_size)
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    return image_normalized, image_resized

def rgb_to_lab_l_channel(rgb_image):
    """Convert RGB image to L channel of LAB colorspace"""
    lab_image = skcolor.rgb2lab(rgb_image)
    l_channel = lab_image[:, :, 0] / 50.0 - 1.0  # Normalize to [-1, 1]
    return l_channel

def lab_to_rgb(l_channel, ab_channels):
    """Convert LAB channels back to RGB"""
    # Denormalize
    l_channel = (l_channel + 1.0) * 50.0
    ab_channels = ab_channels * 128.0
    
    # Combine channels
    lab_image = np.concatenate([l_channel[:, :, np.newaxis], ab_channels], axis=2)
    
    # Convert to RGB
    rgb_image = skcolor.lab2rgb(lab_image)
    
    # Clip to valid range
    rgb_image = np.clip(rgb_image, 0, 1)
    
    return rgb_image

def load_image_from_url(url):
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        st.error(f"❌ Error loading image from URL: {e}")
        return None

def colorize_grayscale_image(grayscale_image, model):
    """Colorize a grayscale image using the trained model"""
    # Normalize grayscale to [-1, 1] for L channel
    l_channel = grayscale_image * 2.0 - 1.0
    l_input = l_channel[np.newaxis, :, :, np.newaxis]
    
    # Generate AB channels
    generated_ab = model(l_input, training=False)
    
    # Convert back to RGB
    colorized_rgb = lab_to_rgb(l_channel, generated_ab[0])
    
    return colorized_rgb

def process_and_colorize_image(image, model):
    """Process and colorize an image"""
    if image is None:
        return None, None
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, caption="Input Image", use_container_width=True)
    
    # Process the image
    with st.spinner("🎨 AI is colorizing your image..."):
        try:
            # Preprocess image
            processed_image, resized_image = preprocess_image(image)
            
            # Colorize the image
            colorized_result = colorize_grayscale_image(processed_image, model)
            
            # Convert to PIL Image for display
            colorized_pil = Image.fromarray((colorized_result * 255).astype(np.uint8))
            
            with col2:
                st.subheader("🌈 Colorized Result")
                st.image(colorized_pil, caption="AI Colorized", use_container_width=True)
            
            # Success message
            st.success("✨ Colorization completed successfully!")
            
            # Download button
            buf = io.BytesIO()
            colorized_pil.save(buf, format='PNG')
            byte_im = buf.getvalue()
            
            st.download_button(
                label="📥 Download Colorized Image",
                data=byte_im,
                file_name="colorized_image.png",
                mime="image/png"
            )
            
            # Additional info
            st.info(f"Original size: {image.size} | Processed size: {colorized_pil.size}")
            
            return colorized_pil, byte_im
            
        except Exception as e:
            st.error(f"❌ Error during colorization: {e}")
            return None, None

def main():
    st.title("🎨 AI Image Colorizer")
    st.markdown("Transform your grayscale images into vibrant colorful images using AI!")
    
    # Load model
    model = load_colorization_model()
    
    if model is None:
        st.error("Cannot proceed without a loaded model. Please check if the model file exists.")
        return
    
    # Sidebar for instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        **Choose one of these input methods:**
        1. 📁 Upload a file
        2. 🔗 Enter image URL
        3. 📋 Paste from clipboard
        
        The AI will automatically colorize your grayscale image!
        """)
        
        st.header("ℹ️ About")
        st.markdown("""
        This app uses a trained GAN (Generative Adversarial Network) 
        to add realistic colors to grayscale images. The model was 
        trained on landscape images and works best with similar content.
        """)
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📁 Upload File", "🔗 Image URL", "📋 Clipboard"])
    
    image_to_process = None
    
    with tab1:
        st.markdown("### Upload an image file")
        uploaded_file = st.file_uploader(
            "Choose a grayscale image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a grayscale image to colorize",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
    
    with tab2:
        st.markdown("### Enter image URL")
        image_url = st.text_input(
            "Paste image URL here:",
            placeholder="https://example.com/image.jpg",
            help="Enter a direct link to an image file"
        )
        
        if image_url:
            if st.button("🔄 Load Image from URL", key="load_url"):
                with st.spinner("Loading image from URL..."):
                    image_to_process = load_image_from_url(image_url)
            
            # Auto-load if URL looks valid
            elif image_url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                with st.spinner("Loading image from URL..."):
                    image_to_process = load_image_from_url(image_url)
    
    with tab3:
        st.markdown("### Paste from clipboard")
        st.markdown("Click the button below to paste an image from your clipboard:")
        
        # Use streamlit-paste-button for clipboard functionality
        paste_result = paste_image_button(
            label="📋 Paste Image from Clipboard",
            key="paste_button",
            errors="ignore"
        )
        
        if paste_result.image_data is not None:
            try:
                image_to_process = paste_result.image_data
                st.success("✅ Image pasted successfully!")
            except Exception as e:
                st.error(f"❌ Error processing pasted image: {e}")
        
        st.markdown("---")
        
        # Alternative methods for clipboard
        st.markdown("**Alternative clipboard methods:**")
        
        # Method 1: Base64 text input
        with st.expander("📝 Paste Base64 Image Data"):
            paste_text = st.text_area(
                "Paste base64 image data here:",
                placeholder="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
                height=100,
                help="Copy image as base64 and paste here"
            )
            
            if paste_text:
                try:
                    if paste_text.startswith('data:image'):
                        # Handle data URL format
                        header, data = paste_text.split(',', 1)
                        image_data = io.BytesIO(base64.b64decode(data))
                        image_to_process = Image.open(image_data)
                        st.success("✅ Base64 image decoded successfully!")
                    elif paste_text.startswith('/9j/') or paste_text.startswith('iVBOR'):
                        # Handle raw base64
                        image_data = io.BytesIO(base64.b64decode(paste_text))
                        image_to_process = Image.open(image_data)
                        st.success("✅ Raw base64 image decoded successfully!")
                except Exception as e:
                    st.error(f"❌ Could not decode base64 image data: {e}")
        
        # Method 2: File drop zone
        st.markdown("**Or drag and drop an image below:**")
        dropped_file = st.file_uploader(
            "Drop image here",
            type=['png', 'jpg', 'jpeg'],
            key="clipboard_uploader",
            label_visibility="collapsed"
        )
        
        if dropped_file is not None:
            image_to_process = Image.open(dropped_file)
    
    # Process the image if we have one
    if image_to_process is not None:
        process_and_colorize_image(image_to_process, model)
    
    else:
        # Show example or placeholder
        st.info("👆 Please choose an input method above to get started!")
        
        # Add some example images or instructions
        st.markdown("### 💡 Tips for best results:")
        st.markdown("""
        - Use clear, high-contrast grayscale images
        - Landscape and nature images work particularly well
        - Avoid very dark or very bright images
        - The model works best with images similar to its training data
        
        ### 🔗 Example URLs to try:
        - `https://example.com/grayscale-landscape.jpg`
        - Any direct image URL ending in .jpg, .png, etc.
        """)

if __name__ == "__main__":
    main()