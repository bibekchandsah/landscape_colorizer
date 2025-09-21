"""
AI Image Colorizer - Streamlit Cloud Optimized
Minimal dependencies, maximum compatibility
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io
import requests
import base64

# Configure page
st.set_page_config(
    page_title="AI Image Colorizer",
    page_icon="🎨",
    layout="wide"
)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_colorization_model():
    """Load the compressed TensorFlow Lite model"""
    try:
        import tensorflow as tf
        
        # Load TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path="compressed_model.tflite")
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        st.success("✅ Model loaded successfully!")
        return interpreter, input_details, output_details
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

def rgb_to_lab_simple(rgb_image):
    """Simple RGB to LAB conversion using numpy"""
    # Normalize RGB to [0, 1]
    rgb = rgb_image.astype(np.float32) / 255.0
    
    # Simple approximation of RGB to LAB conversion
    # This is a simplified version that works without scikit-image
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    # Convert to XYZ (simplified)
    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b
    
    # Convert to LAB (simplified)
    l = 116 * np.power(y, 1/3) - 16
    a = 500 * (np.power(x, 1/3) - np.power(y, 1/3))
    b_lab = 200 * (np.power(y, 1/3) - np.power(z, 1/3))
    
    return l, a, b_lab

def lab_to_rgb_simple(l_channel, ab_channels):
    """Simple LAB to RGB conversion"""
    # Denormalize
    l = (l_channel + 1.0) * 50.0
    a = ab_channels[:, :, 0] * 128.0
    b = ab_channels[:, :, 1] * 128.0
    
    # Convert LAB to XYZ (simplified)
    fy = (l + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    x = fx ** 3
    y = fy ** 3
    z = fz ** 3
    
    # Convert XYZ to RGB (simplified)
    r = 3.240479 * x - 1.537150 * y - 0.498535 * z
    g = -0.969256 * x + 1.875992 * y + 0.041556 * z
    b_rgb = 0.055648 * x - 0.204043 * y + 1.057311 * z
    
    # Stack and clip
    rgb = np.stack([r, g, b_rgb], axis=2)
    rgb = np.clip(rgb, 0, 1)
    
    return rgb

def preprocess_image_simple(image, target_size=(256, 256)):
    """Simple image preprocessing using PIL only"""
    # Convert to grayscale if needed
    if isinstance(image, Image.Image):
        if image.mode != 'L':
            image = ImageOps.grayscale(image)
        image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(image_resized)
    else:
        # Convert numpy array to PIL and back
        if len(image.shape) == 3:
            # Convert to grayscale
            image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        pil_image = Image.fromarray(image.astype(np.uint8))
        pil_resized = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(pil_resized)
    
    # Normalize to [0, 1]
    image_normalized = image_array.astype(np.float32) / 255.0
    
    return image_normalized, image_array

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

def colorize_image_tflite(grayscale_image, interpreter, input_details, output_details):
    """Colorize using TensorFlow Lite model"""
    # Prepare input
    l_channel = grayscale_image * 2.0 - 1.0
    l_input = l_channel[np.newaxis, :, :, np.newaxis].astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], l_input)
    interpreter.invoke()
    generated_ab = interpreter.get_tensor(output_details[0]['index'])
    
    # Convert back to RGB using simple conversion
    colorized_rgb = lab_to_rgb_simple(l_channel, generated_ab[0])
    
    return colorized_rgb

def process_and_display_image(image, model_data):
    """Process and display the colorized image"""
    if image is None or model_data[0] is None:
        return None, None
    
    interpreter, input_details, output_details = model_data
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, caption="Input Image", use_container_width=True)
    
    # Process the image
    with st.spinner("🎨 AI is colorizing your image..."):
        try:
            # Preprocess
            processed_image, _ = preprocess_image_simple(image)
            
            # Colorize
            colorized_result = colorize_image_tflite(
                processed_image, interpreter, input_details, output_details
            )
            
            # Convert to PIL Image
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
    model_data = load_colorization_model()
    
    if model_data[0] is None:
        st.error("❌ Cannot load the AI model. Please check if 'compressed_model.tflite' exists.")
        st.info("This app requires the compressed model file to function.")
        return
    
    # Show model info
    st.info("🚀 Using optimized TensorFlow Lite model for fast cloud deployment!")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 How to Use")
        st.markdown("""
        1. **Upload** a grayscale image
        2. **Or paste** an image URL
        3. **Watch** the AI colorize it
        4. **Download** your result!
        """)
        
        st.header("ℹ️ About")
        st.markdown("""
        This app uses a Generative Adversarial Network (GAN) 
        trained on landscape images to add realistic colors 
        to grayscale photos.
        
        **Best results with:**
        - Landscape photos
        - Nature scenes
        - Clear, high-contrast images
        """)
    
    # Input tabs
    tab1, tab2 = st.tabs(["📁 Upload File", "🔗 Image URL"])
    
    image_to_process = None
    
    with tab1:
        st.markdown("### Upload an image file")
        uploaded_file = st.file_uploader(
            "Choose a grayscale image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a grayscale image to colorize"
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
            if st.button("🔄 Load Image from URL"):
                with st.spinner("Loading image from URL..."):
                    image_to_process = load_image_from_url(image_url)
    
    # Process image if available
    if image_to_process is not None:
        process_and_display_image(image_to_process, model_data)
    else:
        # Show placeholder
        st.info("👆 Please upload an image or enter a URL to get started!")
        
        st.markdown("### 💡 Tips for best results:")
        st.markdown("""
        - Use clear, high-contrast grayscale images
        - Landscape and nature images work particularly well
        - Avoid very dark or very bright images
        - The AI works best with images similar to its training data
        """)

if __name__ == "__main__":
    main()