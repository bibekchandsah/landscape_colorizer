# AI Image Colorizer - Deployment Ready

Transform grayscale images into vibrant colorful images using AI!

## Features

- File Upload: Drag & drop or browse files
- URL Input: Load images from web URLs  
- Clipboard Support: Paste images directly
- AI Colorization: Advanced GAN-based colorization
- Download Results: Save colorized images
- Responsive Design: Works on all devices

## Technology

- Model: Generative Adversarial Network (GAN)
- Framework: TensorFlow/Keras  
- Interface: Streamlit
- Processing: LAB colorspace conversion
- Deployment: Streamlit Cloud

## Best Results

- Use clear, high-contrast grayscale images
- Landscape and nature images work particularly well
- Avoid very dark or very bright images

## Local Development

```bash
# Clone repository
git clone YOUR_REPO_URL
cd YOUR_REPO_NAME

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app_deploy.py
```

## Deployment

This app uses a compressed TensorFlow Lite model (55.8 MB) that fits within GitHub's 100MB limit.

Main file for deployment: `app_deploy.py`

## License

This project is licensed under the MIT License.