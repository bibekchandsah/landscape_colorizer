# 🎨 AI Image Colorizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

Transform your grayscale images into vibrant colorful images using advanced AI technology! This app uses a trained Generative Adversarial Network (GAN) to intelligently add realistic colors to black and white photos.

## ✨ Features

### 🖼️ **Multiple Input Methods**
- **📁 File Upload**: Drag & drop or browse for image files
- **🔗 URL Input**: Load images directly from web URLs
- **📋 Clipboard Support**: Paste images from your clipboard

### 🤖 **AI-Powered Colorization**
- **Advanced GAN Model**: Trained on landscape images for realistic results
- **LAB Color Space**: Uses perceptually uniform color space for better results
- **Real-time Processing**: Fast inference with optimized TensorFlow Lite model
- **Side-by-side Comparison**: View original and colorized images together

### 💾 **Export & Download**
- **High-Quality Output**: Download colorized images as PNG files
- **Preserve Resolution**: Maintains image quality during processing
- **Instant Download**: One-click download of results

### 🌐 **Cloud-Optimized**
- **Compressed Model**: 55.8 MB TensorFlow Lite model (50% size reduction)
- **Fast Deployment**: Optimized for Streamlit Cloud
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## 🚀 Live Demo

Try the app live: **[AI Image Colorizer](https://landscapecolorizer.streamlit.app/)**

## 📸 Screenshots

| Original Grayscale | AI Colorized |
|-------------------|--------------|
| ![Grayscale](https://via.placeholder.com/300x200/808080/FFFFFF?text=Grayscale+Input) | ![Colorized](https://via.placeholder.com/300x200/4CAF50/FFFFFF?text=AI+Colorized) |

## 🛠️ Technology Stack

- **🧠 AI Model**: Generative Adversarial Network (GAN)
- **🔧 Framework**: TensorFlow/Keras with TensorFlow Lite optimization
- **🖥️ Interface**: Streamlit for web application
- **🎨 Image Processing**: PIL, NumPy, scikit-image
- **☁️ Deployment**: Streamlit Cloud with optimized dependencies

## 📊 Model Details

| Specification | Details |
|---------------|---------|
| **Architecture** | U-Net Generator + PatchGAN Discriminator |
| **Training Data** | Landscape images dataset |
| **Original Size** | 111.7 MB |
| **Compressed Size** | 55.8 MB (TensorFlow Lite) |
| **Input Resolution** | 256x256 pixels |
| **Color Space** | LAB (L*a*b*) |
| **Inference Time** | ~2-3 seconds per image |

## 🎯 Best Results

The AI model works best with:

- ✅ **Landscape and nature scenes**
- ✅ **Clear, high-contrast images**
- ✅ **Well-lit photographs**
- ✅ **Images with distinct objects/subjects**

Less optimal for:
- ❌ Very dark or overexposed images
- ❌ Abstract or artistic images
- ❌ Images with unusual lighting conditions

## 🔧 Local Development

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/bibekchandsah/landscape_colorizer.git
cd ai-image-colorizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the model**
   - Ensure `compressed_model.tflite` is in the project root
   - Or train your own model using the provided notebooks

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open in browser**
   - Navigate to `http://localhost:8501`

### Development Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application |
| `compressed_model.tflite` | Optimized AI model |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System dependencies for cloud deployment |
| `optimize_grayscale_to_colorful.ipynb` | Model training notebook |

## 🚀 Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub**
```bash
git add .
git commit -m "Deploy AI Image Colorizer"
git push origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file: `app.py`
   - Click "Deploy"

### Local Docker Deployment

```bash
# Build Docker image
docker build -t ai-colorizer .

# Run container
docker run -p 8501:8501 ai-colorizer
```

## 📈 Performance Optimizations

- **Model Compression**: 50% size reduction using TensorFlow Lite
- **Caching**: Streamlit resource caching for model loading
- **Headless OpenCV**: Cloud-compatible image processing
- **Minimal Dependencies**: Reduced package conflicts
- **Efficient Processing**: Optimized image preprocessing pipeline

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Areas
- 🎨 UI/UX improvements
- 🧠 Model enhancements
- 🚀 Performance optimizations
- 📱 Mobile responsiveness
- 🌐 Internationalization

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Landscape images from various open sources
- **Architecture**: Based on pix2pix GAN architecture
- **Framework**: Built with Streamlit and TensorFlow
- **Inspiration**: Research in image-to-image translation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/bibekchandsah/landscape_colorizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bibekchandsah/landscape_colorizer/discussions)
- **Email**: your-email@example.com

## 🔗 Links

- **Live App**: [AI Image Colorizer](https://landscapecolorizer.streamlit.app/)
- **GitHub**: [Repository](https://github.com/bibekchandsah/landscape_colorizer)
- **Model Training**: [Jupyter Notebook](optimize_grayscale_to_colorful.ipynb)
- **Deployment Guide**: [STREAMLIT_CLOUD_FIX.md](STREAMLIT_CLOUD_FIX.md)

---

<div align="center">

**Made with ❤️ and AI**

Transform your memories from grayscale to colorful! 🌈

[⭐ Star this repo](https://github.com/bibekchandsah/landscape_colorizer) | [🐛 Report Bug](https://github.com/bibekchandsah/landscape_colorizer/issues) | [💡 Request Feature](https://github.com/bibekchandsah/landscape_colorizer/issues)

</div>