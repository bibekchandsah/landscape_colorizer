"""
Model compression utilities for deployment
Reduces model size while maintaining performance
"""

import tensorflow as tf
import numpy as np
import os
from pathlib import Path

def quantize_model(model_path, output_path, quantization_type='float16'):
    """
    Quantize model to reduce size
    
    Args:
        model_path: Path to original model
        output_path: Path for compressed model
        quantization_type: 'float16', 'int8', or 'dynamic'
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    if quantization_type == 'float16':
        # Convert to float16
        print("Converting to float16...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        
        # Save as .tflite
        tflite_path = output_path.replace('.keras', '.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Float16 model saved to {tflite_path}")
        
    elif quantization_type == 'int8':
        # Convert to int8 (requires representative dataset)
        print("Converting to int8...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        
        tflite_path = output_path.replace('.keras', '_int8.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Int8 model saved to {tflite_path}")
    
    else:  # dynamic quantization
        print("Applying dynamic quantization...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = output_path.replace('.keras', '_dynamic.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Dynamic quantized model saved to {tflite_path}")
    
    return tflite_path

def representative_data_gen():
    """Generate representative data for int8 quantization"""
    # Generate sample data matching your model's input shape
    for _ in range(100):
        yield [np.random.random((1, 256, 256, 1)).astype(np.float32)]

def prune_model(model_path, output_path, sparsity=0.5):
    """
    Prune model to reduce size
    
    Args:
        model_path: Path to original model
        output_path: Path for pruned model
        sparsity: Fraction of weights to prune (0.0 to 1.0)
    """
    try:
        import tensorflow_model_optimization as tfmot
        
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
        # Define pruning schedule
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=sparsity,
            begin_step=0,
            end_step=1000
        )
        
        # Apply pruning
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            model, pruning_schedule=pruning_schedule
        )
        
        # Compile the pruned model
        pruned_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Save pruned model
        pruned_model.save(output_path)
        print(f"Pruned model saved to {output_path}")
        
        return output_path
        
    except ImportError:
        print("❌ tensorflow-model-optimization not installed")
        print("Run: pip install tensorflow-model-optimization")
        return None

def compress_weights_only(model_path, output_path):
    """
    Save only model weights (smaller file)
    
    Args:
        model_path: Path to original model
        output_path: Path for weights file
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Save only weights
    weights_path = output_path.replace('.keras', '_weights.h5')
    model.save_weights(weights_path)
    print(f"Weights saved to {weights_path}")
    
    # Save model architecture separately
    architecture_path = output_path.replace('.keras', '_architecture.json')
    with open(architecture_path, 'w') as f:
        f.write(model.to_json())
    print(f"Architecture saved to {architecture_path}")
    
    return weights_path, architecture_path

def get_model_size(file_path):
    """Get model file size in MB"""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def main():
    """Compress the colorization model"""
    original_model = "landscape_colorization_gan_generator.keras"
    
    if not os.path.exists(original_model):
        print(f"❌ Model file {original_model} not found!")
        return
    
    original_size = get_model_size(original_model)
    print(f"Original model size: {original_size:.2f} MB")
    
    # Method 1: Float16 quantization
    print("\n🔄 Applying float16 quantization...")
    try:
        tflite_path = quantize_model(original_model, "compressed_model.keras", "float16")
        if os.path.exists(tflite_path):
            compressed_size = get_model_size(tflite_path)
            print(f"Compressed size: {compressed_size:.2f} MB")
            print(f"Size reduction: {((original_size - compressed_size) / original_size * 100):.1f}%")
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
    
    # Method 2: Weights only
    print("\n🔄 Extracting weights only...")
    try:
        weights_path, arch_path = compress_weights_only(original_model, "compressed_model.keras")
        if os.path.exists(weights_path):
            weights_size = get_model_size(weights_path)
            print(f"Weights size: {weights_size:.2f} MB")
            print(f"Size reduction: {((original_size - weights_size) / original_size * 100):.1f}%")
    except Exception as e:
        print(f"❌ Weights extraction failed: {e}")

if __name__ == "__main__":
    main()