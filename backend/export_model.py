"""
Export DistilBERT to ONNX and TFLite
====================================
Creates mobile-optimized model for on-device inference.

Output:
- models/distilbert/model.onnx (ONNX format)
- models/distilbert/model.tflite (TFLite int8 quantized)
"""

import os
import json
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "distilbert")
ONNX_PATH = os.path.join(MODEL_DIR, "model.onnx")
TFLITE_PATH = os.path.join(MODEL_DIR, "model.tflite")


def export_to_onnx():
    """Export DistilBERT to ONNX using optimum."""
    print("\n" + "=" * 60)
    print("STEP 1: EXPORT TO ONNX")
    print("=" * 60)
    
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import DistilBertTokenizer
    
    print(f"[INFO] Loading model from {MODEL_DIR}")
    
    # Export using optimum
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        export=True,
        provider="CPUExecutionProvider"
    )
    
    # Save ONNX model
    onnx_save_dir = os.path.join(MODEL_DIR, "onnx")
    os.makedirs(onnx_save_dir, exist_ok=True)
    ort_model.save_pretrained(onnx_save_dir)
    
    # Also save tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(onnx_save_dir)
    
    print(f"[INFO] ONNX model saved to {onnx_save_dir}")
    
    # List files
    for f in os.listdir(onnx_save_dir):
        size = os.path.getsize(os.path.join(onnx_save_dir, f))
        print(f"       - {f} ({size / 1024 / 1024:.1f} MB)")
    
    return onnx_save_dir


def verify_onnx(onnx_dir: str):
    """Verify ONNX model works correctly."""
    print("\n" + "=" * 60)
    print("STEP 2: VERIFY ONNX")
    print("=" * 60)
    
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import DistilBertTokenizer
    import torch
    
    print("[INFO] Loading ONNX model for verification...")
    
    tokenizer = DistilBertTokenizer.from_pretrained(onnx_dir)
    model = ORTModelForSequenceClassification.from_pretrained(onnx_dir)
    
    # Test cases
    test_texts = [
        ("Hi, I'll reach by 7 PM. See you soon.", "SAFE"),
        ("Share OTP to receive your refund of Rs 5000", "SCAM"),
        ("Digital arrest. Stay on video call now.", "SCAM"),
    ]
    
    print("\n[TEST] Running inference on test cases...")
    
    correct = 0
    for text, expected in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        outputs = model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_class].item()
        
        label = "SCAM" if pred_class == 1 else "SAFE"
        status = "PASS" if label == expected else "FAIL"
        correct += 1 if label == expected else 0
        
        print(f"  [{status}] {label} ({confidence:.0%}) - {text[:40]}...")
    
    print(f"\n[RESULT] Accuracy: {correct}/{len(test_texts)}")
    
    return correct == len(test_texts)


def export_to_tflite(onnx_dir: str):
    """Convert ONNX to TFLite with int8 quantization."""
    print("\n" + "=" * 60)
    print("STEP 3: CONVERT TO TFLITE")
    print("=" * 60)
    
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf
    
    onnx_path = os.path.join(onnx_dir, "model.onnx")
    
    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX model not found at {onnx_path}")
        return None
    
    print(f"[INFO] Loading ONNX model from {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    
    print("[INFO] Converting ONNX to TensorFlow...")
    tf_rep = prepare(onnx_model)
    
    # Export to SavedModel
    saved_model_dir = os.path.join(MODEL_DIR, "tf_saved_model")
    tf_rep.export_graph(saved_model_dir)
    print(f"[INFO] TF SavedModel saved to {saved_model_dir}")
    
    # Convert to TFLite with quantization
    print("[INFO] Converting to TFLite with int8 quantization...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    
    tflite_model = converter.convert()
    
    # Save
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    
    size_mb = os.path.getsize(TFLITE_PATH) / 1024 / 1024
    print(f"[INFO] TFLite model saved to {TFLITE_PATH} ({size_mb:.1f} MB)")
    
    return TFLITE_PATH


def main():
    print("=" * 70)
    print("DISTILBERT MODEL EXPORT PIPELINE")
    print("=" * 70)
    
    # Step 1: Export to ONNX
    onnx_dir = export_to_onnx()
    
    # Step 2: Verify ONNX
    if not verify_onnx(onnx_dir):
        print("[WARN] ONNX verification had failures")
    
    # Step 3: Try TFLite conversion
    try:
        tflite_path = export_to_tflite(onnx_dir)
        if tflite_path:
            print("\n[SUCCESS] TFLite export complete!")
    except Exception as e:
        print(f"\n[WARN] TFLite conversion failed: {e}")
        print("[INFO] ONNX model is still available for use")
        print("[INFO] TFLite conversion can be done manually using:")
        print("       python -m tf2onnx.convert --opset 13 --onnx models/distilbert/onnx/model.onnx ...")
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    
    onnx_path = os.path.join(onnx_dir, "model.onnx")
    if os.path.exists(onnx_path):
        size = os.path.getsize(onnx_path) / 1024 / 1024
        print(f"  ONNX:   {onnx_path} ({size:.1f} MB)")
    
    if os.path.exists(TFLITE_PATH):
        size = os.path.getsize(TFLITE_PATH) / 1024 / 1024
        print(f"  TFLite: {TFLITE_PATH} ({size:.1f} MB)")
    else:
        print(f"  TFLite: Not generated (see warnings above)")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
