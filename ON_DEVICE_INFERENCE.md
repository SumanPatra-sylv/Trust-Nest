# On-Device Inference Path

## Model Export Status

| Format | Status | Size | Location |
|--------|--------|------|----------|
| PyTorch | ✅ Ready | 256 MB | `models/distilbert/model.safetensors` |
| **ONNX** | ✅ Ready | 255 MB | `models/distilbert/onnx/model.onnx` |
| TFLite | ⚠️ Not generated | - | - |

## ONNX Verification Results

```
Test Case 1: "Hi, I'll reach by 7 PM..." → SAFE (78%) ✅
Test Case 2: "Share OTP to receive..."   → SCAM (71%) ✅
Test Case 3: "Digital arrest..."         → SCAM (70%) ✅

Accuracy: 3/3 (100%)
```

---

## On-Device Integration Options

### Option 1: ONNX Runtime Mobile (Recommended for Android)

ONNX Runtime supports Android directly. No TFLite needed.

**Gradle dependency:**
```kotlin
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.0")
```

**Kotlin inference:**
```kotlin
class DistilBertOnnx(context: Context) {
    private val session: OrtSession
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    
    init {
        val modelBytes = context.assets.open("model.onnx").readBytes()
        session = env.createSession(modelBytes)
    }
    
    fun predict(inputIds: LongArray, attentionMask: LongArray): Float {
        val inputIdsT = OnnxTensor.createTensor(env, longArrayOf(1, inputIds.size.toLong()), inputIds)
        val maskT = OnnxTensor.createTensor(env, longArrayOf(1, attentionMask.size.toLong()), attentionMask)
        
        val inputs = mapOf(
            "input_ids" to inputIdsT,
            "attention_mask" to maskT
        )
        
        val output = session.run(inputs)
        val logits = (output[0].value as Array<FloatArray>)[0]
        
        // Softmax
        val expSum = logits.map { exp(it.toDouble()) }.sum()
        return (exp(logits[1].toDouble()) / expSum).toFloat()
    }
}
```

### Option 2: TFLite (Alternative)

TFLite conversion failed due to missing `onnx_tf` dependency. To manually convert:

```bash
pip install onnx-tf
python -m onnx_tf.convert -i models/distilbert/onnx/model.onnx -o models/distilbert/tf_saved_model
python -c "
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('models/distilbert/tf_saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open('models/distilbert/model.tflite', 'wb').write(tflite_model)
"
```

---

## Tokenization on Android

DistilBERT uses WordPiece tokenization. Options:

1. **HuggingFace Tokenizers (Rust)**: Use `tokenizers-android` library
2. **Pre-tokenize on server**: Send tokenized IDs to device
3. **Simple vocab lookup**: For demo, use simplified tokenization

**Vocab file**: `models/distilbert/onnx/vocab.txt` (30K tokens)

---

## Model Size Optimization

Current ONNX size: **255 MB** (too large for mobile)

**Quantization options:**
1. **Dynamic quantization**: ~64 MB
2. **ONNX Runtime quantization**:
   ```python
   from onnxruntime.quantization import quantize_dynamic
   quantize_dynamic("model.onnx", "model_quant.onnx")
   ```

---

## Recommended Architecture

```
Android App
├── Rule Engine (Kotlin)     ← Always runs first, <10ms
├── ONNX Runtime Mobile      ← For uncertain cases, ~200ms  
└── Backend Fallback (API)   ← If model not available
```

**Priority order:**
1. Rule Engine catches 80%+ of scams instantly
2. ONNX model handles remaining 20%
3. Backend API as last resort (requires internet)
