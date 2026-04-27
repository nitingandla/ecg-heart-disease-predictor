import tensorflow as tf
import tf2onnx
import numpy as np

# Load your keras model
model = tf.keras.models.load_model("ecg_best.keras")

# Convert to ONNX
input_signature = [tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)]
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

# Save
with open("ecg_best.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

print("✅ Conversion done! ecg_best.onnx created.")