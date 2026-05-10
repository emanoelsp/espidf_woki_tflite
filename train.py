# Adaptado de:
# https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train.py

import math
import os

import numpy as np
import tensorflow as tf

# --- Flags ---
EPOCHS = 600          # original: 500
SAVE_DIR = "/tmp/hello_world_models"
SAVE_TF_MODEL = False

# --- Data ---
SAMPLES = 1500        # original: 1000

def get_data():
    np.random.seed(42)
    x = np.random.uniform(low=0, high=2 * math.pi, size=SAMPLES).astype(np.float32)
    y = np.sin(x).astype(np.float32)
    # Add small noise to help generalization
    y += 0.01 * np.random.randn(*y.shape).astype(np.float32)
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


def create_model():
    model = tf.keras.Sequential()
    # First layer: takes a scalar input and feeds it through 16 'neurons'.
    # The activation function is important; 'relu' will help the network
    # learn more complex shapes.
    model.add(tf.keras.layers.Dense(16, activation="relu", input_shape=(1,)))
    # The new second and third layer will help the network learn more complex
    # representations.
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    # Final single-value output
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse",
                  metrics=["mae"])
    return model


def train_model(model, x, y):
    model.fit(
        x,
        y,
        epochs=EPOCHS,
        batch_size=32,           # original: 64 — smaller batches improve generalization
        validation_split=0.2,
        verbose=1,
    )


def convert_tflite_model(model):
    """Converts the Keras model to TFLite format (int8 quantized)."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset for full-integer quantization
    x_rep = np.linspace(0, 2 * math.pi, 200).astype(np.float32)

    def representative_dataset():
        for val in x_rep:
            yield [np.array([val], dtype=np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    return tflite_model


def save_tflite_model(tflite_model, save_dir, model_name="model.tflite"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)
    with open(save_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {save_path}")
    print(f"Model size: {len(tflite_model)} bytes")
    return save_path


if __name__ == "__main__":
    print("Generating data...")
    x, y = get_data()

    print("Creating model...")
    model = create_model()
    model.summary()

    print(f"\nTraining for {EPOCHS} epochs...")
    train_model(model, x, y)

    print("\nConverting to TFLite (int8)...")
    tflite_model = convert_tflite_model(model)
    tflite_path = save_tflite_model(tflite_model, SAVE_DIR)

    if SAVE_TF_MODEL:
        tf_model_path = os.path.join(SAVE_DIR, "tf_model")
        model.save(tf_model_path)
        print(f"TF model saved to: {tf_model_path}")

    print("\nDone! To update model.cc, run:")
    print(f"  xxd -i {tflite_path} > main/model.cc")
    print("Then fix the variable name to 'g_model' and add 'alignas(8)' as before.")
