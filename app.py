import streamlit as st
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from tensorflow.keras.models import load_model

# --------------------------
# Config
# --------------------------
IMG_SIZE = (128, 128)
MODEL_PATH = "models/best_cnn_stego.keras"

# --------------------------
# High-pass residual (same as training)
# --------------------------
def high_pass_residual(img_array):
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    res = np.stack([convolve(img_array[:,:,c], kernel, mode='reflect') for c in range(3)], axis=2)
    # Normalize to 0-1 exactly like training
    res = (res - res.min()) / (res.max() - res.min() + 1e-7)
    return res

# --------------------------
# Embed LSB message
# --------------------------
def embed_message(image: Image.Image, message: str):
    arr = np.array(image).astype(np.float32)/255.0
    arr = high_pass_residual(arr)
    arr = np.expand_dims(arr, axis=0)
    flat = arr.flatten()
    binary_msg = ''.join(format(ord(c), '08b') for c in message) + "1111111111111110"
    if len(binary_msg) > len(flat):
        raise ValueError("Message too long for image")
    for i, bit in enumerate(binary_msg):
        flat[i] = (flat[i] & 0b11111110) | int(bit)
    arr2 = flat.reshape(arr.shape)
    return Image.fromarray(arr2)

# --------------------------
# Load model
# --------------------------
@st.cache_resource
def load_cnn_model(path):
    return load_model(path)

model = load_cnn_model(MODEL_PATH)

# --------------------------
# Streamlit UI
# --------------------------
st.title("🖼️ LSB Steganography Detector with Embed Option")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
message_input = st.text_area("Optional: Message to embed in image")

st.write("---")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
    st.image(image, caption="Original Image", use_column_width=True)

    # Step 1: Embed
    if st.button("Embed Message"):
        if not message_input.strip():
            st.warning("Please enter a message to embed.")
        else:
            image = embed_message(image, message_input)
            st.image(image, caption="Stego Image", use_column_width=True)
            st.success("✅ Message embedded successfully!")

    # Step 2: Predict
    if st.button("Predict"):
        arr = np.array(image)/255.0
        arr = high_pass_residual(arr)
        arr = np.expand_dims(arr, axis=0)  # batch dimension

        pred_prob = model.predict(arr)[0][0]
        label = "Stego (hidden message)" if pred_prob > 0.5 else "Cover (clean)"
        st.write(f"**Prediction:** {label}")
        st.write(f"Probability: {pred_prob:.4f}")
