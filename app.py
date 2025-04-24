import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# ---------------------------
# Load the Segmentation Model
# ---------------------------
segmentation_model = tf.keras.models.load_model(
    "final_brain_segmentation_model_v2.h5", compile=False
)

# ---------------------------
# Preprocessing Function
# ---------------------------
def preprocess_image(image, target_size=(224, 224)):
    """
    Resize and normalize the image.
    Converts image to RGB and scales pixel values to [0,1].
    """
    image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize
    return image

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.title("🧠 Brain Tumor Segmentation App")

st.markdown("""
Upload an **MRI scan**, and the model will generate a segmentation mask highlighting the tumor.
""")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)
    processed_image_exp = np.expand_dims(processed_image, axis=0)  # Shape: [1, 224, 224, 3]
    
    # ---------------------------
    # Segmentation Prediction
    # ---------------------------
    segmentation_pred = segmentation_model.predict(processed_image_exp)
    segmentation_mask = segmentation_pred[0]  # Shape: (224, 224, 1)
    
    # Convert to binary mask using threshold 0.5
    segmentation_mask_binary = (segmentation_mask > 0.5).astype(np.uint8) * 255
    
    # Squeeze to remove extra dimension (224, 224, 1) → (224, 224)
    segmentation_mask_binary = np.squeeze(segmentation_mask_binary)

    # Convert mask to green overlay
    mask_colored = np.zeros((224, 224, 3), dtype=np.uint8)  # Black background
    mask_colored[:, :, 1] = segmentation_mask_binary  # Apply green channel

    # Convert NumPy arrays to PIL images
    image_resized = image.resize((224, 224))  # Resize original image
    overlay = Image.blend(image_resized.convert("RGBA"), Image.fromarray(mask_colored, "RGB").convert("RGBA"), alpha=0.5)

    # Display results
    st.image(overlay, caption="Tumor Segmentation (Green Mask)", use_column_width=True)
