import streamlit as st
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# URL of the Flask API
FLASK_API_URL = "http://127.0.0.1:5000/predict"

st.title("🧠 Brain Tumor Segmentation App")
st.markdown("""
Upload an MRI scan, and the model will generate a segmentation mask highlighting the tumor.
""")

uploaded_file = st.file_uploader("Upload an MRI image", type=["png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Reset the file pointer to the beginning so that Flask can read the file
    uploaded_file.seek(0)
    
    # Prepare the file for sending to Flask API
    files = {"image": uploaded_file}
    
    try:
        # Send the image to Flask API for segmentation prediction
        response = requests.post(FLASK_API_URL, files=files)
    
        if response.status_code == 200:
            # Get the segmentation mask (as a list) from the response JSON
            result = response.json()
            mask_list = result.get('mask')
            if mask_list is not None:
                # Convert list to NumPy array
                mask_array = np.array(mask_list, dtype=np.uint8)
    
                # Squeeze extra channel if needed
                if mask_array.ndim == 3 and mask_array.shape[2] == 1:
                    mask_array = np.squeeze(mask_array, axis=2)
    
                # Create a green overlay from the mask
                mask_colored = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
                mask_colored[:, :, 1] = mask_array  # Set green channel
    
                # Resize original image to mask dimensions
                image_resized = image.resize((mask_array.shape[1], mask_array.shape[0]))
    
                # Blend the original image and the green overlay
                overlay = Image.blend(
                    image_resized.convert("RGBA"),
                    Image.fromarray(mask_colored, "RGB").convert("RGBA"),
                    alpha=0.5
                )
    
                st.image(overlay, caption="Predicted Tumor Mask Overlay", use_column_width=True)
            else:
                st.error("No mask returned from the API.")
        else:
            st.error(f"Error in predicting the tumor mask. Status Code: {response.status_code}. Response: {response.text}")
    
    except Exception as e:
        st.error(f"An exception occurred: {e}")
