from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ---------------------------
# Load the Segmentation Model
# ---------------------------
segmentation_model = tf.keras.models.load_model('final_brain_segmentation_model_v2.h5', compile=False)

# ---------------------------
# Preprocessing Function
# ---------------------------
def preprocess_image(image, target_size=(224, 224)):
    """
    Convert the image to RGB, resize it, normalize, and return a NumPy array.
    """
    image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

# ---------------------------
# Create the Flask App
# ---------------------------
app = Flask(__name__)

# Home route for quick testing
@app.route('/')
def home():
    return "Welcome to the Brain Tumor Segmentation API. Use the /predict endpoint for predictions."

# ---------------------------
# Predict Endpoint
# ---------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the image is provided in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    
    # Validate that the file is a PNG image
    if not file.filename.endswith('.png'):
        return jsonify({"error": "Only PNG images are supported."}), 400

    try:
        # Open the image directly from the file stream
        image = Image.open(file.stream)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        processed_image_exp = np.expand_dims(processed_image, axis=0)  # Add batch dimension
        
        # Predict the segmentation mask using the loaded model
        segmentation_pred = segmentation_model.predict(processed_image_exp)
        segmentation_mask = segmentation_pred[0]
        
        # Convert predictions to a binary mask using threshold 0.5
        segmentation_mask_binary = (segmentation_mask > 0.5).astype(np.uint8) * 255
        
        # Convert the mask to a list so it can be sent as JSON
        mask_list = segmentation_mask_binary.tolist()

        return jsonify({"mask": mask_list})
    except Exception as e:
        # Log the exception for debugging purposes
        print("Exception during prediction:", e)
        return jsonify({"error": f"Error processing the image: {str(e)}"}), 400

# ---------------------------
# Run the Flask App
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
