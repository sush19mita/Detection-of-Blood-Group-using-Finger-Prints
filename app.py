import streamlit as st
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load your trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/high_acc_model.h5")

model = load_model()

# Define class names
CLASS_NAMES = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Function to preprocess image
def preprocess_image(img):
    img = load_img(img, target_size=(64, 64))  # resize
    img_array = img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

# Streamlit UI
st.title("🩸 Blood Group Detection from Fingerprints")

st.write("Upload a fingerprint image to predict the blood group.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Blood Group"):
        try:
            # Preprocess
            img = preprocess_image(uploaded_file)

            # Predict
            predictions = model.predict(img)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = round(float(np.max(predictions[0])) * 100, 2)

            if predicted_class < len(CLASS_NAMES):
                predicted_label = CLASS_NAMES[predicted_class]
            else:
                predicted_label = "Unknown"

            # Show results
            st.success(f"### Predicted Blood Group: **{predicted_label}**")
            st.info(f"Confidence: {confidence}%")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
