
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st


# Load the saved model
output_labels = ['vit K', 'vit C', 'vit D', 'vit B', 'vit A', 'vit E']
# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='my_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
def preprocess_image(image):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

st.title("Vitamin Deficiency Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("")

    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)

    # Run inference
    interpreter.invoke()

    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)

    # Make prediction
    predicted_label = output_labels[predicted_class]

    st.write(f"Predicted Vitamin Deficiency: {predicted_label}")
