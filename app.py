import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL

# Load model and class names
model = load_model('plant_disease_model.h5')
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 
               'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 
               'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
               'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
               'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

# UI
st.title("Plant Disease Detection")
uploaded_file = st.file_uploader("Upload a plant leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    img = PIL.Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    pred = model.predict(img_array)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    st.markdown(f"Prediction: {predicted_class}")
    st.markdown(f"Confidence: {confidence:.2f}%")