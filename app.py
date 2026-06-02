import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_my_model():
    return load_model("models/final_densenet121.keras")

model = load_my_model()

# -------------------------------
# Class Labels
# -------------------------------
class_labels = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# -------------------------------
# Title
# -------------------------------
st.title("🌱 Plant Disease Detection System")

st.markdown("""
Upload a leaf image and the **DenseNet121** model will predict the disease.

**Best Model Performance**
- Accuracy: **95.44%**
- Architecture: **DenseNet121**
- Dataset: **PlantVillage RGB**
""")

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction Section
# -------------------------------
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded Leaf Image",
        width=400
    )

    img_resized = img.resize((224, 224))

    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array, verbose=0)[0]

    disease = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader("Prediction Result")

    if confidence < 50:
        st.warning(
            f"Low confidence prediction ({confidence:.2f}%). "
            "Please upload a clearer image."
        )
    else:
        st.success(f"Disease Detected: {disease}")

    st.info(f"Confidence Score: {confidence:.2f}%")

    # -------------------------------
    # Top 5 Predictions
    # -------------------------------
    st.subheader("Top 5 Predictions")

    top5_idx = np.argsort(prediction)[-5:][::-1]

    for rank, idx in enumerate(top5_idx, start=1):
        st.write(
            f"{rank}. {class_labels[idx]} — {prediction[idx] * 100:.2f}%"
        )

    # -------------------------------
    # Probability Bar Chart
    # -------------------------------
    st.subheader("Prediction Probabilities")

    chart_data = {
        class_labels[idx]: float(prediction[idx] * 100)
        for idx in top5_idx
    }

    st.bar_chart(chart_data)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "Developed by **Sumit Bombale** | MCA (AI & Data Science)"
)