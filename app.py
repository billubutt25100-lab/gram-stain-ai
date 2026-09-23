import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("gram_stain_model.h5")

st.set_page_config(
    page_title="Gram Stain AI",
    page_icon="🔬"
)

st.title("🔬 Gram Stain AI Classifier")

uploaded_file = st.file_uploader(
    "Upload microscope image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔬 Predict"):

        img = image.convert("RGB")
        img = img.resize((224,224))

        img = np.array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)

        score = prediction[0][0]

        if score > 0.5:
            result = "Gram Negative"
            confidence = score * 100
        else:
            result = "Gram Positive"
            confidence = (1-score) * 100

        st.success(f"Result: {result}")
        st.info(f"Confidence: {confidence:.2f}%")
