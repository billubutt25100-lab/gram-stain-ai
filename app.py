
import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("gram_stain_model.h5")


def predict(image):

    image = image.resize((224,224))

    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        result = "🦠 Gram Negative"
        confidence = prediction * 100
    else:
        result = "🧫 Gram Positive"
        confidence = (1-prediction) * 100

    return result, f"Confidence: {confidence:.2f}%"


with gr.Blocks(theme=gr.themes.Base()) as app:

    gr.Markdown(
    """
    # 🧬 Gram Stain AI Classifier

    Upload a Gram stain microscope image.
    """
    )

    image = gr.Image(type="pil", label="Upload Image")

    btn = gr.Button("🔬 Predict")

    result = gr.Textbox(label="Result")

    confidence = gr.Textbox(label="Confidence")


    btn.click(
        predict,
        inputs=image,
        outputs=[result, confidence]
    )


    gr.Markdown(
    """
    ---
    **Developed by Hasnan**
    """
    )


app.launch()
