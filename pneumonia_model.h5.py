import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# تحميل موديل Pneumonia
model = tf.keras.models.load_model("pneumonia_model.h5", compile=False)

def predict(image):
    try:
        image = image.convert("RGB")
        image = image.resize((128, 128))

        img = np.array(image) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]

        if prediction > 0.5:
            label = "Pneumonia"
            confidence = prediction * 100
        else:
            label = "Normal"
            confidence = (1 - prediction) * 100

        return f"{label} ({confidence:.2f}%)"

    except Exception as e:
        return f"Error: {str(e)}"


gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🫁 Pneumonia Detection AI",
    description="Upload a chest X-ray image"
).launch(share=True)