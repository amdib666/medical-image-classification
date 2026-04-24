import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# تحميل موديل جاهز
model = tf.keras.applications.MobileNetV2(weights="imagenet")

def predict(image):
    try:
        # تأكد الصورة RGB
        image = image.convert("RGB")

        # تغيير الحجم
        image = image.resize((224, 224))

        # تحويل إلى array
        img = np.array(image).astype("float32")

        # معالجة
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # prediction
        pred = model.predict(img)

        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(pred, top=1)[0][0]

        label = decoded[1]
        confidence = decoded[2] * 100

        return f"{label} ({confidence:.2f}%)"

    except Exception as e:
        return f"Error: {str(e)}"


# واجهة Gradio
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="AI Image Classifier",
    description="Upload any image and get prediction"
).launch(share=True)