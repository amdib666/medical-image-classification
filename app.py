import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# تحميل موديل جاهز
model = tf.keras.applications.MobileNetV2(weights="imagenet")


def predict_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    image = Image.open(file_path).convert("RGB")
    image_resized = image.resize((224, 224))

    img_array = np.array(image_resized)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(prediction, top=1)[0][0]

    label = decoded[1]
    confidence = decoded[2] * 100

    result_label.config(text=f"{label} ({confidence:.2f}%)", fg="blue")

    img_display = ImageTk.PhotoImage(image.resize((300, 300)))
    image_label.config(image=img_display)
    image_label.image = img_display


# واجهة البرنامج
root = tk.Tk()
root.title("AI Image Classifier")

btn = tk.Button(root, text="Upload Image", command=predict_image)
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()