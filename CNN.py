import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

# =========================
# 1. Paths
# =========================
train_path = "chest_xray/train"
test_path = "chest_xray/test"

# =========================
# 2. Data Augmentation (خفيف)
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# ⚠️ نسخة خفيفة لتجنب الكراش
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(128, 128),
    batch_size=16,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(128, 128),
    batch_size=16,
    class_mode='binary'
)

# =========================
# 3. Model (خفيف + قوي)
# =========================
model = models.Sequential([
    layers.Input(shape=(128,128,3)),

    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# =========================
# 4. Compile
# =========================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# 5. Training (خفيف)
# =========================
history = model.fit(
    train_generator,
    epochs=3,   # ⚠️ خففناها
    validation_data=test_generator
)

print("Training finished!")

# =========================
# 6. Save Model
# =========================
model.save("pneumonia_model.h5")
print("Model saved!")

# =========================
# 7. Test Image
# =========================
img = cv2.imread("image.jpg")

if img is None:
    print("Image not found!")
else:
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = img.reshape(1,128,128,3)

    prediction = model.predict(img)

    print("Prediction value:", prediction)

    if prediction[0][0] > 0.5:
        print("CNN Prediction: PNEUMONIA")
    else:
        print("CNN Prediction: NORMAL")