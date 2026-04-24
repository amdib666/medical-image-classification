import tensorflow as tf

model = tf.keras.models.load_model(
    "pneumonia_model.h5",
    compile=False,
    safe_mode=False
)

model.save("fixed_model.keras")

print("✅ Done")