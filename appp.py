# app.py
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('plant_leaf_detection_model.h5')

# Function to make a prediction on an image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return "NON-HEALTHY" if prediction[0] > 0.5 else "HEALTHY"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = f"static/{file.filename}"
            file.save(file_path)
            result = predict_image(file_path)
            return render_template("result.html", prediction=result, image_path=file_path)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)