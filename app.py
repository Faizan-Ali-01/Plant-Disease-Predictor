import os
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model('plant_disease_prediction_model.h5')
print("Model loaded successfully!")

class_names = [
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
    'Tomato___healthy',
    'Unknown'
]

IMAGE_SIZE = 96
CONFIDENCE_THRESHOLD = 75.0

def predict(img):
    img = img.convert("RGB")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file found')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)
            actual_label = filename.split('_img')[0]
            img = tf.keras.preprocessing.image.load_img(
                filepath, 
                target_size=(IMAGE_SIZE, IMAGE_SIZE)
            )
            predicted_class, confidence = predict(img)
            if predicted_class == 'Unknown':
                return render_template(
                    'index.html',
                    error_message='⚠️ Unknown class detected. This may not be a plant leaf from our supported crops, or the image quality is poor. Please upload a clear image of: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, or Tomato leaves.'
                )
            if confidence < CONFIDENCE_THRESHOLD:
                return render_template(
                    'index.html',
                    error_message=f'⚠️ Low confidence ({confidence}%). The model is not confident about this prediction. Please upload a clearer, well-lit photo of a plant leaf.'
                )
            return render_template(
                'index.html',
                image_path=filepath,
                actual_label=actual_label,
                predicted_label=predicted_class,
                confidence=confidence
            )
    return render_template('index.html', message='Upload an image')

def allowed_file(filename):
    """
    Check if uploaded file has allowed extension
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)