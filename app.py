from flask import Flask, request, render_template, flash, url_for, redirect
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
# from PIL import Image
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model('bone_.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    try:
        img = load_img(img_path, target_size=(150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        prediction = model.predict(img_array)
        confidence = float(abs(prediction[0][0]))
        return 'Fracture' if prediction > 0.5 else 'No Fracture', confidence
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, None

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('home'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('home'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload PNG, JPG, or JPEG files.', 'error')
        return redirect(url_for('home'))
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        prediction, confidence = predict_image(filepath)
        
        if prediction is None:
            flash('Error processing image', 'error')
            return redirect(url_for('home'))
        
        return render_template('result2.html', 
                             prediction=prediction,
                             confidence=confidence,
                             img_path=filepath)
    
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
    
    