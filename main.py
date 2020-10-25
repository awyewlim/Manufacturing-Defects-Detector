import os
import numpy as np
import cv2
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from skimage import transform
from PIL import Image

app = Flask(__name__)
model = load_model('casting_data/model1.h5')

@app.route('/')
def home():
    return render_template('index.html')

def get_filename():
    file = request.files['img']
    return file.filename

@app.route('/predict', methods=['POST'])
def predict():
    image = Image.open(request.files['img'])
    image = np.array(image).astype('float32')/255
    image = transform.resize(image, (300, 300, 1))
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    if(prediction < 0.5):
        cv2.putText(image, "def_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        output = 'Defect'
        
    else:
        cv2.putText(image, "ok_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        output = 'Ok'

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
