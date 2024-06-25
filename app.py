import cv2
import matplotlib as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

from flask import Flask,render_template,request
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/upload',methods=['POST'])
def predict():
    #return render_template('upload.html') 
    imagefile=request.files['imagefile']   
    image_path="./testimages/"+imagefile.filename
    imagefile.save(image_path)
    result=predict_new(image_path)
    return render_template('upload.html',result=result)
    

def predict_new(path):
    img = cv2.imread(path)
    RGBImg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    RGBImg= cv2.resize(RGBImg,(512,512))
    image = np.array(RGBImg) / 255.0
    model = keras.models.load_model('DensenetModel_3.h5')
    predict=model.predict(np.array([image]))
    return 'Normal' if predict[0][0] < 0.5 else 'Cataract'

if __name__=='__main__':
    app.run(port=3000,debug=True)                                                                                                                       