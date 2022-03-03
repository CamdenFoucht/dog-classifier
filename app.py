import urllib.request
import os
import torch
import pandas as pd
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pickle
import requests

from model import DogBreedPretrainedGoogleNet, to_device, get_default_device, device
from flask import Flask, render_template, request, flash, redirect, url_for
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets import ImageFolder
from collections import OrderedDict
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image


UPLOAD_FOLDER = 'static/uploads'


app = Flask(__name__)
app.secret_key = 'my-secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




breeds = ['Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih Tzu', 'Blenheim spaniel', 'papillon', 'toy terrier', 'Rhodesian ridgeback', 'Afghan hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black and tan coonhound', 'Walker hound', 'English foxhound', 'redbone', 'borzoi', 'Irish wolfhound', 'Italian greyhound', 'whippet', 'Ibizan hound', 'Norwegian elkhound', 'otterhound', 'Saluki', 'Scottish deerhound', 'Weimaraner', 'Staffordshire bullterrier', 'American Staffordshire terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire haired fox terrier', 'Lakeland terrier', 'Sealyham terrier', 'Airedale', 'cairn', 'Australian terrier', 'Dandie Dinmont', 'Boston bull', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer', 'Scotch terrier', 'Tibetan terrier', 'silky terrier', 'soft coated wheaten terrier', 'West Highland white terrier', 'Lhasa', 'flat coated retriever', 'curly coated retriever', 'golden retriever',
    'Labrador retriever', 'Chesapeake Bay retriever', 'German short haired pointer', 'vizsla', 'English setter', 'Irish setter', 'Gordon setter', 'Brittany spaniel', 'clumber', 'English springer', 'Welsh springer spaniel', 'cocker spaniel', 'Sussex spaniel', 'Irish water spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old English sheepdog', 'Shetland sheepdog', 'collie', 'Border collie', 'Bouvier des Flandres', 'Rottweiler', 'German shepherd', 'Doberman', 'miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard', 'Eskimo dog', 'malamute', 'Siberian husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond', 'Brabancon griffon', 'Pembroke', 'Cardigan', 'toy poodle', 'miniature poodle', 'standard poodle', 'Mexican hairless', 'dingo', 'dhole', 'African hunting dog']

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'DogBreedPretrainedGoogleNet':
            from model import DogBreedPretrainedGoogleNet
            return DogBreedPretrainedGoogleNet
        return super().find_class(module, name)

model = CustomUnpickler(open('model.pkl', 'rb')).load().cpu()



def model_prediction(predict_img):
    convert_tensor = transforms.ToTensor()
    img = convert_tensor(predict_img)
    xb = img.unsqueeze(0) # adding extra dimension
    xb = to_device(xb, device)
    preds = model(xb)  
    predictions = preds[0]
    max_val, kls = torch.max(predictions, dim=0)
    print('Predicted :', breeds[kls])
    return breeds[kls]



@app.route('/img-upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        print('No file in request')
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        print('No filename')
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        print('File exists')
        filename = secure_filename(file.filename)

        print('Saving file')
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        print('upload_image filename: ' + filename)

        predict_img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        prediction_text = model_prediction(predict_img)

        print('Predicted :',prediction_text)
        return render_template('index.html', filename=filename, prediction_text = prediction_text)

    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/predict',methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    url = request.form['url']
    print("Url is", url)

    convert_tensor = transforms.ToTensor()
    response = requests.get(url)
    predict_img = Image.open(BytesIO(response.content))
    prediction_text = model_prediction(predict_img)

    print('Predicted :', prediction_text)

    return render_template('index.html', prediction_text = prediction_text, image_url = url)


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


app.run()
