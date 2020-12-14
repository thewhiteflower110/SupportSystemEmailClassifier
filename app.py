import zipfile
import io
import os
import pathlib
from flask import Flask
from flask import request
from flask import json
import main
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from flask_cors import CORS, cross_origin

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./Zip_File"

@app.route("/", methods=['POST'])
def execute():
        #method = request.form.get('method')
        zipfile = request.form.get('file')
        #bert = request.form.get('bert')
        #lda = request.form.get('lda')
        #combined = request.form.get('combined')
        #main.predict(zipfile,bert=False,lda=False,combined=True) #sample   
        #response = main.predict(app,zipfile,bert=str_to_bool(bert),lda=str_to_bool(lda),combined=str_to_bool(combined))
        response = main.predict(app,zipfile)
        return response

@app.route("/train", methods=['POST'])
def train_model():
        filename= request.form.get('filename')
        data = request.form.get('data')
        learning_rate = request.form.get('lr')
        epochs = request.form.get('epochs')
        response = main.train(filename,data, learning_rate, epochs)
        return response

@app.route('/uploader', methods = ['POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		filename = secure_filename(f.filename)
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		return 'file uploaded successfully'


if __name__=='__main__':
        app.run()