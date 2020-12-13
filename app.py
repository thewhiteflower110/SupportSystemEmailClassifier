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



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "C:/Users/Dell/Desktop/New folder/SupportSystemEmailClassifier-main/Zip_File"

@app.route("/", methods=['POST'])
def execute():
        #method = request.form.get('method')
        zipfile = request.form.get('file')
        bert = request.form.get('bert')
        lda = request.form.get('lda')
        combined = request.form.get('combined')
        #main.predict(zipfile,bert=False,lda=False,combined=True) #sample   
        response = main.predict(app,zipfile,bert=str_to_bool(bert),lda=str_to_bool(lda),combined=str_to_bool(combined))
        return response

@app.route("/train", methods=['POST'])
def train_model():
        filename= request.form.get('filename')
        data = request.form.get('data')
        response = main.train(filename,data)
        return response

def str_to_bool(s):
        if s=='True':
                return True
        elif s=='False':
                return False
        else:
                return 'valueError'

	
@app.route('/uploader', methods = ['POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		filename = secure_filename(f.filename)
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		return 'file uploaded successfully'


if __name__=='__main__':
        app.run()
