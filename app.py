import zipfile
import io
import pathlib
from flask import Flask
from flask import request
from flask import json
import main

app = Flask(__name__)
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

def train():
        data = request.form.get('data')
        response = main.train(data)
        return response

def str_to_bool(s):
        if s=='True':
                return True
        elif s=='False':
                return False
        else:
                return 'valueError'

if __name__=='__main__':
        app.run()