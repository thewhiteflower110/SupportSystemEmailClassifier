from flask import Flask
from flask import request
from flask import json
import main

app = Flask(__name__)
@app.route("/", methods=['POST'])
def execute():
        zipfile = request.form.get('file')
        bert = request.form.get('bert')
        lda = request.form.get('lda')
        combined = request.form.get('combined')
        #main.predict(zipfile,bert=False,lda=False,combined=True) #sample   
        main.predict(zipfile,bert=str_to_bool(bert),lda=str_to_bool(lda),combined=str_to_bool(combined))
        return method

def str_to_bool(s):
        if s=='True':
                return True
        elif s=='False':
                return False
        else:
                return 'valueError'

if __name__=='__main__':
        app.run()