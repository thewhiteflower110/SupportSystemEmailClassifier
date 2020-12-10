from flask import Flask
from flask import request
from flask import json

app = Flask(__name__)
@app.route("/", methods=['POST'])
def execute():
        method = request.form.get('method')
        return method

if __name__=='__main__':
        app.run()