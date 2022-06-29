# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:06:55 2022

@author: Rozita
"""

# # run python app.py in python IDE
# Pickle will be used to read the model binary that was exported earlier, 
# and Flask will be used to create the web server

import pickle
import flask
import os

app = flask.Flask(__name__)
port = int(os.getenv("PORT", 9099))

# Import the model file that was created previously
model = pickle.load(open("linearmodel.pkl","rb"))

#Add a route that will allow you to send a JSON body of features and will return a prediction
@app.route('/predict', methods=['POST'])
def predict():
    features = flask.request.get_json(force=True)['features']
    prediction = model.predict([features])[0,0]
    response = {'prediction': prediction}

    return flask.jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)