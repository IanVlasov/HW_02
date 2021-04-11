import os
import sys
sys.path.insert(0, "./")

import pickle
import numpy as np

from model.train_model import train_model
from flask import Flask, request, jsonify, Response, abort

app = Flask(__name__)


@app.route('/heart/<params>')
def heart(params):
    params = params.split(',')
    params = [float(num) for num in params]
    
    model_path = "./model/trained_model.pkl"
    if not os.path.exists(model_path):
        train_model()
    
    model = pickle.load(open(model_path, 'rb'))
    params = np.array(params).reshape(1, -1)
    predict = model.predict(params)
    
    return str(predict)


@app.route('/heart_post', methods=['POST'])
def predict_json():
    content = request.json
    
    try:
        params = content['sample']
    except KeyError:
        return abort(
            400,
            "Key 'sample' was not found in the request",
        )
        
    if isinstance(params, str):
        params = process_str_params(params)
    elif isinstance(params, (tuple, list)):
        params = process_collection_params(params)
    else:
        abort(
            400,
            "Invalid data for prediction. Expected 13 numerical objects",
        )
    
    model_path = "./model/trained_model.pkl"
    if not os.path.exists(model_path):
        train_model()
    
    model = pickle.load(open(model_path, 'rb'))
    predict = model.predict(params)
    predict = {'class': str(predict[0])}
    
    return jsonify(predict)


def process_str_params(params):
    params = params.split(',')
    try:
        params = [float(num.strip()) for num in params]
    except ValueError:
        abort(400, "Invalid data for prediction. Expected string with 13 numerical objects")
        
    params = np.array(params).reshape(1, -1)
    
    return params


def process_collection_params(params):
    if not all(isinstance(num, (int, float)) for num in params):
        abort(
            400, 
            "Invalid data for prediction. Expected 13 numerical objects")
        
    return np.array(params).reshape(1, -1)