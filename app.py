import sys
import os
from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from etl.feature_extractor import Items, Interactions
from etl.preprocessing import Preprocessor
from src.model import Model
import pickle
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route('/train')
def train():
    if not os.path.exists('./artifacts'):
        os.makedirs('./artifacts')
    model.train()
    return "Model trained"

@app.route('/predict/<user_id>', defaults={'recom_count': 100})
@app.route('/predict/<user_id>/<recom_count>')
def predict(user_id, recom_count):
    user_id = int(user_id)
    recom_count = int(recom_count)
    model.model_without_items = model.load_models('model_without_items')
    model.model_with_items = model.load_models('model_with_items')
    
    ### These should already be created with app launch
    # model.model_data.cate_enc_dict['visitorid'] = model.load_models('cate_enc_dict_visitorid')
    # model.model_data.cate_enc_dict['itemid'] = model.load_models('cate_enc_dict_itemid')
    # model.load_matrices('feature')
    
    final_result = model.get_predictions(recom_count,user_id,model.model_without_items)
    print("Success")
    return jsonify(repr(final_result))

@app.route('/predict_file', defaults={'recom_count': 100})
@app.route('/predict_file/<recom_count>')
def predict_file(recom_count):
    if not os.path.exists('./output'):
        os.makedirs('./output')
    recom_count = int(recom_count)
    model.model_without_items = model.load_models('model_without_items')
    model.model_with_items = model.load_models('model_with_items')
    
    ### These should already be created with app launch
    # model.model_data.cate_enc_dict['visitorid'] = model.load_models('cate_enc_dict_visitorid')
    # model.model_data.cate_enc_dict['itemid'] = model.load_models('cate_enc_dict_itemid')
    # model.load_matrices('feature')
    
    model.predict_file(recom_count,model.model_without_items)
    return "Success"

if __name__ == "__main__":
    global model
    model = Model()
    model.load_training_data()
    app.run(host='0.0.0.0')