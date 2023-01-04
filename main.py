import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
#from marshmallow import Schema, fields, INCLUDE


app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')


model = pickle.load(open('model.pkl', 'rb'))


expected_features = ['experience', 'degree_percentage', 'post_graduation', 'iit_nit',
       'designation', 'aws_gcp_azure_ml_ops', 'tableau_powerbi']

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    request_json = {}
    request_value_list = [x for x in request.form.values()]
    experience = float(request_value_list[0])
    degree_percentage = float(request_value_list[1])
    post_graduation = 1 if str(request_value_list[2]).lower()=='yes' else 0
    iit_nit = 1 if str(request_value_list[3]).lower()=='yes' else 0

    designation = str(request_value_list[4]).lower()

    if designation in ["senior data scientist", "sr data scientist"]:
        designation = 1
    elif designation == "data scientist":
        designation = 0
    else:
        designation = -1

    aws_gcp_azure_ml_ops = 1 if str(request_value_list[5]).lower()=='yes' else 0
    tableau_powerbi = 1 if str(request_value_list[6]).lower()=='yes' else 0

    feature_values = [[experience, degree_percentage, post_graduation, iit_nit, designation,
                      aws_gcp_azure_ml_ops, tableau_powerbi]]

    predicted_salary = round(model.predict(np.array(feature_values))[0])
    return render_template('index.html', prediction_text='Greetings! Your expected monthly salary is {}'.format(predicted_salary))

if __name__ == "__main__":
    app.run(debug=True)
