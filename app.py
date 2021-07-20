import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask_api import status
from regression_surplus_income_v2 import output_prediciton
import json
import mysql.connector
from datetime import datetime
import os
import glob
import yaml



FILE_PATH = os.path.abspath('__file__')
CODE_PATH = os.path.dirname(FILE_PATH)
ROOT_PATH = os.path.dirname(CODE_PATH)
DATA_PATH = os.path.join(ROOT_PATH, 'Surplus_Income')
INPUT_PATH = os.path.join(ROOT_PATH, 'Input')
MODEL_PATH = os.path.join(ROOT_PATH, 'model')


app = Flask(__name__)

with open("Input\\app_config.yml") as file:
    app_config = yaml.load(file, Loader=yaml.FullLoader)
list_of_files = glob.glob(app_config['Dir_path'])
latest_file = max(list_of_files, key=os.path.getctime)
filename= latest_file.split("\\")[-1]
print(filename)

dict_keys_path = os.path.join(INPUT_PATH, 'dict_keys.json')

dict_keys = json.load(open(dict_keys_path))
#model = pkl.load(open(filename,'rb'))

app = Flask(__name__)
model1 = pickle.load(open(app_config['Read_path']+filename,'rb'))
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S")


from flask import Flask, render_template, request

def validation(data):
    pin_list = [1,2,3,4]
    errors = {}

    if (int(data["lead_id"]) < 0):
        errors['lead_id'] = 'lead_id should be a positive number'

    elif (float(data["number_of_bedrooms"]) < 0):
        errors['number_of_bedrooms'] = 'number_of_bedrooms should be a positive number'

    elif (float(data["family_members_count"]) < 0):
    	errors['family_members_count'] = 'family_members_count should be a positive value'

    elif data["type_of_residence"] not in list(dict_keys['type_of_residence'].keys()):
        errors['type_of_residence'] = 'Type of residence do not match with the list'

    elif (int(data["Score_pincode"]) not in pin_list):
        errors['score_pincode'] = 'Pin code do not match the defined list'

    elif data["type_of_employment"] not in list(dict_keys['type_of_employment'].keys()):
        errors['type_of_employment'] = 'type_of_employment does not match with the list'

    elif (float(data["house_value"]) < 0):
        errors['house_value'] = 'house_value should be a positive number'

    elif (float(data["vehicle_value"]) < 0):
    	errors['vehicle_value'] = 'vehicle_value should be a positive number'

    elif (float(data["emi_agriculure_loan"]) < 0):
    	errors['emi_agriculure_loan'] = 'emi_agriculure_loan should be either 0 or positive number'

    elif (float(data["emi_credit_card"]) < 0):
        errors['emi_credit_card'] = 'emi_credit_card should be either 0 or positive number'

    elif (float(data["emi_education_loan"]) < 0):
        errors['emi_education_loan'] = 'emi_education_loan should be either 0 or positive number'

    elif (float(data["emi_house_loan"]) < 0):
        errors['emi_house_loan'] = 'emi_house_loan should be either 0 or positive number'

    elif (float(data["emi_personal_loan"]) < 0):
        errors['emi_personal_loan'] = 'emi_personal_loan should be either 0 or positive number'

    elif (float(data["emi_vehicle_loan"]) < 0):
        errors['emi_vehicle_loan'] = 'should be either 0 or positive number'

    elif (float(data["provable_income_per_month"]) < 0):
        errors['provable_income_per_month'] = 'provable_income_per_month should be a positive number'

    elif (float(data["amount_health_insurance"]) < 0):
        errors['amount_health_insurance'] = 'amount_health_insurance should be either 0 or positive number'

    if errors != {}:
        response = {'status':'failed','data': {},'errors': errors, 'status_code':400}
        return False, response
    return True, status.HTTP_200_OK



@app.route('/surplus',methods=['POST'])
def surplus():
    headers = request.headers
    auth = headers.get("X-Api-Key")
    auth = app_config['Auth']
    print(auth)
    if auth == app_config['Auth']:
         print(jsonify({"message": "OK: Authorized"}), 200)
    else:
        return jsonify({"message": "ERROR: Unauthorized"})
    data = request.get_json(force=True)
    VALID_FLAG, response = validation(data)
    if not VALID_FLAG:
        return  response
    fin_features = [np.array(list(data.values())).tolist()]
    print(fin_features)
    prediction = model1.predict(output_prediciton(fin_features))
    output = prediction
    output = json.dumps(output.tolist()[0])
    output = int(round(float(output),0))
    database_update(fin_features,output)
    response = {'status':'success', 'data':{'surplus_income':output},'errors': {},'status_code': 200 }
    return jsonify(response)

def database_update(fin_features,output):
    output_list = [str(output)]
    extra_col = [str(0.0)]
    print(output_list)
    flag_col = [0, timestampStr ,0]
    #fin_features = str(fin_features)[1:-1]
    list = fin_features[0] + extra_col + output_list + flag_col
    print(list)
    cnx = mysql.connector.connect(user='root', password='root',host='127.0.0.1',database='surplus_income')
    cursor = cnx.cursor()
    cursor.execute("INSERT INTO test VALUES %r;" % (tuple(list),))
    cnx.commit()



if __name__ == "__main__":
    app.run(port=5000, debug=True)


