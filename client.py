import requests
from flask import jsonify
import json 

url = 'http://127.0.0.1:5000/surplus'


data = {"lead_id":"449","number_of_bedrooms":"3","amount_health_insurance":"0","family_members_count":"4","family_members_count_out_of_age_range":"0","type_of_employment":"goverment","type_of_residence":"owned","Score_pincode":"3","emi_agriculure_loan":"0","emi_credit_card":"167","emi_education_loan":"0","emi_house_loan":"0","emi_personal_loan":"0","emi_vehicle_loan":"0","house_value":"44.16","provable_income_per_month":"156042","vehicle_value":"0.0"}

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
r = requests.post(url, data=json.dumps(data), headers=headers)
if r.status_code==200:
    #resp = jsonify(success=True)
    print(r.json())
else:
    response = {'status':'failed', 'data': {},'errors':{'Something Went Wrong With The Process',},'status_code': 400}
    print(response)
    

#info = json.loads.decode("utf-8")