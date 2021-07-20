# import the required libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from math import *
import json
from sklearn.metrics import r2_score
import pickle
import mysql.connector
import yaml
from datetime import datetime
import time


#yaml file for all configurations
with open("Input\\db_details.yml") as file:
    db_details = yaml.load(file, Loader=yaml.FullLoader)

#Read the dictionary from the specified location
with open(db_details['Input_Path']) as f1:
    dict_keys=json.load(f1)

# To get the timstamp to save the pickle and actualvs predicted with timestamp extension
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d%m%Y%H%M%S")

#List of columns used for the model
cols = ['lead_id','number_of_bedrooms','amount_health_insurance','family_members_count','family_members_count_out_of_age_range','type_of_employment','type_of_residence','score_pincode','emi_agriculure_loan','emi_credit_card','emi_education_loan','emi_house_loan','emi_personal_loan','emi_vehicle_loan','house_value','provable_income_per_month','vehicle_value','Surplus_income']

#Read the input data from mysql database
def read_input_data():
    processed_df = None
    try:
       cnx = mysql.connector.connect(user=db_details['user_name'], password=db_details['password'],host=db_details['host'],database=db_details['database'])
       cursor = cnx.cursor()
       sql_update_surplus = "update historical_data set Surplus_income = predicted_surplus_income where retrain_flag =0 and Surplus_income =0"
       cursor.execute(sql_update_surplus)
       cnx.commit()
       sql_select_Query = "select * from historical_data"
       #sql_update_flag = "update  historical_data set retrain flag =1,retrain_date = current_date where flag=0"
       cursor.execute(sql_select_Query)
       columns = cursor.description
       result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cursor.fetchall()]
       processed_df = pd.DataFrame(result)
       processed_df = processed_df[cols]
    except Exception as e:
       print("error information", e)
    return processed_df

#use the dictionary and apply the categories for type of employment and type_of_residence
#find the EMI total and Asset total
def Replace_and_extract_new_features(processed_df):
    final_df = None
    try:
        for x in dict_keys.keys():
            processed_df[x] = processed_df[x].replace(dict_keys[x].keys(),dict_keys[x].values())
        processed_df1 = pd.get_dummies(processed_df.type_of_employment, prefix='employment')
        processed_df2 = pd.get_dummies(processed_df.type_of_residence, prefix='residence')
        processed_df3 = pd.get_dummies(processed_df.score_pincode, prefix='pincode')
        processed_df = processed_df.drop(['type_of_employment','type_of_residence','score_pincode'],axis =1)
        processed_df = processed_df.join(processed_df1)
        processed_df = processed_df.join(processed_df2)
        processed_df = processed_df.join(processed_df3)

        EMI_df = [col for col in processed_df.columns if  col.startswith('emi')]
        processed_df['emi_total'] =  processed_df.loc[:,EMI_df].sum(axis=1)

        Asset_df = ['house_value','vehicle_value']
        processed_df['Asset_total'] =  processed_df.loc[:,Asset_df].sum(axis=1)
        #Drop all the EMI and Asset related columns as we use only total columns for model
        processed_df = processed_df.drop(EMI_df,axis =1)
        final_df = processed_df.drop(Asset_df,axis =1)
    except Exception as e:
        print("error information", e)
    return final_df

#Build the model and caluculate actual vs predicted and also rmse value
def Regression_model(final_df):
    X=Y=model=predictions=actual_vs_predicted=None
    try:
        model_df = final_df.drop(['lead_id','Surplus_income'],axis =1)
        Y = final_df['Surplus_income']
        X = model_df
        #print(X.columns)
        x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
        reg = LinearRegression()     #initiating linearregression

        reg.fit(x_train.astype(float),y_train.astype(float))
        model= sm.OLS(y_train,x_train.astype(float)).fit()
        predictions = model.summary()
        y_pred = reg.predict(x_test)
        actual_vs_predicted = pd.DataFrame({'Act.ual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
        actual_vs_predicted = actual_vs_predicted.round(0)
        #print(actual_vs_predicted)
        print("r2_score",r2_score(y_test, y_pred))
        test_set_rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
        print(test_set_rmse)
    except Exception as e:
        print("error information", e)
    return X,Y,reg,model, predictions, actual_vs_predicted

# cross validation to check the rmse value using 10 folds
def Cross_validation(X,Y):
    scores = rmse = None
    try:
        crossvalidation = KFold(n_splits=10, random_state=None, shuffle=False)
        reg = LinearRegression()
        scores = cross_val_score(reg, X, Y, scoring="neg_mean_squared_error", cv=crossvalidation, n_jobs=1)
        rmse = sqrt(np.mean(np.abs(scores)))
        print("RMSE_kfolds", rmse)
    except Exception as e:
        print("error information", e)
    return scores,rmse

def Replace_and_extract_new_features_prediction(fin_df,final_cols):
    df = None
    try:
        # Created an empty dataframe to handle one hot encoding on prediction data with train data columns
        empty_df = pd.DataFrame(columns = final_cols)
        for col in empty_df.columns:
            empty_df[col].values[:] = 0
        print("empty dataframe created", empty_df)
        empty_df = empty_df.drop(['lead_id','Surplus_income'],axis =1)
        #merged the columns of empty_df with input data for prediction to get all the columns needed as per training data
        df = pd.merge(fin_df,empty_df, how='left', left_index=True,right_index=True, suffixes=('_x', ''))
        list_cols = df.loc[:, df.columns.str.endswith("_x")]
        list_cols = list_cols.columns.str.replace("_x", "")
        #print("after replacinf _x with empty",list_cols)
        df = df.drop(list_cols ,axis = 1)
        df = df.rename(columns = lambda x: x.strip('_x'))
        df = df.fillna(0)
        df = df[empty_df.columns]
    except Exception as e:
            print("error information", e)
    return df

# main part to call the functions and provide the results with model summary and predicitons
def main_part():
    model = predictions = actual_vs_predicted = final_df = None
    try:
        processed_df = read_input_data()
        final_df = Replace_and_extract_new_features(processed_df)
        #final_df.to_csv("D:\\LetsMD\\surplus_income\\input\\final_df.csv")
        X,Y,reg,model, predictions, actual_vs_predicted = Regression_model(final_df)
        scores,rmse = Cross_validation(X,Y)
        print("RMSE_kfolds", rmse)
    except Exception as e:
        print("error information", e)
    return model, predictions, actual_vs_predicted,final_df

#To predict the surplus income on new input
def output_prediciton(values):
    df = None
    try:
        cols1 = cols[:]
		#Remove surplus income column as this has to be predicted using pickle file generated
        cols1.remove('Surplus_income')
        #print("columns_list_updated",cols1)
        df_pred = pd.DataFrame(values, columns = cols1)
        list1 = list(df_pred.columns)
        list2 = ['type_of_employment', 'type_of_residence']
        res = [ i for i in list1 ]
        for i in list2:
            if i in list1:
                res.remove(i)

        df_prediction = df_pred[res].astype(float)
        df_pred = df_pred.drop(res,axis =1)
        df_pred = df_prediction.join(df_pred)
        final_df = Replace_and_extract_new_features(df_pred)
        #print("df_pred", df_pred)
        final_cols = ['lead_id', 'number_of_bedrooms', 'amount_health_insurance',
       'family_members_count', 'family_members_count_out_of_age_range',
       'provable_income_per_month', 'Surplus_income', 'employment_1',
       'employment_2', 'employment_3', 'employment_4', 'employment_5',
       'residence_1', 'residence_2', 'residence_3', 'residence_4',
       'residence_5', 'pincode_1', 'pincode_2', 'pincode_3', 'pincode_4',
       'emi_total', 'Asset_total']
        df = Replace_and_extract_new_features_prediction(final_df,final_cols)
    except Exception as e:
        print("error information", e)
    return df

#main fuinction processing to call the model and prediciton methods
if __name__ == "__main__":
    model, predictions, actual_vs_predicted, final_df =  main_part()

    filename = 'surplus_final_pickle_'+timestampStr+'.sav'
    pickle.dump(model, open(db_details['Output_Path']+filename, 'wb'))
    #Read the pickle file and use this for prediction
    loaded_model = pickle.load(open(db_details['Output_Path']+filename, 'rb'))
    print(predictions)
    print(actual_vs_predicted)
    actual_vs_predicted.to_csv(db_details['Output_Path']+'actual_vs_predicted_'+timestampStr+'.csv')
	#Input values for new customer
    values = [['439','3','3500000','6','3','unlisted','owned','3','0','1258','0','0','0','0','120','337103','1.8']]
    print("final_columns",final_df.columns)
	#predict the surplus Income with the input data provided using model.predict
    predicted_output = model.predict(output_prediciton(values))
    print(predicted_output)

