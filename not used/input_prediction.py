# -*- coding: utf-8 -*-
"""
Created on Mon May 31 23:56:41 2021

@author: Eirik
"""

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np

test_df = pd.read_csv('test.csv')


print('Please fill in information below: ')
gender = input('Gender: ')
married = input('Married? ')
dependents = input('How many are dependent on you? ')
education = input('Education: ')
self_employed = input('Are you self employed? ')
applicant_income = float(input('What is your income? '))
coapplicant_income = float(input('What is your coapplicant`s income? '))
loan_amount = float(input('Amount to loan: '))
loan_amount_term = float(input('Loan_amount_term: '))
credit_history = input('Your credit history: ')
property_area = input('Your property area? ')


new_data = {'Loan_ID':'LP002991',
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area}

new_df = pd.DataFrame.from_records(new_data, index=[0], columns=new_data.keys())


test_df = test_df.append(new_df, ignore_index=True)

test_df = pd.get_dummies(data=test_df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
test_df['Loan_ID'] = test_df['Loan_ID'].str.lstrip('LP') 
test_df.head()
# Normalizing data using MinMaxScaler
scaler = MinMaxScaler()
test_df = pd.DataFrame(scaler.fit_transform(test_df), columns = test_df.columns)

# Filling NaN data/Missing data with KNNImputer
imputer = KNNImputer(n_neighbors=4)
test_df = pd.DataFrame(imputer.fit_transform(test_df), columns= test_df.columns)

X_test = test_df.iloc[: , :].values

# Load model
filename = 'loan_application_model.sav'
model = pickle.load(open(filename, 'rb'))
y_pred = model.predict(X_test)

# Saving predictions
pred_df = pd.DataFrame(y_pred, columns = ['Loan_Status'])
pred_df['Loan_Status'] = np.where(pred_df['Loan_Status']==0, 'No', 'Yes')
print(pred_df.iloc[[-1]])