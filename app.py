# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:56:48 2021

@author: Eirik
"""

import streamlit as st
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np

test_df = pd.read_csv('test.csv')

st.title('Loan application')
st.markdown('Please fill in information below: ')
gender = st.selectbox('Gender: ',('Male', 'Female'))
married = st.selectbox('Married? ',('No', 'Yes'))
dependents = st.selectbox('How many are dependent on you? ',('0','1','2','+3'))
education = st.selectbox('Education: ',('Graduate', 'Not Graduate'))
self_employed = st.selectbox('Are you self employed? ',('No', 'Yes'))
slider_a_income = st.slider('What is your income?',value=3000, min_value=0, max_value=10000, step=100)
applicant_income = st.number_input('',value=slider_a_income)
slider_co_income = st.slider('What is your coapplicant`s income?' ,value=3500, min_value=0, max_value=10000, step=100)
coapplicant_income = st.number_input('',value=slider_co_income)
slider_loan_amount = st.slider('How much do u want to loan?',value=200, min_value=0, max_value=600, step=10)
loan_amount = st.number_input('',value=slider_loan_amount)
slider_loan_amount_term = st.slider('Term loan:', value=250, min_value=0, max_value=700, step=10)
loan_amount_term = st.number_input('',value=slider_loan_amount_term)
credit_history = st.selectbox('Your credit history: ',('0', '1'))
property_area = st.selectbox('Your property area? ',('Urban', 'Rural', 
                                                     'Semiurban'))

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

result = st.button('Check for loan approval')
if result:
    if pred_df.Loan_Status.iat[-1] == 'No':
        st.markdown('Your loan has been declined')
    else:
        st.markdown('Your loan has been approved')