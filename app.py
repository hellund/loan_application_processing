# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:56:48 2021

@author: Eirik
"""

import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np

# Reading data and making dataframe - Needed because of encoder
test_df = pd.read_csv('test.csv')

# Filling in information in streamlit
st.title('Loan application')
st.markdown('Please fill in information below: ')
gender = st.selectbox('Gender: ',('Male', 'Female'))
married = st.selectbox('Married? ',('No', 'Yes'))
dependents = st.selectbox('How many are dependent on you? ',('0','1','2','3+'))
education = st.selectbox('Education: ',('Graduate', 'Not Graduate'))
self_employed = st.selectbox('Are you self employed? ',('No', 'Yes'))
slider_a_income = st.slider('What is your income?',value=3000, min_value=0, 
                            max_value=10000, step=100)
applicant_income = st.number_input('',value=slider_a_income)
slider_co_income = st.slider('What is your coapplicant`s income?' ,value=3500, 
                             min_value=0, max_value=10000, step=100)
coapplicant_income = st.number_input('',value=slider_co_income)
slider_loan_amount = st.slider('How much do u want to loan?',value=200, 
                               min_value=0, max_value=600, step=10)
loan_amount = st.number_input('',value=slider_loan_amount)
slider_loan_amount_term = st.slider('How long is the term of the loan in months?', 
                                    value=360, min_value=0, max_value=700, step=10)
loan_amount_term = st.number_input('',value=slider_loan_amount_term)
credit_history = st.selectbox('Your credit history: ',('0', '1'))
property_area = st.selectbox('Your property area? ',('Urban', 'Rural', 
                                                     'Semiurban'))

# Saving information
new_data = {
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
        'Property_Area': property_area
        }

# Dropping Loan_ID
test_df = test_df.drop(['Loan_ID'], axis=1)

# Making dataframe with inputted data
new_df = pd.DataFrame.from_records(new_data, index=[0], columns=new_data.keys())

# Adding new data to old dataframe because of encoding
test_df = test_df.append(new_df, ignore_index=True)

# Encoding to match model
test_df_encoded = pd.get_dummies(data=test_df, drop_first=True, 
                                 columns=['Gender', 'Married', 'Dependents', 
                                          'Education', 'Self_Employed', 
                                          'Property_Area'])

# Normalizing data using MinMaxScaler
scaler = MinMaxScaler()
test_df_encoded = pd.DataFrame(scaler.fit_transform(test_df_encoded), 
                               columns = test_df_encoded.columns)

# Filling NaN data/Missing data with KNNImputer
imputer = SimpleImputer(strategy='median')
test_df_encoded = pd.DataFrame(imputer.fit_transform(test_df_encoded), 
                               columns= test_df_encoded.columns)

# Setting up X
X = test_df_encoded.values

# Load model
filename = 'loan_application_model.sav'
model = pickle.load(open(filename, 'rb'))
y_pred = model.predict(X)

# Saving predictions
pred_df = pd.DataFrame(y_pred, columns = ['Loan_Status'])
pred_df['Loan_Status'] = np.where(pred_df['Loan_Status']==0, 'No', 'Yes')

result = st.button('Check for loan approval')
if result:
    if pred_df.Loan_Status.iat[-1] == 'No':
        st.markdown('Your loan has been declined')
    else:
        st.markdown('Your loan has been approved')
