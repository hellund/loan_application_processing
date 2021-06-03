# -*- coding: utf-8 -*-
"""
Created on Mon May 31 23:10:23 2021

@author: Eirik
"""
import pandas as pd


# Load data
og_test_df = pd.read_csv('test.csv')
test_df = pd.read_csv('preprocessed_test.csv')

X_test = test_df.iloc[: , :].values

# Load model
filename = 'loan_application_model.sav'
model = pickle.load(open(filename, 'rb'))
y_pred = model.predict(X_test)

# Saving predictions
pred_df = pd.DataFrame(y_pred, columns = ['Loan_Status'])
pred_df = pred_df.join(og_test_df['Loan_ID'])
pred_df = pred_df.set_index('Loan_ID')
pred_df['Loan_Status'] = np.where(pred_df['Loan_Status']==0, 'No', 'Yes')
pred_df.to_csv('predictions.csv')
print(X_test.shape)