# -*- coding: utf-8 -*-
"""
Created on Mon May 31 21:41:36 2021

@author: Eirik
"""

import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


# Reading data and making dataframe
data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Making dataframes
df = pd.DataFrame(data)
test_df = pd.DataFrame(test_data)

#Target mapping - N: 0, Y = 1
target_mapping = {target: idx for idx, target in 
                  enumerate(np.unique(df['Loan_Status']))}
df['Loan_Status'] = df['Loan_Status'].map(target_mapping)

# Create dummies
df_encoded = pd.get_dummies(data=df, drop_first=True, 
                            columns=['Gender', 'Married', 'Dependents', 
                                     'Education', 'Self_Employed',
                                     'Credit_History', 'Property_Area'])
test_df_encoded = pd.get_dummies(data=test_df, drop_first=True, 
                                 columns=['Gender', 'Married', 'Dependents', 
                                          'Education', 'Self_Employed', 
                                          'Credit_History', 'Property_Area'])

# Dropping Loan_ID
df_encoded.drop(['Loan_ID'], axis=1)

# Normalizing data using MinMaxScaler
scaler = MinMaxScaler()
df_encoded = pd.DataFrame(scaler.fit_transform(df_encoded), 
                          columns = df_encoded.columns)
test_df_encoded = pd.DataFrame(scaler.fit_transform(test_df_encoded), 
                               columns = test_df_encoded.columns)

# Filling NaN data/Missing data with KNNImputer
imputer = KNNImputer(n_neighbors=4)
df_encoded = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)
test_df_encoded = pd.DataFrame(imputer.fit_transform(test_df_encoded), columns= test_df_encoded.columns)

# Save dataframes to csv
df_encoded.to_csv('preprocessed_train.csv', encoding='utf-8',index=None)
test_df_encoded.to_csv('preprocessed_test.csv', encoding='utf-8',index=None)