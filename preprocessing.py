# -*- coding: utf-8 -*-
"""
Created on Mon May 31 21:41:36 2021

@author: Eirik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

# Reading data and making dataframe
data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Making dataframes
df = pd.DataFrame(data)
test_df = pd.DataFrame(test_data)



# Create dummies
df = pd.get_dummies(data=df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
test_df = pd.get_dummies(data=test_df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])

# Extract feature and target names
feature_names = df.columns[:-1]
print("Feature names: {}\n".format(feature_names))
target_names = df['Loan_Status'].unique()
print("Targets: {}\n".format(target_names))

#Target mapping - No: 0, Yes = 1
target_mapping = {target: idx for idx, target in enumerate(np.unique(df['Loan_Status']))}
df['Loan_Status'] = df['Loan_Status'].map(target_mapping)

# Changing Loan_ID for easy conversion from string to float:
df['Loan_ID'] = df['Loan_ID'].str.lstrip('LP')   
test_df['Loan_ID'] = test_df['Loan_ID'].str.lstrip('LP') 
                                    
#Checking for NaN
if df.isnull().sum().sum() == 0 and test_df.isnull().sum().sum() == 0:
    print('No NaN values')
else:
    print("There are NaN values!\n")
print('NaN in train:\n', df.isnull().sum())
print('_______________________')
print('NaN in test:\n', test_df.isnull().sum())
# Saving statistics of data
descr_stats = df.describe()
print(descr_stats)
descr_stats_test = test_df.describe()


# Normalizing data using MinMaxScaler
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
test_df = pd.DataFrame(scaler.fit_transform(test_df), columns = test_df.columns)

# Filling NaN data/Missing data with KNNImputer
imputer = KNNImputer(n_neighbors=4)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
test_df = pd.DataFrame(imputer.fit_transform(test_df), columns= test_df.columns)

print('NaN in train:\n', df.isnull().sum())
print('_______________________')
print('NaN in test:\n', test_df.isnull().sum())

# Save dataframes to csv
df.to_csv('preprocessed_train.csv', encoding='utf-8',index=None)
test_df.to_csv('preprocessed_test.csv', encoding='utf-8',index=None)