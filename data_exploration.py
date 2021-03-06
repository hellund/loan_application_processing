# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:06:48 2021

@author: Eirik
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# Reading data and making dataframe
data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Making dataframes
df = pd.DataFrame(data)
test_df = pd.DataFrame(test_data)

# Shape
print('Shape of data: {}\n'.format(df.shape))
# Extract feature and target names
feature_names = list(df.columns[:-1])
print('Features:')
for name in feature_names:
    print(name)

target_names = df['Loan_Status'].unique()
print("\nTarget:\nLoan_status: {}".format(target_names))

cat = ['Gender', 'Married', 'Dependents', 'Education', 
                'Self_Employed', 'Property_Area','Credit_History',
                'Loan_Amount_Term']

# Plotting columns vs 'Loan_Status'
fig, axes = plt.subplots(4, 2, figsize = (13,13))
sns.set_theme(style="darkgrid")
for index, cat_col in enumerate(cat):
    row, col = index // 2, index % 2
    sns.countplot(x = cat_col, data = df, ax = axes[row, col], hue='Loan_Status')
    
plt.savefig('Categorical Columns vs Loan Status.png')

num = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

fig,axes = plt.subplots(1, 3, figsize = (13, 13))
for index , num_col in enumerate(num):
    sns.boxplot(y = num_col, data = df, x='Loan_Status', ax = axes[index])

plt.savefig('Numerical Columns vs Loan Status.png')

#Target mapping - No: 0, Yes = 1
target_mapping = {target: idx for idx, target in enumerate(np.unique(df['Loan_Status']))}
df['Loan_Status'] = df['Loan_Status'].map(target_mapping)

# Create dummies
df_encoded = pd.get_dummies(data=df, drop_first=True, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','Credit_History', 'Property_Area'])
test_df_encoded = pd.get_dummies(data=test_df, drop_first=True, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area'])

#Checking for NaN
print('NaN in train:\n_______________________\n{}'.format(df_encoded.isnull().sum()))

print('\nNaN in test:\n_______________________\n{}'.format(test_df_encoded.isnull().sum()))

# Statistics 
descr_stats = df_encoded.describe()
print('\n{}'.format(descr_stats))

# Creating histograms
df_encoded.hist(figsize = (13, 13))
plt.savefig('Histograms.png')
plt.show()

# Creating box plots
df_encoded.plot(kind = 'box', 
                 subplots = True, 
                 layout = (5, 3), 
                 sharex = False, 
                 sharey = False,
       figsize=(13, 13))

plt.savefig('Boxplots.png')
plt.show()

# Creating correlation matrix
correlation_matrix = df_encoded.corr()

fig, ax = plt.subplots(figsize=(13,10))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", ax=ax, vmin=-1, vmax=1)
plt.savefig('Correlation matrix.png')
