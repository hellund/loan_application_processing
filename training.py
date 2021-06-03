# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 12:41:44 2021

@author: Eirik
"""

import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.stats import uniform, randint


from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb

# Reading data and making dataframe
df = pd.read_csv('preprocessed_train.csv')

# Setting the y values (target)
y = df['Loan_Status'].values

# Setting up the x values (features)
X = df.drop(['Loan_Status'], axis=1).values

# Split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=1, stratify=y)

model = xgb.XGBClassifier(objective="binary:logistic", random_state=1, use_label_encoder=False)

model.fit(X_train, y_train)


# Hyperparameter tuning
param_grid = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 150), # default 100
    "subsample": uniform(0.6, 0.4)
}

gs_model = RandomizedSearchCV(estimator=model, 
                  param_distributions=param_grid, 
                  scoring='roc_auc', 
                  cv=10,
                  n_jobs=-1,
                  n_iter=200,
                  verbose=1)

gs_model = gs_model.fit(X_train, y_train)
print(gs_model.best_params_)
print('Test accuracy: %.3f' % gs_model.score(X_test, y_test))

# Creating confusion matrix
y_pred = gs_model.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
plt.show()

# Training on all train data
gs_model = gs_model.fit(X, y)

# Saving model
filename = 'loan_application_model.sav'
pickle.dump(gs_model, open(filename, 'wb'))
