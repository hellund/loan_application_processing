# -*- coding: utf-8 -*-
"""
Created on Mon May 31 22:39:01 2021

@author: Eirik
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle

df = pd.read_csv('preprocessed_train.csv')

# Setting the y values (target)
y = df['Loan_Status'].values

# Setting up the x values (features)
X = df.iloc[: , :-1].values

# Split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=1, stratify=y)

# Pipeline with kernel
pipe_kernel = make_pipeline(KernelPCA(n_components=8, kernel='cosine'),
    RandomForestClassifier(random_state=1)
)

pipe_kernel.fit(X_train, y_train)
y_pred = pipe_kernel.predict(X_test)

print('Test Accuracy: %.3f' % pipe_kernel.score(X_test, y_test))

# Doing crossvalidation for kernel pipeline
param_grid = [{
'randomforestclassifier__n_estimators': [250, 300],
'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],
'randomforestclassifier__max_depth' : [6,7,8],
'randomforestclassifier__criterion' :['gini', 'entropy'],
'kernelpca__n_components': [ 7, 8, 9]}]

gs_kern = GridSearchCV(estimator=pipe_kernel, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=3,
                  n_jobs=-1)

gs_kern = gs_kern.fit(X_train, y_train)
print(gs_kern.best_params_)
print('Test accuracy: %.3f' % gs_kern.score(X_test, y_test))


# Creating confusion matrix
y_pred = gs_kern.predict(X_test)
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
gs_kern = gs_kern.fit(X, y)

# Saving model
filename = 'loan_application_model.sav'
pickle.dump(gs_kern, open(filename, 'wb'))

print(X.shape)
print(y.shape)