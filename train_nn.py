#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor

data = np.genfromtxt('group26_training.csv',delimiter=",", dtype="int32")
data1 = np.genfromtxt('group26_cv.csv',delimiter=",", dtype="int32")

X_train = data[:,:-1]
y_train = data[:,12288]

X_cv = data1[:,:-1]
y_cv = data1[:,12288]

X_train_scaled = (X_train / 255)
X_cv_scaled = (X_cv / 255)


model = MLPClassifier(hidden_layer_sizes=(104), max_iter=1000,random_state=1)  
model.fit(X_train_scaled, np.ravel(y_train))

predictions = model.predict(X_cv_scaled)
accuracies = model.score(X_cv_scaled,y_cv)

def getmodel():
    return model

