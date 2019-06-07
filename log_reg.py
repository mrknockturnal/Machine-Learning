#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.linear_model import LogisticRegression

data = np.genfromtxt('group26_training.csv',delimiter=",", dtype="int32")
data1 = np.genfromtxt('group26_cv.csv',delimiter=",", dtype="int32")

X_train = data[:,:-1]
y_train = data[:,12288]

X_cv = data1[:,:-1]
y_cv = data1[:,12288]

X_train_scaled = (X_train / 255)
X_cv_scaled = (X_cv / 255)

clf = LogisticRegression(C = 10,random_state=1, solver='lbfgs', multi_class='multinomial', max_iter=1000)
clf = clf.fit(X_train_scaled,y_train)

def getmodel():
    return clf


# In[ ]:




