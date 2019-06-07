#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.svm import SVC

data = np.genfromtxt('group26_training.csv',delimiter=",", dtype="int32")
data1 = np.genfromtxt('group26_cv.csv',delimiter=",", dtype="int32")

X_train = data[:,:-1]
y_train = data[:,12288]

X_cv = data1[:,:-1]
y_cv = data1[:,12288]

X_train_scaled = (X_train / 255)
X_cv_scaled = (X_cv / 255)

cost = 2**(4)
yes = 2**(-8)
clf = SVC(C=cost,gamma=yes)
clf = clf.fit(X_train_scaled,y_train)

def getmodel():
    return clf


# In[ ]:




