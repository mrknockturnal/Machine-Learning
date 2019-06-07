#!/usr/bin/env python
# coding: utf-8

# ## CSC311 – Machine Learning 2019 – GROUP 26 HAND SHAPE RECOGNITION 

# ### Phase 2 Testing and Results Generation

# ##### SOME INITIALISING CODE

# In[31]:


import numpy as np
from IPython.display import display,HTML

np.set_printoptions(suppress=True)

from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))


# ### Reading Data

# In[32]:


import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from PIL import Image
from scipy import ndimage
import train_svm
import log_reg
import train_nn

data = np.genfromtxt('group26_test.csv',delimiter=",", dtype="int32")


# In[33]:


#Set X and y for test
X_test = data[:,:-1]
y_test = data[:,12288]

# Scaling X
X_test_scaled = (X_test / 255)


# ### Support Vector Machine Technique

# #### Retrieving Model

# In[34]:


from sklearn.svm import SVC

svm_model = train_svm.getmodel()


# ### Results Generation

# #### Overall Accuracy

# In[35]:


y_pred = svm_model.predict(X_test_scaled)
svm_predictions = y_pred
accuracy = svm_model.score(X_test_scaled,y_test)


# In[36]:


accuracy = accuracy * 100

#print("The overall accuracy is "+ str(accuracy))


# ### Confusion Matrix

# In[37]:


from sklearn.metrics import classification_report, confusion_matrix
svm_matrix = confusion_matrix(y_test,y_pred)
#print(svm_matrix) 


# ### Testing Metrics

# In[38]:


#print(classification_report(y_test,y_pred)) 


# ### Per-Class Accuracy

# In[39]:


#for i in range(5):
#    print("Class "+ str(i) + " has " +  str(svm_matrix[i][i]/50*100) +" % Accuracy")


# ### Logistic Regression Technique

# #### Retrieving Model

# In[40]:


from sklearn.linear_model import LogisticRegression

logistic_model = log_reg.getmodel()


# ### Results Generation

# #### Overall Accuracy 

# In[70]:


y_pred = logistic_model.predict(X_test_scaled)
log_predictions = y_pred
accuracy = logistic_model.score(X_test_scaled,y_test)


# In[71]:


accuracy = accuracy * 100

#print("The overall accuracy is "+ str(accuracy))


# ### Confusion Matrix

# In[72]:


logistic_matrix = confusion_matrix(y_test,y_pred)
#print(logistic_matrix)


# ### Testing Metrics

# In[73]:


#print(classification_report(y_test,y_pred)) 


# ### Per-Class Accuracy

# In[74]:


#for i in range(5):
#    print("Class "+ str(i) + " has " +  str(logistic_matrix[i][i]/50*100) +" % Accuracy")


# ### Neural Networks Technique

# #### Retrieving Model

# In[75]:


from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor

nn_model = train_nn.getmodel()


# ### Results Generation

# #### Overall Accuracy 

# In[76]:


y_pred = nn_model.predict(X_test_scaled)
neural_predictions = y_pred
accuracy = nn_model.score(X_test_scaled,y_test)


# In[77]:


accuracy = accuracy * 100

#print("The overall accuracy is "+ str(accuracy))


# ### Confusion Matrix

# In[78]:


neural_matrix = confusion_matrix(y_test,y_pred)
#print(neural_matrix)


# ### Testing Metrics

# In[79]:


#print(classification_report(y_test,y_pred)) 


# ### Per-Class Accuracy

# In[80]:


#for i in range(5):
#    print("Class "+ str(i) + " has " +  str(neural_matrix[i][i]/50*100) +" % Accuracy")


# ### Additional Relevant Analysis

# In[ ]:





# In[ ]:





# #### Storing predictions of models for demo

# In[81]:


def getsvm():
    return svm_predictions

def getlog():
    return log_predictions

def getneural():
    return neural_predictions
    





