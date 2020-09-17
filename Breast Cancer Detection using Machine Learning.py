#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd


# In[14]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
columns = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape','marginal_adhesion','single_epithelial_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class']
df = pd.read_csv(url, names=columns)

#print the shape of dataset
print(df.shape)


# In[16]:


#Preprocess data
df.replace("?", -1, inplace=True)
print(df.axes)
#print the shape of dataset
print(df.shape)


# In[22]:


print(df.iloc[0])
print(df.describe())


# In[23]:


#plot histograms for each variable
0df.hist(figsize=(10,10))
plt.show()


# In[25]:


scatter_matrix(df,figsize=(18,18))
plt.show()


# In[56]:


#generate class data for training
X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)


# In[31]:


#testing options
seed=8
scoring='accuracy'


# In[57]:


models=[]
models.append(('KNN',KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM', SVC(gamma='auto')))

#Evaluate each model in turn
results=[]
names=[]

for name,model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_result=model_selection.cross_val_score(model,X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_result)
    names.append(name)
    print("%s : %f (%f)" % (name,cv_result.mean(), cv_result.std() ))


# In[58]:


#Predict with each model
for name,model in models:
    model.fit(X_train, Y_train)
    predictions=model.predict(X_test)
    print(name)
    print(accuracy_score(Y_test,predictions))
    print(classification_report(Y_test,predictions))


# In[ ]:




