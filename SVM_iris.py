#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.svm import SVC


# In[2]:


url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['sepal-length','sepal-width','petal-length','petal-width','target']
df=pd.read_csv(url,names=names)
print(df)


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df['target'].value_counts()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


X=df.drop('target',axis=1)
y=df['target']
X.head()


# In[9]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=77,)


# In[10]:


X_train.shape


# In[11]:


X_test.shape


# In[12]:


y_train.shape


# In[13]:


y_test.shape


# In[14]:


linear_svm=SVC(kernel='linear',C=1,gamma=1) #keep changing c,gamma,kernel values(rbf,polynomial) for better accuracy 
linear_svm.fit(X_train,y_train)                   


# In[15]:


train_predictions=linear_svm.predict(X_train)
predictions=linear_svm.predict(X_test)
print(predictions)


# In[16]:


y_test  #check/compare predictions and y_test and we will find misclassifications so we have go for evaluation metrics


# In[17]:


cr=classification_report(y_test,predictions)
cm=confusion_matrix(y_test,predictions)
ac=accuracy_score(y_test,predictions)
print(ac)


# In[18]:


print(cm)


# In[19]:


print(cr)


# In[20]:


print(confusion_matrix(y_train,train_predictions))
print(classification_report(y_train,train_predictions))


# In[21]:


rbf_svm=SVC(kernel='rbf',C=1,gamma=1) #keep changing c,gamma,kernel values(rbf,polynomial) for better accuracy 
rbf_svm.fit(X_train,y_train) 
test_predictions=rbf_svm.predict(X_test)
train_predictions=rbf_svm.predict(X_train)
print(confusion_matrix(y_test,test_predictions))
print(classification_report(y_test,test_predictions))
print("Test Accuracy:",accuracy_score(y_test,test_predictions)*100)
print("Train Accuracy:",accuracy_score(y_train,train_predictions)*100)


# In[22]:


poly_svm=SVC(kernel='poly',C=1,gamma=0.1) #keep changing c,gamma,kernel values(rbf,polynomial) for better accuracy 
poly_svm.fit(X_train,y_train) 
test_predictions=poly_svm.predict(X_test)
train_predictions=poly_svm.predict(X_train)
print(confusion_matrix(y_test,test_predictions))
print(classification_report(y_test,test_predictions))
print("Test Accuracy:",accuracy_score(y_test,test_predictions)*100)
print("Train Accuracy:",accuracy_score(y_train,train_predictions)*100)


# In[23]:


from sklearn.model_selection import GridSearchCV


# In[25]:


param_grid_search={'C':[0.001,0.01,0.1,1,10,100,1000],'gamma':[0.01,1,0.001,0.0001,0.1],'kernel':['linear','rbf','poly']}
grid_search_svm=GridSearchCV(estimator=SVC(),param_grid=param_grid_search,scoring='accuracy',cv=5,n_jobs=-1)
grid_search_svm.fit(X,y)
print("Best parameters found using GridSearch:",grid_search_svm.best_params_)


# In[26]:


from sklearn.model_selection import RandomizedSearchCV
import numpy as np


# In[27]:


np.logspace(-3,3,20)


# In[28]:


param_random_search={'C':np.logspace(-3,3,20),'gamma':np.logspace(-3,3,20),'kernel':['linear','rbf','poly']}
random_search_svm=RandomizedSearchCV(estimator=SVC(),param_distributions=param_random_search,n_iter=500,scoring='accuracy',cv=5,random_state=42,verbose=1,n_jobs=-1)
random_search_svm.fit(X_train,y_train)
print("Best parameters found using GridSearch:",random_search_svm.best_params_)


# In[ ]:




