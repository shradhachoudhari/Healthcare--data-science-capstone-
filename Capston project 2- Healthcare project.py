#!/usr/bin/env python
# coding: utf-8

# In[54]:


#import the liabrary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn 
import matplotlib.pyplot as plt


# In[55]:


#load the dataset
dataset = pd.read_csv('/Users/shraddhalipane/Downloads/Project 2/Healthcare - Diabetes/health care diabetes.csv')


# In[56]:


#know dataset
data=pd.read_csv('/Users/shraddhalipane/Downloads/Project 2/Healthcare - Diabetes/health care diabetes.csv')


# In[57]:


# shape , info,describe , null values, neagtive values , head, tail


# In[58]:


data.shape


# In[59]:


data.describe()


# In[60]:


data.isnull().sum()


# In[61]:


data.head()


# In[62]:


data.info()


# In[63]:


plt.hist(data['Glucose'])


# In[64]:


data['Glucose'].value_counts().head(7)


# In[65]:


plt.hist(data['BloodPressure'])


# In[112]:


data['BloodPressure'].value_counts().head()


# In[67]:


plt.hist(data['SkinThickness'])


# In[68]:


plt.hist(data['Insulin'])


# In[69]:


plt.hist(data['BMI'])


# In[70]:


data['Glucose'].value_counts()


# In[71]:


data['BloodPressure'].value_counts()


# In[72]:


data['SkinThickness'].value_counts()


# In[73]:


data['Insulin'].value_counts()


# In[74]:


data['BMI'].value_counts()


# In[75]:


data.describe()


# In[76]:


data ['Outcome'].value_counts().head(7)


# In[77]:


plt.scatter(data['Glucose'],data['BloodPressure'])


# In[78]:


plt.scatter(data['BMI'],data['Insulin'])


# In[113]:


plt.scatter(data['BMI'],data['Glucose'])


# In[114]:


plt.scatter(data['Insulin'],data['Glucose'])


# In[115]:


plt.scatter(data['BloodPressure'],data['BMI'])


# In[116]:


plt.scatter(data['BloodPressure'],data['Insulin'])


# In[80]:


#heatmap
data.corr()


# In[81]:


import seaborn as sns


# In[82]:


sns.heatmap(data.corr())


# In[83]:


data.head()


# In[84]:


#model building
# Extract the features
features=data.iloc[:,0:8].values
label=data.iloc[:,8].values


# In[85]:


features


# In[86]:


label


# In[87]:


from sklearn.model_selection import train_test_split


# In[88]:


X_train,X_test,y_train,y_test = train_test_split(features,
                                                label,
                                                test_size=0.2,
                                                random_state =10)


# In[89]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train) 


# In[90]:


print(classifier.score(X_train,y_train))
print(classifier.score(X_test,y_test))


# In[91]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(label,classifier.predict(features))
cm


# In[92]:


from sklearn.metrics import classification_report
print(classification_report(label,classifier.predict(features)))


# In[93]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score     #ROC Curve


# In[94]:


probability = classifier.predict_proba(features) # predict probabilities


# In[95]:


probability = probability[:, 1] # probabilty for positive outcomes


# In[97]:


auc = roc_auc_score(label, probability) # calculating  AUC


# In[98]:


print('AUC: %.3f' % auc)


# In[100]:


fpr, tpr, thresholds = roc_curve(label, probability)    # ROC curve calculation 


# In[104]:


plt.plot([0, 1], [0, 1], linestyle='--')   
plt.plot(fpr, tpr, marker='.')


# In[105]:


from sklearn.metrics import classification_report
print(classification_report)


# In[106]:


print(classifier.score(X_train,y_train))
print(classifier.score(X_test,y_test))


# In[107]:


classification_report


# In[108]:


classifier.predict(X_test)


# In[110]:


from sklearn.metrics import classification_report
print(classification_report)


# In[ ]:




