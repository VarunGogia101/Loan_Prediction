
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn import ensemble
import numpy as np
from scipy.stats import mode
from sklearn import preprocessing,model_selection

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

#load the dataset
data=pd.read_csv('/home/ubuntu/Desktop/LoanPrediction/train_data.csv')

def missing_number(x):
	
  return sum(x.isnull())

#impute missing values
data['Gender'].fillna(mode(list(data['Gender'])).mode[0], inplace=True)
data['Married'].fillna(mode(list(data['Married'])).mode[0], inplace=True)
data['Self_Employed'].fillna(mode(list(data['Self_Employed'])).mode[0], inplace=True)
# print (data.apply(missing_number, axis=0))

# impute mean for the missing value
data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
mapped_data={'0':0,'1':1,'2':2,'3+':3}
data = data.replace({'Dependents':mapped_data})
data['Dependents'].fillna(data['Dependents'].mean(), inplace=True)
data['Loan_Amount_Term'].fillna(method='ffill',inplace=True)
data['Credit_History'].fillna(method='ffill',inplace=True)
print (data.apply(missing_number,axis=0))

#convert the cateogorical data to numbers using the label encoder

var_mod = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    le.fit(list(data[i].values))
    data[i] = le.transform(list(data[i]))


#Train-test split
x=['Gender','Married','Education','Self_Employed','Property_Area','LoanAmount','Loan_Amount_Term','Credit_History','Dependents']
y=['Loan_Status']
print(data[x])

X_train,X_test,y_train,y_test=model_selection.train_test_split(data[x],data[y],test_size=0.2,random_state=42)

# Apply random forest classifier algorithm

clf=ensemble.RandomForestClassifier(n_estimators=200,max_features=3,min_samples_split=5,oob_score=True,n_jobs=-1,criterion='entropy')
clf.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred=clf.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))

#calculate accuracy
accuracy=clf.score(X_test,y_test)
print(accuracy)

